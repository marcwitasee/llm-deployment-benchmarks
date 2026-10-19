# Step 3: run every ticket in tickets.csv through each model setting, and score the answers.
# Run it from the 01-ticket-triage folder:  python step3_eval.py
#
# It reuses exactly what your step 2 scripts do, just wrapped in functions so we can call them in a loop.
# Results go to eval_results_<prompt version>.csv (one row per ticket per model), and a scorecard prints at the end.

import csv
import json
import os
import statistics
import time
from datetime import datetime

import anthropic
from dotenv import load_dotenv
from openai import OpenAI

from bench_log import log_result

load_dotenv()

# ---------------------------------------------------------------------------
# YOUR CHOICES: which model settings to test. Comment out a line to skip it.
# ---------------------------------------------------------------------------
CONFIGS = [
    # ("claude-opus-5-5", "low"),
    # ("claude-opus-5-5", "high"),
    # ("gpt-oss-120b", "low"),
    # ("gpt-oss-120b", "high"),
    # ("gpt-oss-20b", "low"),   # the untuned 20B: your "before" number for the fine-tuning experiment
    # ("YOUR-FINE-TUNED-MODEL-ID", "low"),   # paste the id Fireworks gives your trained model (the "after")
    # Qwen 3.5 9B runs only on a GPU you rent (a deployment), so these work only while that deployment is up.
    # Effort "none" turns Qwen's thinking off: it was trained to answer directly, and thinking ate all 2000 tokens.
    # Your deployment runs the base Qwen with your LoRA adapter on top. "model#deployment" picks which to call.
    # ("accounts/marcwitasee/models/ft-5m1sqm475pgvz#accounts/marcwitasee/deployments/lynfbcbt", "none"),   # fine-tuned Qwen (the "after")
    ("accounts/fireworks/models/qwen3p5-9b#accounts/marcwitasee/deployments/hbpxucw3", "none"),   # untuned Qwen (the "before"): needs its own deployment of the base model
]

# ---------------------------------------------------------------------------
# YOUR CHOICE: which prompt version to test. All versions live in prompts.py.
# Each version saves to its own file, e.g. eval_results_v3_structured.csv
# ---------------------------------------------------------------------------
from prompts import PROMPTS

PROMPT_VERSION = "v5_hf"
PROMPT_TEMPLATE = PROMPTS[PROMPT_VERSION]

# YOUR CHOICE: which ticket set to test. "tickets" = your 20 hand-written ones, "tickets_hf" = the Hugging Face sample.
TICKET_SET = "tickets_hf"
TICKETS_FILE = f"{TICKET_SET}.csv"
RESULTS_FILE = (
    f"eval_results_{PROMPT_VERSION}.csv" if TICKET_SET == "tickets"
    else f"eval_results_{TICKET_SET}_{PROMPT_VERSION}.csv"
)

# List prices, dollars per million tokens (input, output)
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "gpt-oss-120b": (0.15, 0.60),
    "gpt-oss-20b": (0.07, 0.30),  # PLACEHOLDER: check the real price on the model's Fireworks page
}

claude_client = anthropic.Anthropic()
fireworks_client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=os.environ["FIREWORKS_API_KEY"],
)


def call_claude(prompt, effort):
    """Your step2_claude_call.py, as a function. Returns answer text, timings and token counts."""
    start = time.perf_counter()
    first_token_at = None
    with claude_client.messages.stream(
        model="claude-opus-5-5",
        max_tokens=2000,
        output_config={"effort": effort},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            if first_token_at is None:
                first_token_at = time.perf_counter()
        final = stream.get_final_message()
    end = time.perf_counter()
    return {
        "answer": "".join(b.text for b in final.content if b.type == "text"),
        "ttft_s": (first_token_at or end) - start,
        "total_s": end - start,
        "input_tokens": final.usage.input_tokens,
        "output_tokens": final.usage.output_tokens,
    }


def call_fireworks(prompt, effort, model="gpt-oss-120b"):
    """Your step2_fireworks_call.py, as a function. Same outputs as call_claude."""
    start = time.perf_counter()
    first_answer_token_at = None
    usage = None
    answer = ""
    stream = fireworks_client.chat.completions.create(
        # Short names like "gpt-oss-20b" are Fireworks' public models; full ids ("accounts/...") are your own.
        model=model if model.startswith("accounts/") else f"accounts/fireworks/models/{model}",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort=effort,
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            if first_answer_token_at is None:
                first_answer_token_at = time.perf_counter()
            answer += delta.content
    end = time.perf_counter()
    return {
        "answer": answer,
        "ttft_s": (first_answer_token_at or end) - start,
        "total_s": end - start,
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
    }


def parse_answer(text):
    """Pull the JSON out of the model's reply. Returns (category, urgency), or ("PARSE_ERROR", ...) if it isn't valid."""
    try:
        start, end = text.index("{"), text.rindex("}") + 1  # ignore anything around the JSON, like ``` fences
        data = json.loads(text[start:end])
        return str(data.get("category", "")).lower().strip(), str(data.get("urgency", "")).lower().strip()
    except (ValueError, AttributeError):
        return "PARSE_ERROR", "PARSE_ERROR"


# ---------------------------------------------------------------------------
# The main loop: every setting x every ticket
# ---------------------------------------------------------------------------
with open(TICKETS_FILE, newline="") as f:
    tickets = list(csv.DictReader(f))

all_rows = []
for model, effort in CONFIGS:
    print(f"\n=== {model} / effort {effort} ===")
    call = call_claude if model.startswith("claude") else call_fireworks
    # A fine-tuned model has no per-token price: you pay for the GPU by the hour instead. Logged as $0 here.
    price_in, price_out = PRICES.get(model, (0.0, 0.0))

    for t in tickets:
        prompt = PROMPT_TEMPLATE.format(ticket=t["ticket"])
        try:
            r = call(prompt, effort) if call is call_claude else call(prompt, effort, model)
        except Exception as e:  # one failed request shouldn't kill the whole run
            print(f"  ticket {t['id']}: ERROR {e}")
            continue

        category, urgency = parse_answer(r["answer"])

        cat_ok = t['expected_category'] == category
        urg_ok = t['expected_urgency'] == urgency
        if cat_ok is None or urg_ok is None:
            raise SystemExit("TODO 1 isn't filled in yet: set cat_ok and urg_ok.")
        cost = r["input_tokens"] / 1e6 * price_in + r["output_tokens"] / 1e6 * price_out

        row = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "prompt_version": PROMPT_VERSION,
            "model": model,
            "effort": effort,
            "ticket_id": t["id"],
            "expected": f"{t['expected_category']}/{t['expected_urgency']}",
            "got": f"{category}/{urgency}",
            "category_correct": cat_ok,
            "urgency_correct": urg_ok,
            "ttft_s": round(r["ttft_s"], 3),
            "total_s": round(r["total_s"], 3),
            "input_tokens": r["input_tokens"],
            "output_tokens": r["output_tokens"],
            "cost_usd": round(cost, 6),
            "raw_answer": r["answer"],
        }
        log_result(row, path=RESULTS_FILE)
        all_rows.append(row)

        mark = "✓" if (cat_ok and urg_ok) else "✗"
        print(f"  {mark} ticket {t['id']:>2}: expected {row['expected']:<18} got {row['got']}")

# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------
print(f"\n\nSCORECARD  (prompt: {PROMPT_VERSION}, tickets: {TICKETS_FILE})")
print(f"{'setting':<26}{'category':>9}{'urgency':>9}{'both':>7}{'median TTFT':>13}{'cost/1M tickets':>17}")
for model, effort in CONFIGS:
    rows = [r for r in all_rows if r["model"] == model and r["effort"] == effort]
    if not rows:
        continue
    n = len(rows)

    cat = sum(r["category_correct"] for r in rows)
    urg = sum(r["urgency_correct"] for r in rows)
    both = sum(1 for r in rows if r["category_correct"] and r["urgency_correct"])
    if None in (cat, urg, both):
        raise SystemExit("TODO 2 isn't filled in yet: count cat, urg and both.")
    ttft = statistics.median(r["ttft_s"] for r in rows)
    cost_per_million = statistics.mean(r["cost_usd"] for r in rows) * 1_000_000
    print(f"{model + ' / ' + effort:<26}{cat:>5}/{n:<3}{urg:>5}/{n:<3}{both:>3}/{n:<3}{ttft:>11.2f}s{'$' + format(cost_per_million, ',.0f'):>17}")

print(f"\nFull details are in {RESULTS_FILE}. Look at the ✗ rows: is the model wrong, or is the expected answer debatable?")
