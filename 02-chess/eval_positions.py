# Baseline: before training anything, how good are today's models at picking a chess move?
# Every model setting sees the same test positions (from test_positions.csv) and the same prompt
# that the training data uses, and we score each answer.
#
# Run it from your chess folder:  python eval_positions.py
# Results go to eval_results.csv (one row per position per model), and a scorecard prints at the end.
#
# Three scores per model, from easiest to hardest to earn:
#   legal  - the move is a real, legal move in that position (python-chess checks this)
#   good   - the engine's best move, OR another move the engine listed that is at most
#            GOOD_MARGIN_CP worse. Like partial credit: a move that's nearly as strong still counts.
#   best   - exactly the engine's top move

import csv
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime

import anthropic
import chess
from dotenv import load_dotenv
from openai import OpenAI

from chess_prompts import PROMPT, PROMPT_WITH_LEGAL_MOVES, legal_moves_text

sys.path.append("..")  # fallback; bench_log.py is also in this folder
from bench_log import log_result

load_dotenv("../.env")  # your API keys, in the repository root

# ---------------------------------------------------------------------------
# YOUR CHOICES
# ---------------------------------------------------------------------------
CONFIGS = [
    # Qwen 3.5 9B (the model you'll fine-tune) runs on a deployment you rent, billed while it's up.
    # It goes first so you can delete the deployment as soon as its rows finish.
    # Effort "none" turns its thinking off, as in the ticket project.
    # ("accounts/fireworks/models/qwen3p5-9b#accounts/marcwitasee/deployments/hlp69f2z", "none"),   # done 2026-10-18
    # ("claude-opus-5-5", "low"),   # done 2026-10-18
    # ("claude-opus-5-5", "high"),   # done 2026-10-18
    # ("gpt-oss-120b", "low"),   # done 2026-10-18
    # ("gpt-oss-120b", "high"),   # dropped 2026-10-18: never answered, even with 16,000 tokens (~85s per move)
    # ("accounts/marcwitasee/models/ft-5cua8vpi78cmb#accounts/marcwitasee/deployments/ga3i4qip", "none"),   # 14k fine-tune, done 2026-10-18
    # ("accounts/marcwitasee/models/ft-s4ety4h6hkltk#accounts/marcwitasee/deployments/qbz6vzj0", "none"),   # 73k fine-tune, done 2026-10-19
    # ("accounts/marcwitasee/deployments/nfi6x4bj", "low"),   # Glimmer 30B untuned, 50 positions, done 2026-10-22 (default effort never finished thinking)
    # Glimmer 30B fine-tune: paste "accounts/marcwitasee/models/<ft-id>#accounts/marcwitasee/deployments/<id>" below.
    # Glimmer's lowest reasoning setting is "low" (no "none").
    # ("accounts/marcwitasee/models/ft-0uzen4wm2ec3x#accounts/marcwitasee/deployments/xrud2iwq", "low"),   # Glimmer 30B fine-tune (73k), done 2026-10-24
    # Legal-move-list experiment (GIVE_LEGAL_MOVES = True below): serverless models, no deployment needed.
    # ("gpt-oss-120b", "low"),   # legal list, done 2026-10-24
    # ("claude-opus-5-5", "low"),   # legal list, done 2026-10-24
    # Frontier-size open models, serverless ($3 in / $15 out per million tokens). Effort "low" to keep reasoning (and cost) down.
    ("ember-1", "low"),
    # ("kimi-k3", "low"),   # 10 positions done 2026-10-24: 1/10 best, ~92s and $0.06 per move. Dropped.
]

# Start small: 50 positions x 4 settings is a cheap first look. The full test set is 500.
N_POSITIONS = 50  # The frontier baselines used the first 50; the full test set is 500

START_AT_POSITION = 1  # set higher to resume a run that stopped partway

# How much worse than the engine's best move (in hundredths of a pawn) still counts as a "good" move.
GOOD_MARGIN_CP = 50

MAX_TOKENS = 16000  # thinking counts against this. At 4000, GPT-OSS on high ran out mid-thought and answered nothing
# Experiment 2: list every legal move in the prompt. Its rows go to their own file so they never mix with plain-prompt rows.
GIVE_LEGAL_MOVES = True
RESULTS_FILE = "eval_results_legal.csv" if GIVE_LEGAL_MOVES else "eval_results.csv"

# List prices, dollars per million tokens (input, output)
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "gpt-oss-120b": (0.15, 0.60),
    "kimi-k3": (3.00, 15.00),
    "ember-1": (3.00, 15.00),
}

claude_client = anthropic.Anthropic()
fireworks_client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=os.environ["FIREWORKS_API_KEY"],
)


# ---------------------------------------------------------------------------
# Calling the models: the same two functions as your ticket project's step3_eval.py
# ---------------------------------------------------------------------------
def call_claude(prompt, effort, model):
    start = time.perf_counter()
    first_token_at = None
    with claude_client.messages.stream(
        model=model,
        max_tokens=MAX_TOKENS,
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
        "input_tokens": final.usage.input_tokens,
        "output_tokens": final.usage.output_tokens,
    }


def call_fireworks(prompt, effort, model):
    start = time.perf_counter()
    first_answer_token_at = None
    usage = None
    answer = ""
    stream = fireworks_client.chat.completions.create(
        model=model if model.startswith("accounts/") else f"accounts/fireworks/models/{model}",
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
        **({"reasoning_effort": effort} if effort else {}),  # skip it for models without a reasoning setting
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
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
UCI_MOVE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b")


def parse_move(text, fen):
    """Find the model's move in its reply and return it in UCI form (e.g. e2e4), or None.

    We ask for UCI, but a model may answer in the usual human notation instead ("Nf3", "O-O").
    That's a formatting slip, not a chess mistake, so we translate it rather than mark it wrong.
    """
    match = UCI_MOVE.search(text.lower())
    if match:
        return match.group(1)
    board = chess.Board(fen)
    for word in re.findall(r"[A-Za-z0-9+#=\-]+", text):
        try:
            return board.parse_san(word).uci()
        except ValueError:
            continue
    return None


def score_move(fen, move, best_move, candidates):
    """Return (legal, good, best, cp_loss) for one answer.

    candidates holds the engine's listed moves with their scores for the player to move,
    e.g. {"g2e4": 311, "c2e2": 280}. cp_loss is how much worse the move is than the best one,
    and is None when the engine didn't list the move, so we can't say how good it is.
    """
    if move is None:
        return False, False, False, None
    try:
        # Raises an error if the move is illegal. Also rewrites "king takes own rook" castling (e1h1)
        # into standard form (e1g1), so it matches the engine's move.
        move = chess.Board(fen).parse_uci(move).uci()
    except ValueError:
        return False, False, False, None

    is_best = move == best_move
    cp_loss = candidates[best_move] - candidates[move] if move in candidates else None
    good = is_best or (cp_loss is not None and cp_loss <= GOOD_MARGIN_CP)
    return True, good, is_best, cp_loss


# ---------------------------------------------------------------------------
# The main loop: every setting x every test position
# ---------------------------------------------------------------------------
with open("test_positions.csv", newline="") as f:
    positions = list(csv.DictReader(f))[:N_POSITIONS]
first = START_AT_POSITION  # skip positions an earlier run already did

all_rows = []
for model, effort in CONFIGS:
    print(f"\n=== {model} / effort {effort or 'default'} ===")
    call = call_claude if model.startswith("claude") else call_fireworks
    price_in, price_out = PRICES.get(model, (0.0, 0.0))  # a rented GPU has no per-token price

    errors_in_a_row = 0
    for i, p in enumerate(positions[first - 1:], first):
        try:
            prompt = (PROMPT_WITH_LEGAL_MOVES.format(fen=p["fen"], legal_moves=legal_moves_text(p["fen"]))
                      if GIVE_LEGAL_MOVES else PROMPT.format(fen=p["fen"]))
            r = call(prompt, effort, model)
            errors_in_a_row = 0
        except Exception as e:  # one failed request shouldn't kill the whole run
            print(f"  position {i}: ERROR {e}")
            errors_in_a_row += 1
            if errors_in_a_row == 3:  # something is wrong with this model's setup, not one position
                print("  3 errors in a row: skipping this setting")
                break
            continue

        move = parse_move(r["answer"], p["fen"])
        legal, good, best, cp_loss = score_move(p["fen"], move, p["best_move"], json.loads(p["candidates"]))
        cost = r["input_tokens"] / 1e6 * price_in + r["output_tokens"] / 1e6 * price_out

        row = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "effort": effort,
            "position": i,
            "fen": p["fen"],
            "engine_move": p["best_move"],
            "model_move": move,
            "legal": legal,
            "good": good,
            "best": best,
            "cp_loss": cp_loss,
            "ttft_s": round(r["ttft_s"], 3),
            "input_tokens": r["input_tokens"],
            "output_tokens": r["output_tokens"],
            "cost_usd": round(cost, 6),
            "raw_answer": r["answer"],
        }
        log_result(row, path=RESULTS_FILE)
        all_rows.append(row)

        mark = "★" if best else "✓" if good else "·" if legal else "✗"
        print(f"  {mark} position {i:>3}: engine {p['best_move']:<6} model {move}")

# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------
print(f"\n\nSCORECARD  ({len(positions)} positions; ★ best, ✓ good, · legal but weaker, ✗ illegal or no move)")
print(f"{'setting':<28}{'legal':>9}{'good':>9}{'best':>9}{'median TTFT':>13}{'cost/1k moves':>15}")
for model, effort in CONFIGS:
    rows = [r for r in all_rows if r["model"] == model and r["effort"] == effort]
    if not rows:
        continue
    n = len(rows)
    legal = sum(r["legal"] for r in rows)
    good = sum(r["good"] for r in rows)
    best = sum(r["best"] for r in rows)
    ttft = statistics.median(r["ttft_s"] for r in rows)
    cost_per_1k = statistics.mean(r["cost_usd"] for r in rows) * 1000
    name = model.split("#")[0].split("/")[-1] + " / " + (effort or "default")
    print(f"{name:<28}{legal:>5}/{n:<3}{good:>5}/{n:<3}{best:>5}/{n:<3}{ttft:>11.2f}s{'$' + format(cost_per_1k, ',.2f'):>15}")

print(f"\nFull details are in {RESULTS_FILE}. To see a position as a board:")
print("  python -c \"import chess; print(chess.Board('PASTE_FEN'))\"")
