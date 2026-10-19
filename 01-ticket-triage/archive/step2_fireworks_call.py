# Step 2b: one timed request to GPT-OSS 120B on Fireworks.
# Same ticket, same prompt, same measurements as step2_claude_call.py, so the numbers are comparable.
# Run it from your llm-bench folder:  python step2_fireworks_call.py

import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from bench_log import log_result
from datetime import datetime

load_dotenv()

TICKET = "Hi, I was charged twice for my subscription this month (order #48213). Please refund the duplicate charge ASAP."
PROMPT = (
    'Classify this support ticket. Reply with only JSON like '
    '{"category": "billing|technical|account|other", "urgency": "low|medium|high"}\n\n'
    f"Ticket: {TICKET}"
)
EFFORT = "low"

# DIFFERENCE 1: we use OpenAI's library, but point it at Fireworks' address.
# Fireworks deliberately copies OpenAI's API format so customers can switch with a two-line change.
client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=os.environ["FIREWORKS_API_KEY"],  # unlike Anthropic's SDK, we pass this key explicitly
)

start = time.perf_counter()
first_any_token_at = None     # first token of ANY kind, including the model's reasoning
first_answer_token_at = None  # first token of the actual answer (comparable to the Claude script)
usage = None

stream = client.chat.completions.create(
    model="accounts/fireworks/models/gpt-oss-120b",
    max_tokens=2000,
    messages=[{"role": "user", "content": PROMPT}],
    reasoning_effort=EFFORT,
    stream=True,
    stream_options={"include_usage": True},  # DIFFERENCE 2: ask for token counts at the end of the stream
)

answer = ""
for chunk in stream:
    # print(chunk)  # print the raw chunk so you can see the streaming format
    # The very last chunk has no text, only the usage (token counts).
    if chunk.usage:
        usage = chunk.usage
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta

    # DIFFERENCE 3: GPT-OSS is a reasoning model and Fireworks streams its reasoning separately.
    # We don't print it, but we note when it starts so you can see the "thinking" gap.
    reasoning = getattr(delta, "reasoning_content", None)
    if reasoning and first_any_token_at is None:
        first_any_token_at = time.perf_counter()

    if delta.content:
        answer += delta.content
        now = time.perf_counter()
        if first_any_token_at is None:
            first_any_token_at = now
        if first_answer_token_at is None:
            first_answer_token_at = now
        print(delta.content, end="", flush=True)

end = time.perf_counter()

# List prices for GPT-OSS 120B on Fireworks, in dollars per million tokens
PRICE_IN, PRICE_OUT = 0.15, 0.60

print("\n")
if first_answer_token_at:
    print(f"Time to first token (any):    {first_any_token_at - start:.2f} s")
    print(f"Time to first answer token:   {first_answer_token_at - start:.2f} s   <- compare with Claude's TTFT")
print(f"Total time:                   {end - start:.2f} s")
if usage:
    cost = usage.prompt_tokens / 1e6 * PRICE_IN + usage.completion_tokens / 1e6 * PRICE_OUT
    print(f"Input tokens:                 {usage.prompt_tokens}")
    print(f"Output tokens:                {usage.completion_tokens}  (includes reasoning, billed as output)")
    print(f"Cost:                         ${cost:.6f}")
else:
    print("No usage data came back, so cost can't be computed. Tell Claude!")

log_result({
    "time": datetime.now().isoformat(timespec="seconds"),
    "model": "gpt-oss-120b",
    "effort": EFFORT,
    "ttft_s": round(first_answer_token_at - start, 3) if first_answer_token_at else None,
    "total_s": round(end - start, 3),
    "input_tokens": usage.prompt_tokens if usage else None,
    "output_tokens": usage.completion_tokens if usage else None,
    "cost_usd": round(cost, 6) if usage else None,
    "answer": answer,
})