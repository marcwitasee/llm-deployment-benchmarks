# Step 2: one timed request to Claude Opus 5.5.
# Run it from your llm-bench folder (where your .env lives):  python step2_claude_call.py

import time
from dotenv import load_dotenv
import anthropic
from bench_log import log_result
from datetime import datetime

load_dotenv()  # reads ANTHROPIC_API_KEY from .env; the SDK finds it automatically

# The task: classify one support ticket. Short input, short output, like a lot of real business traffic.
TICKET = "Hi, I was charged twice for my subscription this month (order #48213). Please refund the duplicate charge ASAP."
PROMPT = (
    'Classify this support ticket. Reply with only JSON like '
    '{"category": "billing|technical|account|other", "urgency": "low|medium|high"}\n\n'
    f"Ticket: {TICKET}"
)
EFFORT = "high"  # how hard the model should think; a simple classification doesn't need much

client = anthropic.Anthropic()

start = time.perf_counter()  # a high-precision stopwatch
first_token_at = None

# Streaming means the reply arrives in pieces as it's generated, so we can see WHEN the first piece lands.
# Without streaming you only learn the total time, not time to first token (TTFT).
with client.messages.stream(
    model="claude-opus-5-5",
    max_tokens=2000,                   # a ceiling, not a target; you pay only for what's generated
    thinking={"type": "adaptive", "display": "summarized"},
    output_config={"effort": EFFORT},   # how hard the model thinks; a simple classification doesn't need much
    messages=[{"role": "user", "content": PROMPT}],
) as stream:
    for text in stream.text_stream:
        if first_token_at is None:
            first_token_at = time.perf_counter()
        print(text, end="", flush=True)
    final = stream.get_final_message()

end = time.perf_counter()
usage = final.usage

# List prices for Opus 5.5, in dollars per million tokens
PRICE_IN, PRICE_OUT = 4.00, 20.00
cost = usage.input_tokens / 1e6 * PRICE_IN + usage.output_tokens / 1e6 * PRICE_OUT

print("\n")
print(f"Time to first token: {first_token_at - start:.2f} s")
print(f"Total time:          {end - start:.2f} s")
print(f"Input tokens:        {usage.input_tokens}")
print(f"Output tokens:       {usage.output_tokens}  (includes any thinking, which is billed as output)")
print(f"Cost:                ${cost:.6f}")

log_result({
    "time": datetime.now().isoformat(timespec="seconds"),
    "model": "claude-opus-5-5",
    "effort": EFFORT,
    "ttft_s": round(first_token_at - start, 3),
    "total_s": round(end - start, 3),
    "input_tokens": usage.input_tokens,
    "output_tokens": usage.output_tokens,
    "cost_usd": round(cost, 6),
    "answer": "".join(b.text for b in final.content if b.type == "text"),
})
