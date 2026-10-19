# Build the fine-tuning file for Fireworks: 1,000 Hugging Face tickets with the correct answers,
# none of which are in your test set (tickets_hf.csv) or your dev set (dev_tickets.csv).
#
# Run it:  python make_training_set.py
# Output:  train_hf.jsonl, the file you upload in Fireworks' "New job" wizard.
#
# Each line is one training example in the same chat format step3_eval.py sends:
#   user      = the v4_hf_basic prompt with the ticket filled in
#   assistant = the correct JSON answer
# Training and testing must use the SAME prompt. The model learns "when I see this prompt, answer like this",
# so if you later test it with a different prompt, it's partly guessing again.

import csv
import json
import random
from collections import Counter

from datasets import load_dataset

from prompts import PROMPTS

TRAIN_SIZE = 1000
SEED = 123  # different from the test (42) and dev (7) seeds
PROMPT_VERSION = "v4_hf_basic"
OUT_FILE = "train_hf.jsonl"

QUEUES = ["Technical Support", "IT Support", "Customer Service", "Product Support", "Billing and Payments"]

data = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")

# Tickets we must never train on.
held_out = set()
for path in ("tickets_hf.csv", "dev_tickets.csv"):
    with open(path, newline="") as f:
        held_out |= {row["ticket"] for row in csv.DictReader(f)}

def ticket_text(r):
    subject = (r["subject"] or "").strip()
    body = r["body"].strip()
    return f"{subject}\n\n{body}" if subject else body

pool, seen = [], set()
for r in data:
    if (r["language"] == "en"
            and str(r["priority"]).lower() in ("low", "medium", "high")
            and r["body"]
            and r["queue"] in QUEUES):
        text = ticket_text(r)
        if text not in held_out and text not in seen:  # also skip duplicates within the dataset
            seen.add(text)
            pool.append(r)

random.seed(SEED)
train = random.sample(pool, TRAIN_SIZE)

template = PROMPTS[PROMPT_VERSION]
chars = 0
with open(OUT_FILE, "w") as f:
    for r in train:
        prompt = template.format(ticket=ticket_text(r))
        answer = json.dumps({"category": r["queue"].lower(), "urgency": str(r["priority"]).lower()})
        f.write(json.dumps({"messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]}) + "\n")
        chars += len(prompt) + len(answer)

print(f"Saved {TRAIN_SIZE} examples to {OUT_FILE} (none overlap your test or dev tickets)")
print(f"Roughly {chars / 4 / 1e6:.2f}M tokens per pass over the data (about 4 characters per token)")
print("\nQueues in the training set:")
for q, n in Counter(r["queue"] for r in train).most_common():
    print(f"  {n:>4}  {q}")
print("\nPriorities:")
for p, n in Counter(str(r["priority"]).lower() for r in train).most_common():
    print(f"  {n:>4}  {p}")
print("\nFirst example, so you can see the format:")
with open(OUT_FILE) as f:
    print(f.readline()[:600])
