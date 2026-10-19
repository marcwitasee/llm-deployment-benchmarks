# Pull a random sample of real-looking support tickets from Hugging Face into tickets_hf.csv,
# in the same format as tickets.csv, so step3_eval.py can run on it.
#
# One-time setup:  pip install datasets
# Run it:          python make_hf_sample.py
#
# Dataset: https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets (CC BY-NC 4.0, non-commercial)

import csv
import random
from collections import Counter

from datasets import load_dataset

SAMPLE_SIZE = 100   # how many tickets to keep
TOP_QUEUES = 5      # keep only the most common queues, so the model has a short, clear list to choose from
SEED = 42           # fixed seed = the same "random" sample every time you run it (reproducible)

print("Downloading the dataset (first run takes a minute)...")
data = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
print(f"{len(data):,} tickets downloaded")

# Keep English tickets with a normal priority label and some text in the body.
rows = [
    r for r in data
    if r["language"] == "en"
    and str(r["priority"]).lower() in ("low", "medium", "high")
    and r["body"]
]
print(f"{len(rows):,} English tickets with low/medium/high priority")

# Find the most common queues and keep only tickets from those.
queue_counts = Counter(r["queue"] for r in rows)
top_queues = [q for q, _ in queue_counts.most_common(TOP_QUEUES)]
rows = [r for r in rows if r["queue"] in top_queues]

random.seed(SEED)
sample = random.sample(rows, SAMPLE_SIZE)

with open("tickets_hf.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "ticket", "expected_category", "expected_urgency", "why_its_tricky"])
    for i, r in enumerate(sample, 1):
        subject = (r["subject"] or "").strip()
        ticket = f"{subject}\n\n{r['body'].strip()}" if subject else r["body"].strip()
        # Lowercase the labels, because step3_eval.py lowercases the model's answers before comparing.
        writer.writerow([i, ticket, r["queue"].lower(), str(r["priority"]).lower(), "from HF dataset"])

print(f"\nSaved {SAMPLE_SIZE} tickets to tickets_hf.csv")
print("\nQueues in the sample (these become the categories):")
for q, n in Counter(r["queue"] for r in sample).most_common():
    print(f"  {n:>3}  {q}")
print("\nPriorities in the sample:")
for p, n in Counter(str(r["priority"]).lower() for r in sample).most_common():
    print(f"  {n:>3}  {p}")
print("\nPaste this output to Claude so the prompt can list exactly these queues.")
