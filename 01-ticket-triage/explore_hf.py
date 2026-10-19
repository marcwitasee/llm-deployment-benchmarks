# Explore the Hugging Face ticket data to learn what each queue and priority really means,
# WITHOUT looking at the 100 test tickets in tickets_hf.csv.
#
# Run it:  python explore_hf.py
# Output:  dev_tickets.csv (200 different tickets to browse in Excel), plus a summary printed below.
#
# Why separate tickets? If you write your prompt by studying the test tickets, your score goes up
# because you memorised the test, not because the prompt got better. Professionals keep a
# "development set" for exploring and a "test set" that's only used for the final score.

import csv
import random
from collections import Counter, defaultdict

from datasets import load_dataset

DEV_SIZE = 200
EXAMPLES_PER_QUEUE = 4
SEED = 7  # different from the test sample's seed

QUEUES = ["Technical Support", "IT Support", "Customer Service", "Product Support", "Billing and Payments"]

data = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")  # uses the copy you already downloaded

# The 100 test tickets, so we can leave them out.
with open("tickets_hf.csv", newline="") as f:
    test_texts = {row["ticket"] for row in csv.DictReader(f)}

def ticket_text(r):
    subject = (r["subject"] or "").strip()
    body = r["body"].strip()
    return f"{subject}\n\n{body}" if subject else body

pool = [
    r for r in data
    if r["language"] == "en"
    and str(r["priority"]).lower() in ("low", "medium", "high")
    and r["body"]
    and r["queue"] in QUEUES
    and ticket_text(r) not in test_texts
]

random.seed(SEED)
dev = random.sample(pool, DEV_SIZE)

with open("dev_tickets.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "queue", "priority", "type", "ticket"])
    for i, r in enumerate(dev, 1):
        writer.writerow([i, r["queue"], str(r["priority"]).lower(), r["type"], ticket_text(r)])
print(f"Saved {DEV_SIZE} development tickets to dev_tickets.csv (none of them are in your test set)\n")

# How priority is spread across each queue. If every queue has a similar mix, priority probably
# isn't tied to the kind of problem, which is useful to know before writing rules about it.
print("Queue x priority counts:")
print(f"  {'queue':<24}{'low':>6}{'medium':>8}{'high':>6}")
table = defaultdict(Counter)
for r in dev:
    table[r["queue"]][str(r["priority"]).lower()] += 1
for q in QUEUES:
    c = table[q]
    print(f"  {q:<24}{c['low']:>6}{c['medium']:>8}{c['high']:>6}")

# A few subjects per queue, to get a feel for what lands where.
print("\nExample subjects per queue:")
by_queue = defaultdict(list)
for r in dev:
    by_queue[r["queue"]].append(r)
for q in QUEUES:
    print(f"\n  {q}")
    for r in by_queue[q][:EXAMPLES_PER_QUEUE]:
        subject = (r["subject"] or r["body"][:70]).strip().replace("\n", " ")
        print(f"    [{str(r['priority']).lower():>6}] {subject[:90]}")

print("\nNext: open dev_tickets.csv, sort by queue, and read 5-10 tickets per queue.")
print("Write down, in your own words, what seems to decide the queue and the priority.")
