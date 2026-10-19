# Pull a sample of Stockfish-analyzed chess positions from Lichess, so we can look at the data
# before deciding how to filter it or what the model should learn.
#
# Run it:  python get_sample.py
# Output:  sample_<size>k.csv (one row per analyzed position), plus a summary printed below.
#
# The full dataset has hundreds of millions of rows, far too many to download. "Streaming" reads it
# like a video: rows arrive one at a time and we stop once we have enough.

import csv
from collections import Counter

from datasets import load_dataset

SAMPLE_SIZE = 140_000  # about 100k distinct positions (the 20k-row sample gave 14.5k)
OUT_FILE = f"sample_{SAMPLE_SIZE // 1000}k.csv"
SEED = 7

stream = load_dataset("Lichess/chess-position-evaluations", split="train", streaming=True)
# The file may be stored in some order (e.g. by position), so mix rows up before taking a sample.
stream = stream.shuffle(seed=SEED, buffer_size=200_000)

rows = []
for r in stream:
    rows.append(r)
    if len(rows) >= SAMPLE_SIZE:
        break

with open(OUT_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["fen", "line", "depth", "knodes", "cp", "mate"])
    writer.writeheader()
    for r in rows:
        writer.writerow({k: r[k] for k in writer.fieldnames})
print(f"Saved {len(rows)} rows to {OUT_FILE}\n")

# A few numbers to start from. None of these decide anything yet; they show what's in the data.
fens = Counter(r["fen"] for r in rows)
print(f"Distinct positions: {len(fens)} (out of {len(rows)} rows)")
print(f"Positions that appear more than once: {sum(1 for c in fens.values() if c > 1)}")

print(f"\nRows with a forced mate instead of a score: {sum(1 for r in rows if r['mate'] is not None)}")

depths = sorted(r["depth"] for r in rows)
print(f"\nDepth: lowest {depths[0]}, middle {depths[len(depths) // 2]}, highest {depths[-1]}")

line_lengths = Counter(len(r["line"].split()) for r in rows)
print("\nMoves per line (length: rows):")
for n in sorted(line_lengths):
    print(f"  {n:>3}: {line_lengths[n]}")

print(f"\nNext: open {OUT_FILE} and look at 20 or so rows. Questions to answer in your own words:")
print("  1. When the same position appears twice, how do its rows differ?")
print("  2. How many of the positions look like openings, middlegames, or endgames?")
print("  3. What would you skip, and why?")
