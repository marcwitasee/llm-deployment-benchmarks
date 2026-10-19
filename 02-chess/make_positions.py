# Turn a sample file (several engine rows per position) into one labeled example per position,
# keeping 500 test positions aside and writing the rest as training data.
#
# Run it:  python make_positions.py
# Output:  test_positions.csv  - 500 positions kept aside for scoring, with ALL the engine's candidate
#                                 moves, so later you can give partial credit for near-best moves.
#          TRAIN_FILE          - the rest, in the chat format Fireworks fine-tuning expects.
#
# Same rule as the ticket project: the model must never train on a position it will be tested on.

import csv
import json
import os
import random
from collections import defaultdict

import chess

from chess_prompts import PROMPT  # shared with eval_positions.py

TEST_SIZE = 500
SEED = 42
SAMPLE_FILE = "sample_140k.csv"  # was sample.csv (20k rows) for the first fine-tune
TRAIN_FILE = "train_chess_100k.jsonl"  # the first fine-tune used train_chess.jsonl


def to_number(value):
    """CSV stores everything as text; turn '311' into 311 and '' into None."""
    return int(value) if value not in ("", "None") else None


def score_for_mover(r):
    """One number for how good this row's move is for the player about to move. Bigger is better."""
    if r["mate"] is None:
        points = r["cp"]  # a normal evaluation, from White's side
    elif r["mate"] > 0:
        points = 100_000 - r["mate"]  # White forces mate: huge, and bigger the faster it comes
    else:
        points = -100_000 - r["mate"]  # Black forces mate: hugely negative, worst when fastest
    # Flip the sign on Black's turn, so that "bigger is better" holds for whoever is moving.
    white_to_move = r["fen"].split()[1] == "w"
    return points if white_to_move else -points


def standard_uci(fen, move):
    """Lichess writes castling as "king takes its own rook" (e1h1); standard notation is e1g1.
    Returns the move in standard form, or None if it isn't legal in normal chess
    (a few positions come from Chess960, a variant with shuffled starting pieces)."""
    try:
        return chess.Board(fen).parse_uci(move).uci()
    except ValueError:
        return None


def best_row(rows):
    """Given all the engine rows for ONE position, return the row whose first move is your label.

    Each row is a dict with: fen, line, depth, knodes, cp, mate (numbers already converted).
    Useful facts:
      - Whose turn it is: rows[0]["fen"].split()[1] is "w" or "b".
      - cp is scored from White's side: higher is better for White, lower is better for Black.
      - Some rows have cp = None and a mate value instead (mate in N moves; negative = Black mates).
    """
    # Step 1: only trust the deepest analysis. Rows from shallower searches are older, less careful
    # opinions about the same position, so we drop them.
    deepest = max(r["depth"] for r in rows)
    rows = [r for r in rows if r["depth"] == deepest]

    # Step 2: score every row for the player about to move (see score_for_mover above),
    # and the best move is simply the row with the biggest score.
    return max(rows, key=score_for_mover)


# ---- Plumbing below: read, group, split, write. ----

by_fen = defaultdict(list)
with open(SAMPLE_FILE, newline="") as f:
    for r in csv.DictReader(f):
        for key in ("depth", "knodes", "cp", "mate"):
            r[key] = to_number(r[key])
        by_fen[r["fen"]].append(r)

# Keep the SAME test positions across experiments, so scores stay comparable.
# If test_positions.csv already exists, reuse it; otherwise pick 500 positions at random.
if os.path.exists("test_positions.csv"):
    with open("test_positions.csv", newline="") as f:
        test_fens = [r["fen"] for r in csv.DictReader(f)]
    write_test = False
else:
    fens = sorted(by_fen)
    random.seed(SEED)
    random.shuffle(fens)
    test_fens = fens[:TEST_SIZE]
    write_test = True
held_out = set(test_fens)
train_fens = [fen for fen in sorted(by_fen) if fen not in held_out]  # never train on a test position
print(f"{len(by_fen)} distinct positions in {SAMPLE_FILE}: {len(train_fens)} for training "
      f"({len(by_fen) - len(train_fens)} skipped because they are test positions)")

skipped = 0
if write_test:
    with open("test_positions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fen", "best_move", "candidates"])
        for fen in test_fens:
            rows = by_fen[fen]
            best_move = standard_uci(fen, best_row(rows)["line"].split()[0])
            if best_move is None:
                skipped += 1
                continue
            # Every candidate first move from the deepest analysis, with its score for the player to move,
            # e.g. {"g2e4": 311, "c2e2": 280}. eval_positions.py uses these for partial credit.
            deepest = max(r["depth"] for r in rows)
            candidates = {}
            for r in rows:
                move = standard_uci(fen, r["line"].split()[0])
                if r["depth"] == deepest and move:
                    candidates[move] = score_for_mover(r)
            writer.writerow([fen, best_move, json.dumps(candidates)])

with open(TRAIN_FILE, "w") as f:
    for fen in train_fens:
        best_move = standard_uci(fen, best_row(by_fen[fen])["line"].split()[0])
        if best_move is None:
            skipped += 1
            continue
        f.write(json.dumps({"messages": [
            {"role": "user", "content": PROMPT.format(fen=fen)},
            {"role": "assistant", "content": best_move},
        ]}) + "\n")

print(f"Skipped {skipped} positions whose best move isn't legal in normal chess (Chess960)")
print(f"Saved {TRAIN_FILE}" + (" and test_positions.csv" if write_test else " (test_positions.csv unchanged)"))
print("\nFirst training example, so you can see the format:")
with open(TRAIN_FILE) as f:
    print(f.readline())
