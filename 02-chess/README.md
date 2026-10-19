# Can Fine-Tuning Teach a Small Model to Play Chess?

**Short answer: it teaches the format, not the thinking.** After fine-tuning on 73,000 engine-labeled positions, Qwen 3.5 9B played a legal move 71% of the time (up from 36% untuned), but it found the engine's best move only 3% of the time. A random legal move would find it about 6% of the time.

## The question

The [ticket triage project](../01-ticket-triage) found that fine-tuning made a small model worse, but its labels were noisy, so the cause was unclear. Chess has perfect labels: a chess engine (Stockfish) gives the best move for every position. This project tests whether fine-tuning on clean labels can teach a small model to pick good moves, and how it compares with frontier models on accuracy, speed and cost.

## Setup

**Data.** Positions from the [Lichess/chess-position-evaluations](https://huggingface.co/datasets/Lichess/chess-position-evaluations) dataset on Hugging Face. The model sees only the board (a FEN string). The label is the engine's best first move. Engine scores and search depth are left out of the input to avoid leaking the answer.
- **Test set:** 499 positions, held out and kept fixed across every run. All of the engine's candidate moves are kept, so a near-best move can count as "good".
- **Training sets:** 14,069 positions for the first fine-tune and 72,955 for the second, none of them in the test set.

**Models.**
| Model | How it's served |
|---|---|
| Claude Opus 5.5 (low and high reasoning effort) | Anthropic API, pay per token |
| GPT-OSS 120B (low effort) | Fireworks serverless, pay per token |
| Qwen 3.5 9B, untuned and two LoRA fine-tunes | Fireworks dedicated GPU, pay per hour |

**Fine-tuning.** LoRA on Fireworks, rank 8, learning rate 0.0001. The 14k run used 2 epochs and cost $1.37; the 73k run used 1 epoch and cost $1.49.

**Scoring.** *Legal* means a real move in that position. *Best* means exactly the engine's top move. *Good* also counts any engine candidate within 50 centipawns (half a pawn) of the best. Latency is the median time to first token.

## Results

| Model | Positions | Legal | Good | Best | Latency | Cost / 1k moves |
|---|---|---|---|---|---|---|
| Claude Opus 5.5, high | 50 | 46 | 14 | 13 | 14.2s | $25.60 |
| Claude Opus 5.5, low | 50 | 47 | 10 | 10 | 6.7s | $10.50 |
| GPT-OSS 120B, low | 50 | 42 | 3 | 3 | 5.8s | $0.76 |
| Qwen 3.5 9B, untuned | 50 | 18 | 0 | 0 | 0.34s | GPU hours |
| Qwen 3.5 9B, fine-tuned on 14k | 50 | 31 | 2 | 2 | 0.35s | GPU hours |
| Qwen 3.5 9B, fine-tuned on 73k | 50 | 38 | 1 | 1 | ~0.35s | GPU hours |
| **Qwen 3.5 9B, fine-tuned on 73k** | **499** | **352 (71%)** | **22 (4%)** | **15 (3%)** | ~0.35s | GPU hours |
| Random legal move (estimate) | 499 | 499 | n/a | ~29 (6%) | none | none |

The 50-position rows use the first 50 test positions. GPT-OSS 120B at high effort never finished an answer, even with a 16,000-token budget, so it has no row.

## Findings

- **More data improved legality, not play.** Legal moves rose from 18 to 31 to 38 of 50 as training data grew. Mistakes shifted from moving from empty squares or moving the opponent's pieces to moving its own piece somewhere it can't go. Best-move accuracy stayed flat.
- **The engine searches; the model only pattern-matches.** Stockfish evaluates millions of positions per move. A fine-tuned model sees only the answer and has no way to search. Frontier models at least reason step by step, which is why Claude found the best move far more often.
- **Speed and cost still favor the small model.** It answered in about 0.35s, against 6.7s for Claude at low effort ($10.50 per 1,000 moves) and 14.2s at high effort ($25.60).

## What this means for deployments

Fine-tuning is good at teaching a model a format or a narrow mapping, but it does not add reasoning the base model lacks. For tasks that need search or multi-step reasoning, a model that thinks before answering, or a tool such as an engine the model can call, is the better path.

## Limits

- Each configuration ran once. The 50-position comparisons are small, so only large gaps are meaningful.
- 401 of the 499 test positions list only one engine move, so "good" is mostly the same as "best".
- Untested ideas: give the model the list of legal moves at run time, or spell the board out square by square.

## Reproduce it

Run from this folder. API keys come from the `.env` file at the repository root (see the [top-level README](../README.md)).

1. `python get_sample.py` streams positions from Hugging Face into `sample_<size>k.csv` (set `SAMPLE_SIZE`; the first run used 20,000 rows and the scale-up 140,000).
2. `python make_positions.py` picks one label per position (deepest search, then the best score for the side to move, with faster mates ranked higher), normalizes castling notation, skips Chess960 positions, reuses `test_positions.csv`, and writes the training file.
3. Upload the training file to Fireworks and run a supervised LoRA fine-tune on `qwen3p5-9b` (settings above).
4. Set the models in `eval_positions.py` and run `python eval_positions.py`. Results go to `eval_results.csv`.

The raw samples and training files are not committed because they are large and can be rebuilt with steps 1 and 2. The Qwen runs need your own Fireworks on-demand deployment; delete it when the run finishes, because it bills by the second.
