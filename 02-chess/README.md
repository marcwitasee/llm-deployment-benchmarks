# Can an Open Model Match Frontier Models at Chess?

**Short answer: only a frontier-class open model, and only by reasoning as hard.** Fine-tuning small open models on 73,000 engine-labeled positions taught them the format, not the thinking: Qwen 3.5 9B played a legal move 71% of the time (up from 36% untuned) but found the engine's best move only 3% of the time, no better than a random legal move (about 6%). Ember-1, a frontier-class open model on Fireworks, tied Claude Opus 5.5 (12 vs 13 best moves out of 50), but reasoned about 4x longer, so it was about 4x slower and 4x more expensive per move.

## The question

The [ticket triage project](../01-ticket-triage) found that fine-tuning made a small model worse, but its labels were noisy, so the cause was unclear. Chess has perfect labels: a chess engine (Stockfish) gives the best move for every position. This project tests whether fine-tuning on clean labels can teach a small model to pick good moves, whether a bigger or frontier-class open model closes the gap, and how they all compare with Claude on accuracy, speed and cost.

## Setup

**Data.** Positions from the [Lichess/chess-position-evaluations](https://huggingface.co/datasets/Lichess/chess-position-evaluations) dataset on Hugging Face. The model sees only the board (a FEN string). The label is the engine's best first move. Engine scores and search depth are left out of the input to avoid leaking the answer.
- **Test set:** 499 positions, held out and kept fixed across every run. All of the engine's candidate moves are kept, so a near-best move can count as "good".
- **Training sets:** 14,069 positions for the first fine-tune and 72,955 for the second, none of them in the test set.

**Models.**
| Model | How it's served |
|---|---|
| Claude Opus 5.5 (low and high reasoning effort) | Anthropic API, pay per token |
| GPT-OSS 120B (low effort) | Fireworks serverless, pay per token |
| Ember-1 and Kimi K3 (low effort) | Fireworks serverless, pay per token ($3 in / $15 out per million) |
| Qwen 3.5 9B, untuned and two LoRA fine-tunes | Fireworks dedicated GPU, pay per hour |
| Muse Glimmer 30B, untuned and one LoRA fine-tune | Fireworks dedicated GPU, pay per hour |

Ember-1 is built by Fireworks on Kimi K3.

**Fine-tuning.** LoRA on Fireworks, rank 8, learning rate 0.0001. The Qwen 14k run used 2 epochs and cost $1.37; the Qwen 73k run used 1 epoch and cost $1.49. The Glimmer 30B fine-tune used the same 73k examples (eval loss 2.96 to 0.45).

**Two prompts.** The plain prompt gives only the FEN. The second experiment (`GIVE_LEGAL_MOVES = True`) adds the full list of legal moves, so legality is free and only move choice is tested.

**Scoring.** *Legal* means a real move in that position. *Best* means exactly the engine's top move. *Good* also counts any engine candidate within 50 centipawns (half a pawn) of the best. Latency is the median time to first token.

## Results

### Plain prompt (FEN only)

| Model | Positions | Legal | Good | Best | Latency | Cost / 1k moves |
|---|---|---|---|---|---|---|
| Claude Opus 5.5, high | 50 | 46 | 14 | 13 | 14.2s | $25.60 |
| Claude Opus 5.5, low | 50 | 47 | 10 | 10 | 6.7s | $10.50 |
| GPT-OSS 120B, low | 50 | 42 | 3 | 3 | 5.8s | $0.76 |
| Muse Glimmer 30B, untuned, low | 50 | 41 | 3 | 3 | 6.6s | GPU hours |
| Muse Glimmer 30B, fine-tuned on 73k | 50 | 31 | 3 | 3 | 0.35s | GPU hours |
| Qwen 3.5 9B, untuned | 50 | 18 | 0 | 0 | 0.34s | GPU hours |
| Qwen 3.5 9B, fine-tuned on 14k | 50 | 31 | 2 | 2 | 0.35s | GPU hours |
| Qwen 3.5 9B, fine-tuned on 73k | 50 | 38 | 1 | 1 | ~0.35s | GPU hours |
| **Qwen 3.5 9B, fine-tuned on 73k** | **499** | **352 (71%)** | **22 (4%)** | **15 (3%)** | ~0.35s | GPU hours |
| Random legal move (estimate) | 499 | 499 | n/a | ~29 (6%) | none | none |

The 50-position rows use the first 50 test positions. GPT-OSS 120B at high effort and Glimmer at its default reasoning setting never finished an answer, even with a 16,000-token budget, so they have no row. The raw Claude rows in `eval_results.csv` predate a castling-notation fix; the table shows them rescored.

### With the list of legal moves in the prompt

| Model | Positions | Legal | Best | Latency | Output tokens (median) | Cost / 1k moves |
|---|---|---|---|---|---|---|
| Claude Opus 5.5, low | 50 | 50 | **13** | 5.7s | ~400 | $9.80 |
| Ember-1, low | 50 | 50 | **12** | 24s | ~1,700 | $38 |
| GPT-OSS 120B, low | 50 | 48 | 4 | 4.8s | ~950 | $0.64 |
| Kimi K3, low | 10 | 10 | 1 | 92s | ~3,000 | $62 |

Kimi K3 was stopped after 10 positions because it was slow and weak. Claude and Ember each found the best move on 8 of the same positions; Ember found 4 that Claude missed and Claude found 5 that Ember missed. Fifty positions is a small sample, so 12 vs 13 is a tie.

## Findings

- **More data improved legality, not play.** Legal moves rose from 18 to 31 to 38 of 50 as Qwen's training data grew. Mistakes shifted from moving from empty squares or moving the opponent's pieces to moving its own piece somewhere it can't go. Best-move accuracy stayed flat.
- **A bigger model helped only because it reasons.** Untuned Glimmer 30B reached 41 legal by thinking for about 1,400 tokens per move. Fine-tuning it on move-only answers taught it to stop thinking: it got 20x faster, but legal moves fell to 31 and best moves didn't change.
- **Low training loss is not a hit rate.** Loss is averaged over tokens, and easy tokens pull it down. Glimmer's loss fell from 2.96 to 0.45 and it still missed nearly every best move.
- **The legal-move list isolates judgment.** With legality free, Claude improved from 10 to 13 best, while GPT-OSS barely moved (3 to 4). Only reasoning strength changed the quality of the choice.
- **Frontier-class open models can match Claude, at a price.** Ember-1 tied Claude, but it used about 4x as many output tokens, so a lower price per token did not mean a cheaper answer.
- **The engine searches; the models don't.** Stockfish evaluates millions of positions per move. A fine-tuned model sees only the answer and has no way to search, which is why models that reason step by step did far better.

## What this means for deployments

Fine-tuning is good at teaching a model a format or a narrow mapping, but it does not add reasoning the base model lacks, and training on answer-only data can remove the reasoning it had. For tasks that need search or multi-step reasoning, use a model that thinks before answering, or give it a tool such as an engine it can call. Open models can now match the frontier on this task, so the choice becomes a deployment one: owning the model (privacy, control, the option to fine-tune) versus cost and speed per answer. Measure cost per task, not per token, and set reasoning effort explicitly: Kimi K3's default is "max", and Glimmer's default never finished.

## Limits

- Each configuration ran once. The 50-position comparisons are small, so only large gaps are meaningful.
- 401 of the 499 test positions list only one engine move, so "good" is mostly the same as "best".
- Untested idea: reinforcement fine-tuning (RFT), letting the model reason freely and rewarding only the final move (for example 1 for best, 0.5 for good, 0.1 for legal). That would test whether training can improve move choice without removing the reasoning.

## Reproduce it

Run from this folder. API keys come from the `.env` file at the repository root (see the [top-level README](../README.md)).

1. `python get_sample.py` streams positions from Hugging Face into `sample_<size>k.csv` (set `SAMPLE_SIZE`; the first run used 20,000 rows and the scale-up 140,000).
2. `python make_positions.py` picks one label per position (deepest search, then the best score for the side to move, with faster mates ranked higher), normalizes castling notation, skips Chess960 positions, reuses `test_positions.csv`, and writes the training file.
3. Upload the training file to Fireworks and run a supervised LoRA fine-tune on `qwen3p5-9b` or Glimmer 30B (settings above).
4. Set the models in `CONFIGS` in `eval_positions.py` and run `python eval_positions.py`. Other settings: `N_POSITIONS`, `START_AT_POSITION` (to resume a run) and `GIVE_LEGAL_MOVES`. Plain-prompt results go to `eval_results.csv`, legal-move-list results to `eval_results_legal.csv`.

The raw samples and training files are not committed because they are large and can be rebuilt with steps 1 and 2. The Qwen and Glimmer runs need your own Fireworks on-demand deployment; delete it when the run finishes, because it bills by the second.
