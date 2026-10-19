# Do You Need a Frontier Model to Triage Support Tickets?

**Short answer: no.** On 100 customer support tickets, a 9B open model (Qwen 3.5 9B) and GPT-OSS 120B matched Claude Opus 5.5. The serverless GPT-OSS option did it at about 1/34th to 1/45th of the cost. The 9B model answered 4 to 7 times faster than Claude.

## The question

Support teams have to route every incoming ticket to the right department with the right priority, so customers get the right response quickly. Teams tend to reach for a frontier model for this kind of work. Ticket triage, though, is a routine, high-volume classification task, so a smaller and cheaper model may do just as well. I tested whether that holds, and what it means for how you deploy.

## Setup

**Data.** I used tickets from the [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) dataset on Hugging Face (CC BY-NC 4.0). The tickets appear to be synthetic. Each ticket is labeled with one of five queues (technical support, IT support, customer service, product support, billing and payments) and a priority (low, medium, high).
- **Test set:** 100 tickets, used only for scoring.
- **Dev set:** 200 separate tickets. I wrote the structured prompt by looking only at these, so the prompt was never tuned to the test set.
- **Training set:** 1,000 more tickets, overlapping neither of the other two, used only for fine-tuning.

**Models.**
| Model | How it's served | Settings tested |
|---|---|---|
| Claude Opus 5.5 | Anthropic API, pay per token | Low and high reasoning effort |
| GPT-OSS 120B (OpenAI's open-weight model) | Fireworks serverless, pay per token | Low and high reasoning effort |
| Qwen 3.5 9B | Fireworks dedicated GPU, pay per hour | Reasoning off |
| Qwen 3.5 9B, fine-tuned | Fireworks dedicated GPU, pay per hour | Reasoning off |

**Prompts.**
- **Basic:** a short instruction plus the list of allowed labels.
- **Structured:** adds a definition for each queue and priority, written from patterns in the dev set, plus defaults for unclear cases.

**Fine-tuning.** I trained a LoRA adapter on Qwen 3.5 9B using the 1,000 training tickets (2 epochs, rank 8). Training cost about $0.14. Because it was trained on the basic prompt, the tuned model was only tested with that prompt.

**Scoring.** For each ticket I recorded whether the category was right, whether the urgency was right, and whether **both** were right. I also logged time to first token (how long before the answer starts arriving) and cost per ticket from token counts and list prices.

## Results

Scores are out of 100 test tickets. Latency is the median time to first token. Cost is per 1,000 tickets (pay-per-token models only).

| Model | Prompt | Category | Urgency | **Both right** | Latency | Cost / 1k tickets |
|---|---|---|---|---|---|---|
| Claude Opus 5.5, low | Basic | 43 | 43 | **25** | 1.23s | $1.26 |
| Claude Opus 5.5, high | Basic | 42 | 43 | **22** | 2.16s | $2.93 |
| GPT-OSS 120B, low | Basic | 38 | 45 | **22** | 0.56s | $0.06 |
| GPT-OSS 120B, high | Basic | 34 | 37 | **12** | 1.79s | $0.27 |
| Qwen 3.5 9B | Basic | 40 | 41 | **18** | 0.33s | GPU hours |
| Qwen 3.5 9B, fine-tuned | Basic | 30 | 35 | **11** | 0.34s | GPU hours |
| Claude Opus 5.5, low | Structured | 42 | 47 | **24** | 1.25s | $4.98 |
| Claude Opus 5.5, high | Structured | 43 | 45 | **25** | 2.29s | $6.63 |
| GPT-OSS 120B, low | Structured | 42 | 45 | **26** | 0.56s | $0.15 |
| GPT-OSS 120B, high | Structured | 45 | 42 | **25** | 2.55s | $0.49 |
| Qwen 3.5 9B | Structured | 44 | 49 | **25** | 0.34s | GPU hours |

## Findings

- **With a good prompt, every model landed at about 25 "both right".** Claude, GPT-OSS and the 9B model all scored between 24 and 26 with the structured prompt. A gap of one or two tickets on a 100-ticket test is noise, so these are ties. The fact that very different models converge on the same number suggests the limit is in the labels, not the models (see Limits).
- **Cheap models matched the frontier model.** GPT-OSS 120B at low effort scored 26 at $0.15 per 1,000 tickets, against $4.98 to $6.63 for Claude, which works out to 34 to 45 times cheaper. The untuned Qwen 3.5 9B scored 25, had the best urgency score of any run (49), and answered in 0.34s, against 1.25 to 2.29s for Claude.
- **Prompting beat fine-tuning.** The structured prompt lifted the untuned 9B from 18 to 25. Fine-tuning it on 1,000 tickets dropped it from 18 to 11. The tuned model mostly learned the label mix rather than the logic: for example, it predicted "high" 52 times when the right answer was high only 34 times.

## Recommendation

For a support team triaging tickets like these, I would **start with GPT-OSS 120B on serverless, low reasoning effort, with the structured prompt.** It matches the frontier model's quality at a fraction of the cost, it answers in about half a second, and serverless means you pay nothing when no tickets come in.

The choice between serverless and a dedicated GPU depends on volume:
- **Low or uneven volume:** serverless pay-per-token wins. A dedicated GPU costs roughly $8 an hour whether or not it is busy.
- **Steady, high volume, or strict latency needs:** a small model on a dedicated GPU starts to make sense. At about $8 an hour, the 9B model would need to process around 1,600 tickets an hour to beat Claude on cost, and around 53,000 an hour to beat serverless GPT-OSS. In return you get the fastest responses (0.34s) and full control of the model.

**Don't fine-tune on these labels.** Fine-tuning pays off when you have consistent labels, usually thousands of examples from your own business. Before spending on training, check label quality first: have two people label the same 50 tickets and see how often they agree.

## Limits

- **Small test set.** 100 tickets is enough to see large gaps, such as 11 vs 25, but not to rank models that are within a few points of each other. Each configuration was also run only once.
- **Noisy labels.** The ~25 ceiling and the fine-tuning drop both point to inconsistent labels in the dataset, where similar tickets carry different queues or priorities. That is likely, not proven, and it means the absolute scores understate how useful these models would be on clean data.
- **Synthetic data.** The tickets appear to be generated, so real tickets from a specific business could behave differently.
- **Cost assumptions.** Costs use list prices on the test date. The dedicated GPU comparison assumes the GPU is fully busy, and real utilization would be lower.

## Reproduce it

Run from this folder. API keys come from the `.env` file at the repository root (see the [top-level README](../README.md)).

1. `python make_hf_sample.py` builds the 100-ticket test set (`tickets_hf.csv`, seed 42).
2. `python explore_hf.py` builds the 200-ticket dev set (`dev_tickets.csv`, seed 7). Only these tickets were read when writing the structured prompt.
3. `python make_training_set.py` builds the 1,000-ticket fine-tuning file (`train_hf.jsonl`, seed 123), skipping every test and dev ticket.
4. In `step3_eval.py`, choose the models in `CONFIGS` and set `PROMPT_VERSION` (`v4_hf_basic` for the basic prompt, `v5_hf` for the structured one; the prompts live in `prompts.py`), then run `python step3_eval.py`. Results go to `eval_results_tickets_hf_<prompt>.csv`.

The Qwen runs need your own Fireworks on-demand deployment. Delete it as soon as the run finishes, because it bills by the second while it is up.

## Files

| File | What it is |
|---|---|
| `eval_results_tickets_hf_v4_hf_basic.csv`, `eval_results_tickets_hf_v5_hf.csv` | Final results on the 100 test tickets (basic and structured prompts), one row per ticket per model |
| `tickets.csv`, `eval_results_v1_basic.csv`, `eval_results_v2_marc.csv`, `eval_results_v3_structured.csv` | Earlier pilot: 20 hand-written tickets with my own answer key, three prompt versions |
| `tickets_hf.csv`, `dev_tickets.csv` | Test and dev sets |
| `archive/` | First single-call timing scripts and their logs, kept for reference |
