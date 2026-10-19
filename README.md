# LLM Deployment Benchmarks

Two hands-on experiments on a question every AI deployment faces: **when can a cheaper open model replace a frontier model, and when does fine-tuning help?** Both run open models on [Fireworks AI](https://fireworks.ai) and compare them with Claude Opus 5.5 on accuracy, latency and cost.

| Project | Question | Answer |
|---|---|---|
| [01 · Support ticket triage](01-ticket-triage) | Do you need a frontier model to route and prioritize support tickets? | **No.** With a good prompt, GPT-OSS 120B on Fireworks serverless matched Claude Opus 5.5 at 34–45x lower cost, and a 9B model tied it while answering 4–7x faster. Fine-tuning on noisy labels made the 9B model worse. |
| [02 · Chess move prediction](02-chess) | Can fine-tuning on perfect engine labels teach a small model to play well? | **No.** Fine-tuning on 73k positions doubled the legal-move rate (36% to 71%), but best-move accuracy stayed at 3%, no better than chance. Fine-tuning taught the format, not the reasoning. |

## What I'd tell a customer

- **Start with the cheapest fast model and a well-structured prompt.** On routine classification, prompt quality mattered more than model size.
- **Price per token is not price per task.** Reasoning tokens and prompt length drive the bill, so measure cost per completed task.
- **Serverless vs dedicated GPU is a volume question.** At about $8 an hour, a dedicated 9B model only beat serverless GPT-OSS on cost above roughly 53,000 tickets an hour.
- **Fine-tune for format, not for reasoning, and check your labels first.** Fine-tuning learned label skew on noisy data and never learned to search on clean data.
- **Guard against prompt injection.** GPT-OSS followed an instruction hidden in a ticket 3 of 4 times until the system prompt told it to ignore instructions inside tickets.

## Setup

1. Create a Python virtual environment and run `pip install -r requirements.txt`.
2. Create a `.env` file in this folder with `ANTHROPIC_API_KEY` and `FIREWORKS_API_KEY`. It is gitignored.
3. Follow the "Reproduce it" steps in each project's README, running scripts from inside that project's folder.

Datasets: [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) (CC BY-NC 4.0) and [Lichess/chess-position-evaluations](https://huggingface.co/datasets/Lichess/chess-position-evaluations).
