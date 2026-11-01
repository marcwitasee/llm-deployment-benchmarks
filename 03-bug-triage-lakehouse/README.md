# AI bug triage on the Databricks lakehouse

**Recommendation:** route incoming bug reports with **gpt-oss-120b** and a short, targeted prompt, running inside Databricks through `ai_query()`. On 280 bugs it was never tuned on, it sent **79%** of them to the right team (random guessing: about 7%), for about **$0.27 per 1,000 bugs**. Keep a person reviewing each team's queue, and do **not** rely on the model to flag urgent bugs yet: no model tested did better than about half.

This project treats Mozilla's public Firefox bug tracker as a stand-in for a customer drowning in support tickets. It builds the full path on Databricks Free Edition: raw data to clean tables to a manager's dashboard, then tests whether an LLM can do first-pass triage, what it costs, and where it breaks.

## The customer problem

A support or engineering org receives thousands of tickets a month. Someone has to read each one, decide which team owns it, and decide how urgent it is. That person is slow and expensive, queues fill up, and managers cannot see which teams are falling behind.

Two questions:

1. **Visibility:** which teams are keeping up, and which backlogs are growing?
2. **Triage:** can an LLM route new tickets to the right team, and flag the urgent ones, accurately and cheaply?

## What was built

The data flows through five stages, all in Unity Catalog under `support.bugzilla`:

1. **Ingest:** a Python script pulls bugs from the Bugzilla REST API, and the files are uploaded to the `raw` volume.
2. **Bronze** (`bronze_bugs_all`, 27,786 rows): every Firefox and Core defect filed in 2024-2025, loaded as-is.
3. **Silver** (`silver_bugs_all`, 16,624 rows): robot and bulk-filed test failures removed, each bug mapped to one of 14 teams through the `team_queues` lookup (197 components), with an urgency flag and resolution kept. `silver_bugs` (6,243 rows) holds the bugs pulled with full descriptions, which the AI tests need. It was built before the WPT filter was added (see Limits).
4. **Gold** (`gold_team_weekly`, 14 teams x 105 weeks): opened, closed, open backlog and median days to close per team per week. It feeds the AI/BI dashboard and the Genie agent.
5. **AI evaluation:** `ai_query()` runs each model and prompt over test sets drawn from `silver_bugs`. Results land in `eval_runs` and are logged to MLflow.

## Results

### How the models were tested

Each test set is a fixed sample of real bugs where the right answer is already known, because Mozilla's own triagers filed each bug under a component that maps to one team. The model sees a bug's title and the first 2,000 characters of its description, answers with a team name, and is scored right only on an exact match.

| Test set | Bugs | How it was drawn | Used for |
| --- | --- | --- | --- |
| `eval_set` | 280 | 20 random bugs per team from `silver_bugs` | Comparing models, and studying mistakes to write prompt v2 |
| `eval_set_2` | 280 | Another 20 per team, excluding every bug in `eval_set` | The fair test: no prompt was written by looking at these bugs |
| `urgency_set` | 200 | 100 urgent (S1/S2) and 100 non-urgent bugs, excluding both sets above | Testing whether models can flag urgent bugs |

Two test sets are needed because prompt v2 was written by reading the mistakes on `eval_set`, so its score there is flattering. The second set shows whether the improvement holds on bugs the prompt-writer never saw. Equal numbers per team keep the large teams from dominating the score.

### Routing: which team owns this bug?

Accuracy on `eval_set_2`:

| Model | Prompt v1 (one line per team) | Prompt v2 (targeted rules) | Seconds per bug | Cost per 1,000 bugs |
| --- | --- | --- | --- | --- |
| **gpt-oss-120b** | 71.4% | **78.9%** | ~0.46 | **$0.27** |
| Llama 4 Maverick | 69.6% | 74.6% | ~0.10 | $0.54 |
| gpt-oss-20b | 65.7% | 72.9% | ~0.11 | $0.15 |

- **The prompt mattered more than the model.** Prompt v2 added 5 to 7.5 points for every model; going from the smallest to the largest model added about 6. v2 named the on-screen features each team owns (a certificate error page belongs to Privacy & Security, not the browser interface team) and told the model to use the catch-all team only as a last resort.
- **Pasting in all the documentation did worse.** Prompt v3 included Mozilla's own descriptions of 195 components (8 times longer). It scored 69%, and the model more often ran out of output budget while reasoning, leaving 13 answers blank.
- **Runs vary.** Identical runs of a reasoning model differed by about 3 points, so smaller gaps are noise.

### Cost

At list price, every option is under $1 per 1,000 bugs. Mozilla files about 14,000 of these bugs a year, so even the priciest model costs under $10 a year. Choosing gpt-oss-120b over gpt-oss-20b costs about $0.12 more per 1,000 bugs and avoids about 60 misrouted bugs: a fifth of a cent per misroute avoided. At these volumes, **pick on accuracy**. Revisit the serving setup (provisioned throughput instead of pay-per-token) only when volume is high and steady.

Prices: [Databricks foundation model serving](https://www.databricks.com/product/pricing/foundation-model-serving), at its example rate of $0.07 per DBU. Token counts measured on 50 bugs per model.

### Urgency: should this bug jump the queue?

Tested on 100 urgent (S1/S2) and 100 non-urgent bugs:

| Model | Urgent bugs caught (recall) | False alarms |
| --- | --- | --- |
| gpt-oss-20b | 55% | 29% |
| Llama 4 Maverick | 43% | 20% |
| gpt-oss-120b | 39% | 17% |

This is close to a coin flip, and the cause is mostly the labels. The three models agreed with each other on most bugs; where all three "missed", about a third were experiment QA bugs that Mozilla rates S2 because they block a launch, not because users are hurt. Meanwhile, crashes that Mozilla rated S3 were flagged by every model. Mozilla's severity mixes user impact with release priority. **A customer has to decide which one "urgent" means before any model can be judged on it.**

## What the data itself revealed

The manager dashboard and a Genie agent (plain-English questions over the gold tables) surfaced two problems that changed the story:

- **Bulk-filed test failures distorted the team rankings.** 1,195 "Failing WPT" bugs, filed by script when Mozilla imports new web-platform tests, sat in the queues (1,004 in Layout & CSS). Before filtering, Layout & CSS looked like the worst team (35% close rate, backlog 1,807). After, it is mid-pack (about 47%, backlog about 940). **One cleaning rule changed which team looked worst.**
- **Close rate counts sweeps as work.** A spike of 76 Graphics closures in one week was 67% WORKSFORME and only 13% FIXED: a backlog cleanup, not a burst of fixes. A fixed-only close rate is the fairer team metric.

Current close rates run from about 74% (JavaScript) to about 40% (OS Integration). Note that backlogs here count only bugs filed since January 2024, so every line rises partly by construction; compare teams and slopes.

## How to run it

The Databricks notebooks are in `notebooks/` (exported as `.sql`; cells starting with `%python` are Python). Run them in this order:

| Step | Notebook | What it does |
| --- | --- | --- |
| 1 | `pull_bugzilla.py`, `make_team_queues.py` (on your laptop) | Pull bugs from the Bugzilla REST API to JSONL (`--all-metadata` for every bug without descriptions), and build `team_queues.csv`, the component-to-team lookup. The JSONL files are not in the repo because of their size; `team_queues.csv` is |
| 2 | `00_setup` | Creates the `support` catalog, `bugzilla` schema and `raw` volume. Upload the JSONL files and `team_queues.csv` to `/Volumes/support/bugzilla/raw/` |
| 3 | `01_bronze` | Loads bugs with descriptions into `bronze_bugs` and profiles the fields |
| 4 | `01_bronze_all` | Loads metadata for every bug into `bronze_bugs_all` |
| 5 | `02_silver` | Loads `team_queues` and builds `silver_bugs` (with descriptions, used for the AI tests) |
| 6 | `02_silver_all` | Builds `silver_bugs_all` (all bugs, robot and WPT bugs removed, with resolution) |
| 7 | `03_gold` | Builds `gold_team_weekly`: opened, closed, open backlog and median days to close per team per week |
| 8 | `04_eval_set` | Picks the first routing test set, `eval_set` (20 bugs per team) |
| 9 | `05_ai_triage` | First routing runs: smoke tests, then each model with prompts v1 and v2 on `eval_set`, with accuracy and confusion queries |
| 10 | `07_eval_set_v2` | Picks the second routing test set, `eval_set_2`, with no overlap with the first |
| 11 | `09_urgency_set` | Picks `urgency_set` (100 urgent and 100 not urgent, no overlap with the routing sets) |
| 12 | `06_eval_harness` | The reusable harness. Stores prompts (routing v1 to v3, urgency u1 and u2) in tables; one parameterized cell runs any model, prompt and test set into `eval_runs` or `urgency_runs`; scoreboard cells score them |
| 13 | `08_token_cost` | Measures tokens per bug and cost per 1,000 bugs |
| 14 | `10_mlflow_log` | Logs every run to the MLflow experiment `bug_triage_eval` |

The notebook numbers follow the order they were written, so the test-set notebooks (07, 09) run before the harness (06) that uses them.

Requirements: a Databricks workspace (Free Edition works) with the `databricks-gpt-oss-20b`, `databricks-gpt-oss-120b` and `databricks-llama-4-maverick` serving endpoints. Free Edition serves open models only.

## Limits and next steps

- **Routing labels are good, not perfect.** The answer key is where Mozilla's triagers filed each bug. Some components sit between teams, so a ceiling below 100% is expected.
- **Test sets predate the WPT fix.** About 4% of the routing test bugs are WPT reports, which are easy to route. The conclusions hold; exact numbers would shift by a point or so.
- **No frontier-model baseline.** Free Edition does not serve Claude, GPT or Gemini; a comparison would need an external API.
- **Next:** route to the exact component, then map to team; a fixed-only close rate in gold; an urgency definition agreed with the "customer", then a fresh urgency test set.

Data: [Mozilla Bugzilla](https://bugzilla.mozilla.org) public REST API.
