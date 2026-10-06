# SALUS: Automated Benchmark Auditing

🎉 **Accepted at SIGMOD 2027!**

SALUS automatically detects annotation errors in text-to-SQL benchmarks. It combines execution, SQL-structure, and question-alignment signals from multiple language-model agents into weak labels, then learns which agents are reliable for different SQL queries. No manually labeled examples are used for training.

This repository provides the SALUS pipeline, recorded agent SQL and execution outputs, and the BIRD-Clean-xs minibenchmark introduced in **SALUS: Automated Auditing of NL-to-SQL Benchmarks through Weak Supervision of Multi-Agent Output**.

## Quick start

Use Python 3.10 and the pinned dependencies. The commands below run locally on recorded agent outputs; no API keys, model calls, or database downloads are required.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_pipeline.py --config frontier
```

Run all four configurations:

```bash
python run_pipeline.py
```

The default run reuses the supplied labeling-function (LF) matrices and Snorkel weak labels, then selects and fits the downstream models. Saved artifacts are included for all four configurations in [src/data/weak_supervision](src/data/weak_supervision/). To rebuild the LF matrices and Snorkel outputs from the recorded agent outputs as well:

```bash
python run_pipeline.py --regenerate
```

Full regeneration can be substantially slower because it repeatedly compares large execution results. Add `--config frontier_november` (or another configuration below) to run only that configuration.

## Included configurations

All processed inputs are in `src/data/processed_query/`. They are losslessly compressed JSON files; the loader reads them directly without extracting them.

| Configuration | Benchmark | Tasks | Input |
|:--|:--|--:|:--|
| `frontier` | Previous BIRD development version used by SQLDriller | 1,534 | `bird_dev_frontier.json.gz` |
| `frontier_november` | BIRD development, November 2025 | 1,534 | `bird_dev_nov_4agent.json.gz` |
| `frontier_spider` | Spider development | 1,034 | `spider_dev_frontier.json.gz` |
| `legacy` | Previous BIRD development version used by SQLDriller | 1,534 | `bird_dev_legacy.json.gz` |

Frontier uses GPT-5, GPT-4o, Claude Opus 4.6, and Gemini 2.5 Pro. Legacy uses GPT-4o-mini, GPT-4o, Claude 3 Haiku, and Claude Haiku 4.5.

## BIRD-Clean-xs minibenchmark

**BIRD-Clean-xs** is our minibenchmark for evaluating annotation-error detection, rather than SQL generation. It contains **298 manually verified BIRD tasks: 166 erroneous and 132 correct**. We combine a 200-task random sample with 109 confirmed errors identified through benchmark-version differences, with 11 overlapping tasks.

[BIRD-Clean-xs](src/data/ground_truth/BIRD_CLEAN_xs.csv) covers incorrect SQL, query ambiguity, data inconsistency, misinterpreted domain semantics, and incorrect evidence. It supplies correctness labels and error categories; task IDs link to questions, reference SQL, and agent outputs in the `frontier` input.

## Detection results

| SALUS configuration | BIRD-Clean-xs F1 | BIRD random 200 F1 |
|:--|--:|--:|
| Frontier | 0.9194 | 0.8552 |
| Legacy | 0.8412 | 0.7389 |

The default setting is transductive: SALUS constructs weak labels and learns its decision plane on the benchmark being audited. Manual correctness labels are used only afterward to evaluate fixed predictions.

## Benchmark audit

Raw detection rates count SALUS error predictions. Calibrated estimates account for detection errors using a manually labeled random sample of 200 tasks from each corresponding benchmark version.

| Benchmark | Detected errors | Raw detected rate | Calibrated error rate | 95% interval |
|:--|--:|--:|--:|:--|
| Previous BIRD version used by SQLDriller | 656 / 1,534 | 42.8% | 39% | [33%, 45%] |
| BIRD November 2025 | 571 / 1,534 | 37.2% | 37% | [30%, 43%] |
| Spider | 325 / 1,034 | 31.4% | 27% | [21%, 33%] |

Calibration does not change SALUS predictions or train its models.

## Outputs and implementation

Runs write `reports/pipeline_report.json`, a readable `reports/pipeline_report.md`, `reports/metrics.json`, and per-task predictions in `reports/predictions/`. Use `--output-dir PATH` to keep separate runs.

- [Architecture walkthrough](docs/architecture.md) explains the pipeline and points to the implementation.
- [Data guide](docs/data.md) describes the input format, labels, and provenance.

- `python -m unittest discover -s tests` checks the packaged results against fixed expectations after running the pipeline and optional baseline evaluator.

## Citation

If you use SALUS or BIRD-Clean-xs, please cite our paper. The BibTeX entry will be added here.

```bibtex
% SALUS paper citation to be added.
```
