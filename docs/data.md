# SALUS data

The four files in [src/data/processed_query](../src/data/processed_query/) contain the recorded inputs for the published SALUS configurations. Lossless gzip compression retains the original JSON fields and execution rows. Packaged paths, configurations, and uncompressed checksums are recorded in [manifest.json](../src/data/manifest.json).

## Task records

Each JSON file is a list of tasks with these fields:

| Field | Meaning |
|:--|:--|
| `task_id`, `db_id` | Task identity within its benchmark version and database |
| `question`, `evidence`, `difficulty` | Benchmark task information, where supplied |
| `gold_sql` | The reference SQL being audited, not a trusted correctness label |
| `gold_result` | Recorded reference execution rows, columns, row count, and error |
| `agents` | Agent-name mapping to candidate SQL and recorded execution result |

Frontier agent keys are `gpt5`, `gpt4o`, `claude`, and `gemini`. Legacy keys are `gpt4omini`, `gpt4o_old`, `haiku3`, and `haiku`. The fixed configuration determines their order. Do not join different benchmark versions using task ID alone: reference queries can change between releases.

## Evaluation labels

[src/data/ground_truth](../src/data/ground_truth/) contains:

- `BIRD_CLEAN_xs.csv`: BIRD-Clean-xs, 298 tasks.
- `GT_RS_200.csv`: the original BIRD random 200-task sample, contained in BIRD-Clean-xs.
- `GT_RS_200_nov_bird.csv`: 200 tasks labeled for the November 2025 BIRD release.
- `GT_RS_200_spider.csv`: 200 tasks labeled for Spider.

In every evaluation file, `query_id` joins to `task_id`, 0 means erroneous, and 1 means correct. F1, precision, and recall treat annotation error as the positive class. Manual labels are used only for evaluation and post-prediction calibration. Manual-review reports and human-verified baseline reports are not included.

In BIRD-Clean-xs, the CSV category `Incorrect Assumption` corresponds to *Misinterpreting Domain Semantics* in the paper; the original labels are retained.

## Running the data

The packaged agent outputs contain the execution results required by SALUS, so the default run needs neither live model access nor database files. It is an offline run of the auditing pipeline from recorded outputs, not a fresh run of the source agents.

The included Snorkel artifacts allow downstream fitting without recomputing weak labels. `python run_pipeline.py --regenerate` recomputes the LF matrices and Snorkel outputs before fitting. Keep the pinned numerical dependencies to reproduce confidence ordering and tied mapping selection.

For example, [frontier_november](../src/data/weak_supervision/frontier_november/) includes the 1,534-task, 44-column LF matrix, task IDs, weak labels, class probabilities, and 500 high-confidence samples. `python run_pipeline.py --config frontier_november` uses these saved artifacts and refits the decision plane; a full LF rebuild is not required to run the November benchmark.

Benchmark data originate from BIRD and Spider. Recorded external-system outputs retain their own provenance in [baseline_outputs.md](baseline_outputs.md); including these files does not change the upstream data's applicable terms.

## Git storage

This release uses ordinary Git, not Git LFS. Its largest file is the compressed Legacy input at approximately 37.3 MiB, below GitHub's 50 MiB warning and 100 MiB hard file limit. Files over 25 MiB cannot be uploaded through GitHub's browser interface, so publish this repository with Git push or GitHub Desktop. See [GitHub's file-size guidance](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

Keep the inputs compressed. `.gitattributes` marks compressed inputs and NumPy artifacts as binary, and `.gitignore` excludes extracted input JSON copies. If future datasets grow substantially or change frequently, distribute them through versioned GitHub Releases or configure Git LFS before committing those files. No LFS client is needed for the current release.
