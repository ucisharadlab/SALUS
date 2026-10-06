# Recorded baseline outputs

Evaluate recorded automatic SQLDriller and SAR-Agent outputs on BIRD-Clean-xs (298 tasks) and its random 200-task subset. The evaluator uses the bundled outputs without running either baseline system.

Run:

```bash
python evaluate_baselines.py
```

Results are written to `reports/baselines/metrics.json` and `reports/baselines/results.md`. The script checks that every evaluation task has an explicit prediction.

## SQLDriller

[src/data/baselines/sqldriller](../src/data/baselines/sqldriller/) contains the development-set automatic outputs from the released `prepared/bird/dataset_refine` artifact:

- `issues/dev/modified_gold.tsv`: task ID, automatic execution-consistency verdict, original SQL, and proposed replacement SQL. The second field is the prediction, with 0 meaning an error. A missing replacement (`-`) is not a correctness verdict.
- `sqls/dev_candidates.json`: the released candidate SQL records.
- `issues/dev/exec_res/` and `logs/dev/`: available execution and counterexample outputs.

Traceback file paths omit machine-specific directory prefixes. Verdicts, SQL, and execution results are unchanged.

Only automatic verdicts are scored, not the manually reviewed labels in `stats/dev.tsv`. The LLM backbone used to produce these predictions is not specified in the released outputs.

The upstream SQLDriller repository's license is retained in this directory.

## SAR Agent

[src/data/baselines/sar_agent](../src/data/baselines/sar_agent/) contains the automatic o3 run:

- `analyze_result/<task_id>/final_analyze_result.txt`: generated final reports, including correctness, ambiguity, explanation, and proposed SQL where present.
- `analyze_result/<task_id>/query_result.txt`: generated query-result files.
- `manifest.json`: model, run settings, and output coverage.

The evaluator reads the report's `Correctness:` field, before SAPAR's subsequent expert-review stage.
