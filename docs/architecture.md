# SALUS architecture

SALUS audits a benchmark's reference annotations, rather than treating those annotations as trusted training labels. This release starts from recorded candidate SQL and execution results and runs weak labeling and the decision plane locally.

## Agent outputs and verdicts

Each task supplies the question, evidence, reference SQL, and database ID. Each agent supplies candidate SQL and its execution result. An agent verdict indicates whether its valid execution agrees with the reference execution: 1 means agreement, and 0 means otherwise. Agreement is a detection signal, not proof that the reference is correct.

The input adapter is [labeling/base.py](../src/sqlsalus_core/labeling/base.py). Execution-result comparison is in [execution/comparator.py](../src/sqlsalus_core/execution/comparator.py).

## Weak labeling

The [LF registry](../src/sqlsalus_core/labeling/registry.py) contains 43 task-level labeling functions and one meta LF. Execution signals compare results and agent consensus. Structural signals inspect SQL structure, schema consistency, and known error patterns. Intent signals check question and SQL alignment. The meta LF summarizes the other votes.

Each LF emits correct (1), incorrect (0), or abstain (-1). [generate_lf_matrices.py](../src/sqlsalus_core/generate_lf_matrices.py) constructs the task-by-LF matrix. The Snorkel generative label model aggregates that noisy matrix into correctness probabilities. SALUS selects the highest-confidence 250 examples of each class, creating 500 weakly labeled training samples without manual correctness labels.

The orchestration and high-confidence selection are in [run_pipeline.py](../run_pipeline.py). Saved matrices and label-model outputs are in [src/data/weak_supervision](../src/data/weak_supervision/).

## SQL conditioned reliability and aggregation

For each reference SQL, SALUS extracts 12 features: six structural flags, four structural counts, result cardinality, and an execution-error indicator. The flags cover DISTINCT, GROUP BY, HAVING, ORDER BY, LIMIT, and subqueries. The counts cover joins, tables, aggregate functions, and WHERE clauses.

A separate model learns each agent's reliability from agreement between its verdict and the high-confidence weak label. A fallback model learns task correctness directly from the same SQL features. Model family and aggregation strategy are selected by five-fold cross-validation on the high-confidence weak labels, then fitted on all 500 samples.

[decision_plane.py](../src/sqlsalus_core/decision_plane.py) implements these models and aggregation strategies. In the Frontier BIRD configuration, depth-2 CART reliability models are paired with score fusion: a logistic model combines the reliability-weighted verdict score and the fallback model's correctness probability. The decision plane is therefore not simply majority voting or always selecting one agent.

## Predictions and evaluation

The selected decision plane produces one correctness prediction per task. Only after fitting are predictions joined with manual correctness labels for evaluation. The random 200-task samples also calibrate benchmark-wide error-rate estimates; calibration does not alter predictions.

[stacking.py](../src/sqlsalus_core/stacking.py) implements the Agent Stacking comparison. It uses the same weak training labels but replaces SQL-conditioned routing with one global mapping of four-agent verdict patterns to correctness labels. It is distinct from the decision plane's stacked meta-learner.

[run_pipeline.py](../run_pipeline.py) exports prediction CSVs, detection metrics, and calibrated audit estimates. [evaluate_baselines.py](../evaluate_baselines.py) independently evaluates recorded external-system outputs; it does not run those systems.
