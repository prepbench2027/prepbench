# PrepBench Artifact

This artifact accompanies the VLDB EA&B submission:

**PrepBench: A Factorized Benchmark for Cost-Aware Data-Preparation Pipeline Search [Experiment, Analysis & Benchmark]**

PrepBench is a benchmark and measurement protocol for cost-aware data-preparation pipeline search. The artifact contains the code, released result CSVs, analysis scripts, and figure-generation scripts needed to reproduce the paper's tables, figures, and revision-sensitivity checks.

## What This Artifact Supports

The released files support the following claims from the paper:

- The main controlled matrix over 56 OpenML tabular classification tasks, 9 configurations, 5 seeds, and 60s/300s/600s wall-clock budgets.
- Random feasible controls are strong baselines at the primary 300s and 600s budgets.
- Critical-path costing, staged evaluation, and budget-aware pruning change allocation but are statistically equivalent to the plain DAG baseline in final balanced accuracy.
- The zero-penalty sensitivity: removing the estimated-cost and graph-complexity penalties does not reverse the main ranking.
- The stronger guided-EA sensitivity: increasing the guided EA population and offspring size brings the plain guided DAG into the random-control score range at 300s, while the combined cost-aware variant remains below the plain DAG.
- Candidate-level audits for staged screening, random-space fairness, cost-model calibration, completed-evaluation-only robustness, and the external CtxPipe reference.

## Directory Layout

```text
prepbench-vldb-eab-artifact/
  README.md
  requirements.txt

  paper/
    paper-vldb.pdf
    supplementary_prepbench.pdf

  code/
    baselines/
    parallel_evolutionary_pipeline/
    data_loader/
    experiments/
    figure_scripts/
    scripts/

  results/
    main_matrix/
      eab60/
      eab300/
      eab600/
      paper_result_bundle/

    sensitivity/
      zpen300/
      zpen600/
      strong_ea300/
      zero_penalty_audit/
      strongEA_compare/

    external_baselines/
      ctxpipe_scores.csv
      ctxpipe_paired_comparison.csv

    trace_summary/
      status_summary.csv
      per_run_incubent.csv
      staged_metrics.csv
      staged_full_sample.csv

  figures/
  data/
    deepline_dataset/
    manifest.csv
```

## Quick Start

Create the environment:

```bash
conda create -n prepbench python=3.12
conda activate prepbench
pip install -r requirements.txt
```

If you do not use Conda:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Regenerate the paper result bundle from released CSVs:

```bash
python code/experiments/make_paper_result_bundle.py \
  --results results/main_matrix \
  --out /tmp/prepbench_result_check
```

Regenerate the revision-sensitivity summaries:

```bash
python code/experiments/summarize_sensitivity_runs.py \
  --root results/sensitivity \
  --out /tmp/prepbench_sensitivity_check
```

Regenerate figures:

```bash
python code/figure_scripts/cd_diagram_600.py
python code/figure_scripts/incumbent_convergence_600.py
python code/figure_scripts/budget_delta_score_vs_evals.py
python code/figure_scripts/staged_calibration_dctb_600.py
python code/figure_scripts/efficiency_frontier_600.py
python code/figure_scripts/cost_runtime_calibration.py
```

The regenerated outputs should match the released `results/` and `figures/` files up to small formatting differences in generated PDFs.

## Smoke Test

The smoke test checks that the benchmark runner, dataset loader, and result writers work. It is not intended to reproduce the paper numbers.

```bash
python code/experiments/run_matrix.py \
  --budget 30 \
  --seeds 42 \
  --configs D-Rand D-S-F-A \
  --out /tmp/prepbench_smoke \
  --processes 2 \
  --per-eval-timeout 10 \
  --run-tag smoke
```

Expected output files:

```text
/tmp/prepbench_smoke/
  experiment_records.csv
  run_metrics.csv
  search_history.csv
  summary_by_config.csv
  run_manifest.json
```

## Full Reproduction

Full raw reproduction is computationally expensive. It reruns 56 datasets, 5 seeds, 9 configurations, and three wall-clock budgets, plus the revision-sensitivity blocks.

Example main-matrix run:

```bash
python code/experiments/run_matrix.py \
  --budget 600 \
  --seeds 42 114 256 512 768 \
  --out results/main_matrix/eab600 \
  --processes 32 \
  --per-eval-timeout 10 \
  --run-tag main_600
```

Example zero-penalty sensitivity run:

```bash
python code/experiments/run_matrix.py \
  --budget 600 \
  --seeds 42 114 256 512 768 \
  --configs D-Rand D-S-F-A D-C-T-B \
  --out results/sensitivity/zpen600 \
  --processes 64 \
  --per-eval-timeout 10 \
  --fitness-cost-penalty 0 \
  --fitness-complexity-penalty 0 \
  --run-tag zero_penalty_600
```

Example stronger guided-EA run:

```bash
python code/experiments/run_matrix.py \
  --budget 300 \
  --seeds 42 114 256 512 768 \
  --configs D-S-F-A D-C-T-B \
  --out results/sensitivity/strong_ea300 \
  --processes 64 \
  --per-eval-timeout 10 \
  --fitness-cost-penalty 0 \
  --fitness-complexity-penalty 0 \
  --population-size 40 \
  --offspring-size 40 \
  --run-tag strong_ea300
```

Runtime depends heavily on hardware, parallelism, and dataset cache state. The released CSVs are included so reviewers can reproduce the paper analyses without rerunning the full benchmark.

## Result-to-Claim Mapping

```text
Main controlled matrix:
  results/main_matrix/eab*/
  results/main_matrix/paper_result_bundle/

Zero-penalty sensitivity:
  results/sensitivity/zpen300/
  results/sensitivity/zpen600/
  results/sensitivity/sensitivity_summary_by_config.csv
  results/sensitivity/sensitivity_paired_tests.csv

Stronger guided EA:
  results/sensitivity/strong_ea300/
  results/sensitivity/sensitivity_summary_by_config.csv
  results/sensitivity/sensitivity_paired_tests.csv

Random-space fairness audit:
  results/sensitivity/zero_penalty_audit/random_fairness_audit.csv

Staged-screening diagnostics:
  results/main_matrix/*/search_history.csv
  results/trace_summary/stage_full_sample.csv
  results/trace_summary/staged_metrics.csv
  figures/staged_calibration_dctb_600.pdf

Cost-model calibration:
  results/main_matrix/*/search_history.csv
  figures/cost_runtime_calibration.pdf

External CtxPipe reference:
  results/external_baselines/ctxpipe_result.tsv
  results/external_baselines/ctxpipepines.json
```

## Data

The benchmark uses 56 public OpenML tabular classification tasks inherited from prior data-preparation search work. The file `data/manifest.csv` records each dataset name, OpenML task id, data id, csv path info path, target column, source url and statistic features.

If the dataset directory is not available in the local folder, set a custom data dir with:

```bash
export PREPBENCH_DATA_DIR=/data/deepline_dataset
```

## Reproducibility Notes

- The statistical unit is the dataset. Scores are averaged across the five seeds within each dataset/configuration/budget cell before paired tests.
- The main equivalence margin is 0.01 balanced accuracy; the released result bundle also includes sensitivity at 0.005 and 0.02.
- The staged-screening diagnostic reports an incumbent-threat rate, not a true false-negative rate, because screened candidates are not normally full-evaluated.
- CtxPipe is included only as an external reference. It is not treated as a controlled ablation because it uses different policy, implementation, and time-accounting assumptions.
- Wall-clock budgets depend on hardware and parallelism. For this reason, the artifact reports elapsed time, full evaluations, screened candidates, pruned candidates, and candidate-level trace diagnostics.

## Troubleshooting

If a full run produces many timeouts, reduce `--processes` or increase `--per-eval-timeout`. The smoke test should still produce `experiment_records.csv`, `run_metrics.csv`, and `search_history.csv`.

If generated table values differ slightly from the paper, check that the same seeds, budgets, configuration list, penalty settings, and population/offspring settings are recorded in `run_manifest.json`.
