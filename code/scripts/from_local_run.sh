python baselines/run_external_baselines_from_local.py \
    --manifest results/manifest.csv \
    --out results/external_out \
    --systems rf_default,logreg_default,flaml \
    --budgets 300 \
    --seeds 42 114 256 512 768 \
    --pool-size 6
