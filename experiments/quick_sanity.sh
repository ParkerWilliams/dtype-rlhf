#!/bin/bash
# Quick sanity test - should complete in <5 minutes
# Tests that the basic pipeline works

set -e

echo "=== Quick Sanity Test ==="
echo "Testing single config with minimal steps..."

python scripts/run_ppo_sweep.py \
    --precision_config bf16_pure \
    --use_synthetic_reward \
    --max_steps 50 \
    --batch_size 4 \
    --seed 0 \
    --output_dir ./test_results

echo ""
echo "=== Verifying outputs ==="
cat ./test_results/runs/bf16_pure_seed0/run_summary.json | python -m json.tool

echo ""
echo "=== Sanity test passed! ==="
