#!/bin/bash

# Updated experiments script
# Already completed: hurricane_eal_counties + qwen25_coder
# Remaining: 5 experiments

echo "Running 5 remaining experiments..."
echo "Estimated total time: ~2.5 hours (30 min per experiment)"
echo ""

# ========================================
# QWEN2.5 CODER - New Datasets (2 experiments)
# ========================================

# 1. Inland Flood EAL + Qwen2.5 Coder
echo "=========================================="
echo "1/5: inland_flood_eal + qwen25_coder"
echo "=========================================="
python trial.py \
    --task tasks/inland_flood_eal.json \
    --llm-config llm_configs/qwen25_coder.json \
    --n-models 250 \
    --preload-codes-dir paper_results/inland_flood_eal/qwen25_coder/codes \
    --loo-kl-reference bma \
    --loo-lambda-reg 0.5 \
    --cache-posteriors

echo ""
echo "✓ 1/5 complete"
echo ""

# 2. Wildfire EAL West + Qwen2.5 Coder
echo "=========================================="
echo "2/5: wildfire_eal_west + qwen25_coder"
echo "=========================================="
python trial.py \
    --task tasks/wildfire_eal_west.json \
    --llm-config llm_configs/qwen25_coder.json \
    --n-models 250 \
    --preload-codes-dir paper_results/wildfire_eal_west/qwen25_coder/codes \
    --loo-kl-reference bma \
    --loo-lambda-reg 0.5 \
    --cache-posteriors

echo ""
echo "✓ 2/5 complete"
echo ""

# ========================================
# GEMMA4 - All Three Datasets (3 experiments)
# ========================================

# 3. Hurricane EAL + Gemma4
echo "=========================================="
echo "3/5: hurricane_eal_counties + gemma4_e4b"
echo "=========================================="
python trial.py \
    --task tasks/hurricane_eal_counties.json \
    --llm-config llm_configs/gemma4.json \
    --n-models 250 \
    --preload-codes-dir paper_results/hurricane_eal_counties/gemma4_e4b/codes \
    --loo-kl-reference bma \
    --loo-lambda-reg 0.5 \
    --cache-posteriors

echo ""
echo "✓ 3/5 complete"
echo ""

# 4. Inland Flood EAL + Gemma4
echo "=========================================="
echo "4/5: inland_flood_eal + gemma4_e4b"
echo "=========================================="
python trial.py \
    --task tasks/inland_flood_eal.json \
    --llm-config llm_configs/gemma4.json \
    --n-models 250 \
    --preload-codes-dir paper_results/inland_flood_eal/gemma4_e4b/codes \
    --loo-kl-reference bma \
    --loo-lambda-reg 0.5 \
    --cache-posteriors

echo ""
echo "✓ 4/5 complete"
echo ""

# 5. Wildfire EAL West + Gemma4
echo "=========================================="
echo "5/5: wildfire_eal_west + gemma4_e4b"
echo "=========================================="
python trial.py \
    --task tasks/wildfire_eal_west.json \
    --llm-config llm_configs/gemma4.json \
    --n-models 250 \
    --preload-codes-dir paper_results/wildfire_eal_west/gemma4_e4b/codes \
    --loo-kl-reference bma \
    --loo-lambda-reg 0.5 \
    --cache-posteriors

echo ""
echo "=========================================="
echo "All 5 experiments complete!"
echo "=========================================="
echo ""
echo "Completed experiments:"
echo "  QWEN2.5 Coder:"
echo "    ✓ hurricane_eal_counties (already done)"
echo "    ✓ inland_flood_eal"
echo "    ✓ wildfire_eal_west"
echo ""
echo "  GEMMA4:"
echo "    ✓ hurricane_eal_counties"
echo "    ✓ inland_flood_eal"
echo "    ✓ wildfire_eal_west"
echo ""
echo "Results saved to: experiment_results_anant/"
echo "Cached posteriors in: cached_posteriors/"
echo ""