#!/bin/bash
# Generate Monte Carlo robustness GIFs for ALL datasets

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate main

cd "$PROJECT_ROOT"

# Common settings
MAX_ITERATIONS=15000
FRAME_ITERATIONS="50,100,200,500,1000,2000,5000,10000,15000"
SEED=42

echo "============================================================"
echo "Monte Carlo Robustness GIF Generator - All Datasets"
echo "============================================================"
echo ""

# ============================================================
# MOVIELENS
# ============================================================
echo "========================================"
echo "1/18: MovieLens - GPT-4o-mini"
echo "========================================"
python scripts/generate_monte_carlo_gif.py \
    --data benchmarks/movielens/data/merged_cleaned.csv \
    --output benchmarks/movielens/figures/gifs/monte_carlo_robustness.gif \
    --llm-col "GPT-4o-mini" \
    --min-score 1 --max-score 5 \
    --max-iterations $MAX_ITERATIONS \
    --frame-iterations "$FRAME_ITERATIONS" \
    --seed $SEED

# ============================================================
# WMT-HUMAN (3 models)
# ============================================================
echo ""
echo "========================================"
echo "2/18: WMT-Human - GPT"
echo "========================================"
python scripts/generate_monte_carlo_gif.py \
    --data benchmarks/wmt-human/data/wmt-human_en_de_judged_cleaned.csv \
    --output benchmarks/wmt-human/figures/gifs/monte_carlo_GPT-4o.gif \
    --llm-col "GPT-4o_as_a_judge" \
    --min-score 0 --max-score 6 \
    --max-iterations $MAX_ITERATIONS \
    --frame-iterations "$FRAME_ITERATIONS" \
    --seed $SEED

echo ""
echo "========================================"
echo "3/18: WMT-Human - LLAMA"
echo "========================================"
python scripts/generate_monte_carlo_gif.py \
    --data benchmarks/wmt-human/data/wmt-human_en_de_judged_cleaned.csv \
    --output benchmarks/wmt-human/figures/gifs/monte_carlo_Llama.gif \
    --llm-col "Llama_as_a_judge" \
    --min-score 0 --max-score 6 \
    --max-iterations $MAX_ITERATIONS \
    --frame-iterations "$FRAME_ITERATIONS" \
    --seed $SEED

echo ""
echo "========================================"
echo "4/18: WMT-Human - MISTRAL"
echo "========================================"
python scripts/generate_monte_carlo_gif.py \
    --data benchmarks/wmt-human/data/wmt-human_en_de_judged_cleaned.csv \
    --output benchmarks/wmt-human/figures/gifs/monte_carlo_Mistral.gif \
    --llm-col "Mistral_as_a_judge" \
    --min-score 0 --max-score 6 \
    --max-iterations $MAX_ITERATIONS \
    --frame-iterations "$FRAME_ITERATIONS" \
    --seed $SEED

# ============================================================
# NEWSROOM (4 metrics × 3 models = 12 GIFs)
# ============================================================
COUNT=5
for METRIC in informativeness relevance fluency coherence; do
    for MODEL in GPT-4o Llama Mistral; do
        echo ""
        echo "========================================"
        echo "${COUNT}/18: Newsroom - ${METRIC} - ${MODEL}"
        echo "========================================"
        python scripts/generate_monte_carlo_gif.py \
            --data benchmarks/newsroom/data/newsroom_judged_cleaned.csv \
            --output "benchmarks/newsroom/figures/gifs/monte_carlo_${METRIC}_${MODEL}.gif" \
            --llm-col "${MODEL}_${METRIC}_as_a_judge" \
            --metric "${METRIC}" \
            --min-score 1 --max-score 5 \
            --max-iterations $MAX_ITERATIONS \
            --frame-iterations "$FRAME_ITERATIONS" \
            --seed $SEED
        COUNT=$((COUNT + 1))
    done
done

# ============================================================
# POLITIFACT (legacy interface - separate files)
# ============================================================
echo ""
echo "========================================"
echo "17/18: PolitiFact S3"
echo "========================================"
python scripts/generate_monte_carlo_gif.py \
    --llm-ratings benchmarks/politifact/data/llm_ratings_truth_s3_gpt-4o-mini.csv \
    --human-judgments benchmarks/politifact/data/reliability_matrix_s3.csv \
    --output benchmarks/politifact/figures/gifs/monte_carlo_s3.gif \
    --llm-col "GPT-4o mini" \
    --min-score 0 --max-score 2 \
    --max-iterations $MAX_ITERATIONS \
    --frame-iterations "$FRAME_ITERATIONS" \
    --seed $SEED

echo ""
echo "========================================"
echo "18/18: PolitiFact S6"
echo "========================================"
python scripts/generate_monte_carlo_gif.py \
    --llm-ratings benchmarks/politifact/data/llm_ratings_truth_s6_gpt-4o-mini.csv \
    --human-judgments benchmarks/politifact/data/reliability_matrix_s6.csv \
    --output benchmarks/politifact/figures/gifs/monte_carlo_s6.gif \
    --llm-col "GPT-4o mini" \
    --min-score 0 --max-score 5 \
    --max-iterations $MAX_ITERATIONS \
    --frame-iterations "$FRAME_ITERATIONS" \
    --seed $SEED

# ============================================================
# SUMMARY
# ============================================================
echo ""
echo "============================================================"
echo "✅ All 18 GIFs generated!"
echo "============================================================"
echo ""
echo "Output files:"
echo ""
echo "MovieLens (1):"
echo "  - benchmarks/movielens/figures/monte_carlo_robustness.gif"
echo ""
echo "WMT-Human (3):"
echo "  - benchmarks/wmt-human/figures/gifs/monte_carlo_GPT-4o.gif"
echo "  - benchmarks/wmt-human/figures/gifs/monte_carlo_Llama.gif"
echo "  - benchmarks/wmt-human/figures/gifs/monte_carlo_Mistral.gif"
echo ""
echo "Newsroom (12):"
for METRIC in informativeness relevance fluency coherence; do
    for MODEL in GPT-4o Llama Mistral; do
        echo "  - benchmarks/newsroom/figures/gifs/monte_carlo_${METRIC}_${MODEL}.gif"
    done
done
echo ""
echo "PolitiFact (2):"
echo "  - benchmarks/politifact/figures/gifs/monte_carlo_s3.gif"
echo "  - benchmarks/politifact/figures/gifs/monte_carlo_s6.gif"

