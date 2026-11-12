#!/usr/bin/env bash
#
# End-to-end reproduction of the HSKM benchmark suite.
#
#   bash benchmarks/run_all.sh                # full run
#   QUICK=1 bash benchmarks/run_all.sh        # fast smoke run (minutes, not hours)
#
# Environment variables:
#   QUICK=1            small budgets / subset of models
#   TOKENS=100000000   training token budget for the controlled comparison
#   DEVICES=0,1        GPUs to use (the suite is split across them in parallel)
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
cd "${ROOT}"

QUICK="${QUICK:-0}"
TOKENS="${TOKENS:-100000000}"
DEVICES="${DEVICES:-0,1}"

if [[ "${QUICK}" == "1" ]]; then
  TOKENS=2_000_000
  EVAL_MODELS="gpt2,mamba-130m,smollm2-135m"
  SCALING_MODELS="hskm,gpt2-ctrl,gpt2,pythia-160m,mamba-130m"
  SEQ_LENS="512,1024,2048,4096"
  LIMIT="--limit 100"
else
  EVAL_MODELS="gpt2,opt-125m,pythia-160m,smollm2-135m,pythia-410m,smollm2-360m,opt-350m,qwen2.5-0.5b,mamba-130m,mamba2-130m,mamba2-1.3b,mamba-2.8b"
  SCALING_MODELS="hskm,gpt2-ctrl,gpt2,pythia-160m,mamba-130m,mamba2-130m"
  SEQ_LENS="512,1024,2048,4096,8192,16384"
  LIMIT=""
fi

echo "=============================================================="
echo " HSKM benchmark suite   QUICK=${QUICK}  TOKENS=${TOKENS}"
echo "=============================================================="

# ---------------------------------------------------------------- 0. tests
echo ""
echo ">>> [0/7] Verifying the harness itself (unit tests)"
python3 -m pytest benchmarks/tests -q

# ------------------------------------------------------------- 1. downloads
echo ""
echo ">>> [1/7] Downloading model checkpoints"
python3 benchmarks/scripts/01_download_models.py

echo ""
echo ">>> [1b/7] Recording model configuration details"
python3 benchmarks/scripts/05_dump_configs.py

# ------------------------------------------------------------- 2. training
echo ""
echo ">>> [2/7] Controlled from-scratch training (HSKM vs GPT-2)"
# The two models can be trained concurrently on separate GPUs. They are
# independent runs with an identical budget, so this is safe and roughly
# halves the wall-clock time:
#
#   CUDA_VISIBLE_DEVICES=0 python3 benchmarks/scripts/10_train_controlled.py \
#       --model hskm --tokens ${TOKENS} --summary-name controlled_hskm.json &
#   CUDA_VISIBLE_DEVICES=1 python3 benchmarks/scripts/10_train_controlled.py \
#       --model gpt2 --tokens ${TOKENS} --summary-name controlled_gpt2.json &
#   wait
#
python3 benchmarks/scripts/10_train_controlled.py --model both --tokens "${TOKENS}"

# ------------------------------------------------------------ 3. lm-eval
echo ""
echo ">>> [3/7] lm-evaluation-harness"
python3 benchmarks/scripts/20_run_lm_eval.py --models "${EVAL_MODELS},hskm-ours,gpt2-ours" ${LIMIT}

# ------------------------------------------------------------ 4. perplexity
echo ""
echo ">>> [4/7] Perplexity"
python3 benchmarks/scripts/30_perplexity.py --models all --context 512 --max-words 500000

# -------------------------------------------------------------- 5. scaling
echo ""
echo ">>> [5/7] Scaling benchmark (O(N) vs O(N^2))"
python3 benchmarks/scripts/40_scaling_benchmark.py --models "${SCALING_MODELS}" --seq-lens "${SEQ_LENS}"

# --------------------------------------------------------------- 6. report
echo ""
echo ">>> [6/7] Building tables and figures"
python3 benchmarks/scripts/50_make_report.py

echo ""
echo "=============================================================="
echo " Done.  See benchmarks/results/REPORT.md"
echo "=============================================================="
