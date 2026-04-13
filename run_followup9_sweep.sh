#!/usr/bin/env bash
# theorem2_followup_9: Figures 1/2/3 for the Theorem-2 hypothesis paper figure.
#
# Basic setup (followup8): d=256, n_S=1024, n_I=n_T=512, L=8192, Exp(1)+N(0,0.1^2),
# k=16, LR=5e-4, 10 epochs, 1 seed.
#
# Part A (Fig 1 — interference, naive single decoder only):
#     single_recon on α ∈ {0.1, 0.2, ..., 0.9}            ⇒ 9 runs
#
# Part B (Fig 2 — dimensionality reduction, single decoder + aux loss):
#     single_paired_align at α=0.7 (middle regime), λ ∈ {0, 0.5, 2, 8}
#     NOTE: λ=0 case is redundant with single_recon; we run it here anyway
#     as a sanity check that single_paired_align reduces to single_recon
#     at λ=0.                                              ⇒ 4 runs
#
# Part C (Fig 3 — solution: four methods over α sweep):
#     4 methods × α ∈ {0.1, ..., 0.9}
#     Methods: single_recon, single_paired_align (λ=2),
#              two_recon, ours                              ⇒ 9 runs (all 4 methods in one call per α)
#
# Total wall-clock: ~30–45 minutes on A100.

set -uo pipefail
cd /mnt/working/lvlm_hallucination
source .venv/bin/activate
mkdir -p .log outputs/theorem2_followup_9

LOG=.log/followup9_sweep.log
echo "===== followup9 start $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" >> "$LOG"

# Shared basic-setup flags
BASIC="\
--lr 5e-4 --num-epochs 10 --num-train 50000 --num-eval 10000 --batch-size 256 \
--n-shared 1024 --n-image 512 --n-text 512 --representation-dim 256 \
--num-seeds 1 --seed-base 1 \
--shared-coeff-mode independent --coeff-dist exponential --cmin 0.0 --beta 1.0 \
--max-interference 0.1 --obs-noise-std 0.1 \
--k 16 --latent-size-sweep 8192 \
--device cuda --output-root outputs/theorem2_followup_9"

# Common ours config (same as followup8 primary)
OURS="--lambda-aux-sweep 2.0 --m-s-sweep 1024 --k-align-sweep 6 --aux-norm-sweep global"

# Common baseline lambdas
BASELINE_LAMS="--group-sparse-lambda 0.05 --trace-beta 1e-4 --iso-align-beta 0.03"

run () {
  local TAG=$1; shift
  echo "----- ${TAG} start $(date -u +%H:%M:%SZ) -----" >> "$LOG"
  python synthetic_theorem2_method.py $BASIC $OURS $BASELINE_LAMS "$@" --run-tag "$TAG" >> "$LOG" 2>&1
  echo "----- ${TAG} DONE $(date -u +%H:%M:%SZ) -----" >> "$LOG"
}

# ==================================================================
# Part A — Fig 1: single_recon, fine-grained alpha sweep
# ==================================================================
run "fu9_partA_fig1_single_alpha_sweep" \
    --alpha-sweep "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9" \
    --methods "single_recon"

# ==================================================================
# Part B — Fig 2: single_paired_align at alpha=0.7, lambda sweep
# ==================================================================
for LAM in 0.0 0.5 2.0 8.0; do
  run "fu9_partB_fig2_single_align_lam${LAM}" \
      --alpha-sweep "0.7" \
      --methods "single_paired_align" \
      --single-paired-align-lambda $LAM \
      --single-paired-align-mS 1024 \
      --single-paired-align-norm global
done

# ==================================================================
# Part C — Fig 3: four methods, fine-grained alpha sweep
# single_recon, single_paired_align (lam=2), two_recon, ours
# ==================================================================
run "fu9_partC_fig3_four_methods_alpha_sweep" \
    --alpha-sweep "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9" \
    --methods "single_recon,single_paired_align,two_recon,ours" \
    --single-paired-align-lambda 2.0 \
    --single-paired-align-mS 1024 \
    --single-paired-align-norm global

echo "===== followup9 DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" >> "$LOG"
