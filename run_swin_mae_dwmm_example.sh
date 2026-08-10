#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Swin-MAE + DWMM example
# Single-GPU LOSO self-supervised pretraining + fine-tuning
# ============================================================

DATA_PATH="${DATA_PATH:-./data/fatiguev2_105270}"
CFG="${CFG:-./configs/swin_mae_dwmm/swin_base_mae_dwmm_finetune_stable_img224.yaml}"

LOSO_SUBJECT="${LOSO_SUBJECT:-subject13}"
HELD_OUT_SUBJECT="${HELD_OUT_SUBJECT:-jiang_pengfei}"

PRETRAIN_OUTPUT="${PRETRAIN_OUTPUT:-./output/swin_mae_dwmm_pretrain/${LOSO_SUBJECT}}"
OUTPUT="${OUTPUT:-./output/swin_mae_dwmm_finetune/${LOSO_SUBJECT}}"

PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-100}"
PRETRAIN_WARMUP_EPOCHS="${PRETRAIN_WARMUP_EPOCHS:-10}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-8}"

FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-100}"
FINETUNE_WARMUP_EPOCHS="${FINETUNE_WARMUP_EPOCHS:-5}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-16}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-2}"

NUM_WORKERS="${NUM_WORKERS:-8}"
SKIP_GRADCAM="${SKIP_GRADCAM:-0}"

# Prefer the newest available training driver while remaining compatible
# with repositories that keep an older filename.
TRAIN_SCRIPT=""
for candidate in \
  swin_mae_dwmm_train_gradcam_speed_single_gpu_loso_fixed_recon_v5.py \
  swin_mae_dwmm_train_gradcam_speed_single_gpu_loso_fixed_recon_v4.py \
  swin_mae_dwmm_train_gradcam_speed_single_gpu_loso_fixed_recon_v3.py \
  swin_mae_dwmm_train_gradcam_speed_single_gpu_loso_fixed_recon.py
do
  if [[ -f "$candidate" ]]; then
    TRAIN_SCRIPT="$candidate"
    break
  fi
done

if [[ -z "$TRAIN_SCRIPT" ]]; then
  echo "[ERROR] Swin-MAE+DWMM training script was not found in the project root."
  exit 1
fi

if [[ ! -f "$CFG" ]]; then
  echo "[ERROR] Config not found: $CFG"
  exit 1
fi

if [[ ! -d "$DATA_PATH" ]]; then
  echo "[ERROR] Dataset directory not found: $DATA_PATH"
  echo "Set it with: DATA_PATH=/path/to/dataset bash $0"
  exit 1
fi

mkdir -p "$PRETRAIN_OUTPUT" "$OUTPUT"

EXTRA_ARGS=()
if [[ "$SKIP_GRADCAM" == "1" ]]; then
  EXTRA_ARGS+=(--skip-gradcam)
fi

echo "============================================================"
echo "Swin-MAE + DWMM example"
echo "Training script : $TRAIN_SCRIPT"
echo "Config          : $CFG"
echo "Data            : $DATA_PATH"
echo "LOSO            : $LOSO_SUBJECT"
echo "Pretrain output : $PRETRAIN_OUTPUT"
echo "Finetune output : $OUTPUT"
echo "============================================================"

torchrun --standalone --nproc_per_node=1 \
  "$TRAIN_SCRIPT" \
  --stage full \
  --cfg "$CFG" \
  --data-path "$DATA_PATH" \
  --held-out-subject "$HELD_OUT_SUBJECT" \
  --pretrain-output "$PRETRAIN_OUTPUT" \
  --pretrain-epochs "$PRETRAIN_EPOCHS" \
  --pretrain-warmup-epochs "$PRETRAIN_WARMUP_EPOCHS" \
  --pretrain-batch-size "$PRETRAIN_BATCH_SIZE" \
  --pretrain-workers "$NUM_WORKERS" \
  --pretrain-lr 0.0001 \
  --pretrain-min-lr 0.000001 \
  --pretrain-weight-decay 0.05 \
  --mask-ratio 0.75 \
  --mask-window 4 \
  --decoder-dim 256 \
  --pretrain-drop-path 0.10 \
  --pretrain-use-checkpoint \
  --deformable-offset-scale 1.0 \
  --pretrain-amp \
  --recon-preview-count 4 \
  --recon-save-every 1 \
  --recon-preview-seed 20260808 \
  --batch-size "$FINETUNE_BATCH_SIZE" \
  --accumulation-steps "$ACCUMULATION_STEPS" \
  --amp-opt-level O0 \
  --swin-mae-dwmm-model swin_base_patch4_window7_224_dwmm \
  --gradcam-max-per-class 8 \
  --output "$OUTPUT" \
  --tag swin_mae_dwmm_example \
  "${EXTRA_ARGS[@]}" \
  --opts \
  TRAIN.EPOCHS "$FINETUNE_EPOCHS" \
  TRAIN.WARMUP_EPOCHS "$FINETUNE_WARMUP_EPOCHS" \
  TRAIN.AUTO_RESUME False \
  DATA.NUM_WORKERS "$NUM_WORKERS" \
  DATA.LOSO_SUBJECT "$LOSO_SUBJECT"
