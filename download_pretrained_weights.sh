#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Download pretrained weights used by comparison backbones.
#
# Usage:
#   bash download_pretrained_weights.sh
#   bash download_pretrained_weights.sh ./pretrained standard
#   bash download_pretrained_weights.sh ./pretrained all
#
# standard: supervised/general-purpose backbone weights
# all:      standard + SimMIM/MixMAE self-supervised checkpoints
# ============================================================

DEST="${1:-./pretrained}"
MODE="${2:-standard}"

mkdir -p "$DEST"
cd "$DEST"

download() {
  local url="$1"
  local output="$2"
  echo
  echo "[DOWNLOAD] $output"
  wget -c --tries=5 --timeout=30 "$url" -O "$output"
}

echo "Saving pretrained weights to: $(pwd)"
echo "Mode: $MODE"

# ------------------------------------------------------------
# Standard comparison backbones
# ------------------------------------------------------------

# ConvNeXt-Base, ImageNet-1K, official Meta checkpoint
download \
  "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth" \
  "convnext_base_1k_224_ema.pth"

# DeiT-Base, ImageNet-1K, official Meta checkpoint
download \
  "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth" \
  "deit_base_patch16_224-b5f2ef4d.pth"

# DINOv2 ViT-S/14, official Meta backbone
download \
  "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth" \
  "dinov2_vits14_pretrain.pth"

# ResNet50 ImageNet-1K V2, official TorchVision checkpoint
download \
  "https://download.pytorch.org/models/resnet50-11ad3fa6.pth" \
  "resnet50-11ad3fa6.pth"

# Swin-Base, ImageNet-1K, official Microsoft checkpoint
download \
  "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth" \
  "swin_base_patch4_window7_224.pth"

# CoAtNet-0, timm ImageNet-1K
download \
  "https://huggingface.co/timm/coatnet_0_rw_224.sw_in1k/resolve/main/pytorch_model.bin" \
  "coatnet_0_rw_224_sw_in1k.bin"

# EdgeNeXt-Small, timm / author USI ImageNet-1K
download \
  "https://huggingface.co/timm/edgenext_small.usi_in1k/resolve/main/pytorch_model.bin" \
  "edgenext_small_usi_in1k.bin"

# MobileViTv2-1.0, timm / CVNets ImageNet-1K
download \
  "https://huggingface.co/timm/mobilevitv2_100.cvnets_in1k/resolve/main/pytorch_model.bin" \
  "mobilevitv2_100_cvnets_in1k.bin"

# DeiT III Base, ImageNet-22K -> ImageNet-1K, timm model file
download \
  "https://huggingface.co/timm/deit3_base_patch16_224.fb_in22k_ft_in1k/resolve/main/model.safetensors" \
  "deit3_base_patch16_224_fb_in22k_ft_in1k.safetensors"

# SwinV2-CR Small NS, timm ImageNet-1K
download \
  "https://huggingface.co/timm/swinv2_cr_small_ns_224.sw_in1k/resolve/main/model.safetensors" \
  "swinv2_cr_small_ns_224_sw_in1k.safetensors"

if [[ "$MODE" == "all" ]]; then
  echo
  echo "Installing gdown for official Google Drive checkpoints..."
  python -m pip install -q gdown

  # MixMAE / MixMIM Swin-B/W14, 600-epoch official pretraining checkpoint
  gdown \
    "https://drive.google.com/uc?id=1pZYmTv08xK_kOe2kk6ahuvgJVkHm-ZIa" \
    -O "mixmae_swin_base_600ep.pth"

  # Official SimMIM checkpoints
  gdown \
    "https://drive.google.com/uc?id=1Wcbr66JL26FF30Kip9fZa_0lXrDAKP-d" \
    -O "simmim_pretrain_swin_base_100ep.pth"

  gdown \
    "https://drive.google.com/uc?id=15zENvGjHlM71uKQ3d2FbljWPubtrPtjl" \
    -O "simmim_pretrain_swin_base_800ep.pth"

  gdown \
    "https://drive.google.com/uc?id=1qDxrTl2YUDB0505_4QrU5LU2R1kKmcBP" \
    -O "simmim_pretrain_swin_large_800ep.pth"

  gdown \
    "https://drive.google.com/uc?id=1dJn6GYkwMIcoP3zqOEyW1_iQfpBi8UOw" \
    -O "simmim_pretrain_vit_base_800ep.pth"
fi

echo
echo "============================================================"
echo "Downloaded files"
echo "============================================================"
ls -lh
