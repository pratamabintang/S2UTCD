#!/usr/bin/env bash
# PASCAL-5i end-to-end: deps -> VOC2012 + SegmentationClassAug -> SAM2 ckpt -> train -> test
set -euo pipefail

REPO_URL="${REPO_URL:-}"
CLONE_DIR="${CLONE_DIR:-$HOME/PF}"

if [ -n "$REPO_URL" ]; then
  echo "[1/7] git clone $REPO_URL -> $CLONE_DIR"
  if [ -d "$CLONE_DIR/.git" ]; then
    git -C "$CLONE_DIR" pull --ff-only
  else
    git clone "$REPO_URL" "$CLONE_DIR"
  fi
  ROOT="$CLONE_DIR"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$ROOT"

DPATH="${DPATH:-$ROOT/data/VOCdevkit}"
VOC="$DPATH/VOC2012"
LOGPATH="${LOGPATH:-log_pascal_fold2_1shot}"
FOLD="${FOLD:-2}"
SHOT="${SHOT:-1}"
BSZ="${BSZ:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
NITER="${NITER:-20000}"
NWORKER="${NWORKER:-4}"
LR="${LR:-1e-3}"
ADAPTER_LR="${ADAPTER_LR:-1e-4}"
REDUCE_DIM="${REDUCE_DIM:-64}"
ATTN_MAX_TOKENS="${ATTN_MAX_TOKENS:-4096}"
IMG_SIZE="${IMG_SIZE:-512}"
AMP="${AMP:-1}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
VISUALIZE="${VISUALIZE:-}"
SAM2_CKPT="${SAM2_CKPT:-$ROOT/pretrained/sam2_hiera_large.pt}"

VOC_URL="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
AUG_URL="https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=1"
SAM2_URL="https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"

echo "[2/7] install python deps"
python -c "import torch" 2>/dev/null || \
  pip -q install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip -q install -r "$ROOT/requirements.txt"

echo "[3/7] download PASCAL VOC2012 devkit"
mkdir -p "$ROOT/data"
if [ ! -d "$VOC/JPEGImages" ]; then
  cd "$ROOT/data"
  [ -f VOCtrainval_11-May-2012.tar ] || wget -c "$VOC_URL"
  tar -xf VOCtrainval_11-May-2012.tar
  cd "$ROOT"
fi

echo "[4/7] download SegmentationClassAug"
if [ ! -d "$VOC/SegmentationClassAug" ]; then
  cd "$ROOT/data"
  [ -f SegmentationClassAug.zip ] || wget -c "$AUG_URL" -O SegmentationClassAug.zip
  unzip -q -o SegmentationClassAug.zip
  mv -f SegmentationClassAug "$VOC/SegmentationClassAug"
  cd "$ROOT"
fi

echo "[5/7] download SAM2 checkpoint"
mkdir -p "$(dirname "$SAM2_CKPT")"
[ -f "$SAM2_CKPT" ] || wget -c "$SAM2_URL" -O "$SAM2_CKPT"

echo "[6/7] train"
python train_pascal.py \
  --datapath "$DPATH" \
  --benchmark pascal \
  --logpath "$LOGPATH" \
  --bsz "$BSZ" \
  --grad_accum "$GRAD_ACCUM" \
  --lr "$LR" \
  --adapter_lr "$ADAPTER_LR" \
  --niter "$NITER" \
  --nworker "$NWORKER" \
  --reduce_dim "$REDUCE_DIM" \
  --attn_max_tokens "$ATTN_MAX_TOKENS" \
  --img_size "$IMG_SIZE" \
  --fold "$FOLD" \
  --shot "$SHOT" \
  --sam2_ckpt "$SAM2_CKPT" \
  --amp "$AMP" \
  --amp_dtype "$AMP_DTYPE"

echo "[7/7] test"
python test_pascal.py \
  --datapath "$DPATH" \
  --benchmark pascal \
  --load "logs/${LOGPATH}.log/best_model.pt" \
  --bsz 1 \
  --nworker 0 \
  --reduce_dim "$REDUCE_DIM" \
  --attn_max_tokens "$ATTN_MAX_TOKENS" \
  --img_size "$IMG_SIZE" \
  --fold "$FOLD" \
  --shot "$SHOT" \
  --sam2_ckpt "$SAM2_CKPT" \
  --visualize "$VISUALIZE" \
  --amp "$AMP" \
  --amp_dtype "$AMP_DTYPE"

echo "done: logs/${LOGPATH}.log"
