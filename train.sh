#!/bin/bash
# ComCLIP post-training scripts.
#
# Usage:
#   bash train.sh <experiment>
#   SEED=42 bash train.sh comclip_vitb16_cc3m      # another seed
#   NUM_GPUS=4 BATCH_SIZE=256 bash train.sh ...    # keep the global batch size at 1,024
#
# Run `bash train.sh` without arguments to list all experiments.
set -e

# ---------------------------------------------------------------------------
# Paths: edit these to match your environment.
# ---------------------------------------------------------------------------
CC3M_SHARDS="/path/to/data/cc3m/cc3m-train-{0000..0575}.tar"
CC12M_SHARDS="/path/to/data/cc12m-wds/cc12m-train-{0000..2175}.tar"
COCO_IMG_DIR="/path/to/data/coco/train2017"
COCO_ANN_FILE="/path/to/data/coco/annotations/captions_train2017.json"
IMAGENET_ROOT="/path/to/data/imagenet-1k"
OUTPUT_DIR="/path/to/output"

# ---------------------------------------------------------------------------
# Defaults: 1 epoch, 8 GPUs x 128 = global batch size 1,024, tau = 0.01 (fixed).
# ---------------------------------------------------------------------------
NUM_GPUS=${NUM_GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-128}   # per GPU
SEED=${SEED:-0}

# run <experiment_name> <backbone> <dataset> [extra args...]
# Extra args are appended last and override the defaults below.
run() {
    local name=$1 backbone=$2 dataset=$3
    shift 3
    local data_args
    case $dataset in
        cc3m)  data_args=(--dataset cc3m  --wds_path "$CC3M_SHARDS"  --wds_train_length 3000000) ;;
        cc12m) data_args=(--dataset cc12m --wds_path "$CC12M_SHARDS" --wds_train_length 10968539) ;;
        coco)  data_args=(--dataset coco --coco_img_dir "$COCO_IMG_DIR" --coco_annotation_file "$COCO_ANN_FILE") ;;
        *) echo "Unknown dataset: $dataset"; exit 1 ;;
    esac

    torchrun --nproc_per_node=$NUM_GPUS -m train.align_training_clip \
        --clip_model_name "$backbone" --pretrained openai \
        "${data_args[@]}" \
        --imagenet_root "$IMAGENET_ROOT" --template std --output_normalize False \
        --total_epochs 1 --warmup 410 \
        --batch_size $BATCH_SIZE --opt adamw --lr 1e-6 --wd 0.1 \
        --scale 0.01 --scale_learn False \
        --seed $SEED --wandb False \
        --output_dir "$OUTPUT_DIR" --experiment_name "${name}_seed${SEED}" \
        --log_freq 1 --eval_freq 10 \
        "$@"
}

# comclip <experiment_name> <backbone> <dataset> [extra args...]
# Full ComCLIP objective: L_clip + L_mse + L_rkd (all weights 1.0).
comclip() {
    local name=$1 backbone=$2 dataset=$3
    shift 3
    run "$name" "$backbone" "$dataset" \
        --clip_weight 1.0 --fdimg_weight 1.0 --dinorkd_weight 1.0 "$@"
}

# clip_only <experiment_name> <dataset> [extra args...]
# Contrastive loss only (L_clip) on ViT-B/16, used for Finding 1.
clip_only() {
    local name=$1 dataset=$2
    shift 2
    run "$name" ViT-B-16 "$dataset" --oriclip "$@"
}

EXPERIMENTS="
  Main results (Table 1):
    comclip_vitb16_cc3m        comclip_vitl14_cc3m
    comclip_vitb16_coco        comclip_vitl14_coco
    comclip_vitb16_cc12m       comclip_vitl14_cc12m
  SigLIP backbone (Appendix):
    comclip_siglip_cc3m
  Finding 1, contrastive loss only on ViT-B/16:
    clip_only_tau              tau in {0.07, 0.03, 0.02, 0.01, 0.007, 0.005}
    clip_only_batch_size       per-GPU batch size in {16, 32, 64, 128, 256} x 8 GPUs
    clip_only_learnable_tau    learnable vs. fixed tau on COCO Caption and CC3M
  Ablations on ViT-B/16 + CC3M:
    ablation_loss              all single and pairwise combinations of the three losses
    ablation_loss_weights      (lambda_clip, lambda_mse, lambda_rkd) variations
    ablation_teacher           no teacher / MAE-Large / DINOv2-Base teachers
    ablation_before_proj       L_rkd on the pre-projection embedding
    ablation_2ep               ComCLIP trained for 2 epochs
"

case "$1" in
    # ---------------- Main results ----------------
    comclip_vitb16_cc3m)  comclip comclip_vitb16_cc3m  ViT-B-16 cc3m ;;
    comclip_vitl14_cc3m)  comclip comclip_vitl14_cc3m  ViT-L-14 cc3m ;;
    comclip_vitb16_coco)  comclip comclip_vitb16_coco  ViT-B-16 coco ;;
    comclip_vitl14_coco)  comclip comclip_vitl14_coco  ViT-L-14 coco ;;
    comclip_vitb16_cc12m) comclip comclip_vitb16_cc12m ViT-B-16 cc12m ;;
    comclip_vitl14_cc12m) comclip comclip_vitl14_cc12m ViT-L-14 cc12m ;;

    # ---------------- SigLIP ViT-SO400M/14 ----------------
    comclip_siglip_cc3m)  comclip comclip_siglip_cc3m  SigLip cc3m ;;

    # ---------------- Finding 1: temperature ----------------
    clip_only_tau)
        for tau in 0.07 0.03 0.02 0.01 0.007 0.005; do
            clip_only "clip_only_vitb16_cc3m_tau${tau}" cc3m --scale $tau
        done ;;
    clip_only_batch_size)
        for bs in 16 32 64 128 256; do
            clip_only "clip_only_vitb16_cc3m_bs${bs}x${NUM_GPUS}" cc3m --batch_size $bs
        done ;;
    clip_only_learnable_tau)
        for dataset in coco cc3m; do
            for learn in True False; do
                clip_only "clip_only_vitb16_${dataset}_scale_learn${learn}" $dataset --scale_learn $learn
            done
        done ;;

    # ---------------- Ablations ----------------
    ablation_loss)
        # "lambda_clip lambda_mse lambda_rkd"; the full objective is comclip_vitb16_cc3m
        for w in "1.0 0.0 0.0" "0.0 1.0 0.0" "0.0 0.0 1.0" "0.0 1.0 1.0" "1.0 1.0 0.0" "1.0 0.0 1.0"; do
            set -- $w
            run "loss_clip${1}_mse${2}_rkd${3}_vitb16_cc3m" ViT-B-16 cc3m \
                --clip_weight $1 --fdimg_weight $2 --dinorkd_weight $3
        done ;;
    ablation_loss_weights)
        for w in "1.0 1.0 0.5" "1.0 0.5 1.0" "1.0 0.5 0.5"; do
            set -- $w
            run "weights_clip${1}_mse${2}_rkd${3}_vitb16_cc3m" ViT-B-16 cc3m \
                --clip_weight $1 --fdimg_weight $2 --dinorkd_weight $3
        done ;;
    ablation_teacher)
        run     teacher_none_vitb16_cc3m          ViT-B-16 cc3m --clip_weight 1.0 --fdimg_weight 1.0 --dinorkd_weight 0.0
        comclip teacher_mae_large_vitb16_cc3m     ViT-B-16 cc3m --vision_model mae
        comclip teacher_dinov2_base_vitb16_cc3m   ViT-B-16 cc3m --model_name dinov2_vitb14_reg
        comclip teacher_dinov2_base_vitl14_cc3m   ViT-L-14 cc3m --model_name dinov2_vitb14_reg
        ;;
    ablation_before_proj)
        comclip comclip_before_proj_vitb16_cc3m ViT-B-16 cc3m --before_proj True ;;
    ablation_2ep)
        comclip comclip_2ep_vitb16_cc3m ViT-B-16 cc3m --total_epochs 2 ;;

    *)
        echo "Usage: bash train.sh <experiment>"
        echo "$EXPERIMENTS"
        exit 1 ;;
esac
