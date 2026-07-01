#!/usr/bin/env bash

mkdir -p logs/rbma_strong_encoder_4gpu

run_one() {
  gpu="$1"
  name="$2"
  cfg="$3"

  echo "===== START ${name} on GPU${gpu} $(date) ====="
  CUDA_VISIBLE_DEVICES="${gpu}" \
  MALLOC_ARENA_MAX=2 \
  OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=4 \
  conda run -n kd_mm_beam kd-sensing-train \
    --config "${cfg}" \
    --num-workers 0 \
    data.dataloader.train_batch_size=8 \
    data.dataloader.test_batch_size=8 \
    > "logs/rbma_strong_encoder_4gpu/${name}.log" 2>&1
  code=$?
  echo "===== DONE ${name} on GPU${gpu} exit=${code} $(date) ====="
  return "${code}"
}

base="configs/fusion/experiments/rbma_missing_workflow_strong_encoders"

run_one 0 V0_strong_amber "${base}/amber_style_mask_baseline_fullrun.yaml" &
run_one 1 V1_strong_mask "${base}/weighted_sum_mask.yaml" &
run_one 2 V2_strong_reliability "${base}/weighted_sum_reliability.yaml" &
run_one 3 V3_strong_reliability_proto "${base}/weighted_sum_reliability_beam_proto.yaml" &
wait

run_one 0 V4_strong_reliability_proto_kd "${base}/weighted_sum_reliability_beam_proto_kd.yaml" &
run_one 1 V5_strong_rbma_proto_kd "${base}/no_jepa_rbma_proto_kd_fullrun.yaml" &
wait

conda run -n kd_mm_beam python scripts/summarize_missing_runs.py --root outputs/scene31
