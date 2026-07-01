#!/usr/bin/env bash

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
    > "logs/m2beam_single_modal_scene31/${name}.log" 2>&1
  code=$?
  echo "===== DONE ${name} on GPU${gpu} exit=${code} $(date) ====="
}

mkdir -p logs/m2beam_single_modal_scene31

# ponytail: fixed four-GPU mapping for the four current single-modal runs.
run_one 0 gps configs/fusion/experiments/m2beam_single_modal_scene31/gps.yaml &
run_one 1 radar configs/fusion/experiments/m2beam_single_modal_scene31/radar.yaml &
run_one 2 lidar configs/fusion/experiments/m2beam_single_modal_scene31/lidar.yaml &
run_one 3 image configs/fusion/experiments/m2beam_single_modal_scene31/image.yaml &
wait
