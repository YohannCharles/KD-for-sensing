mkdir -p logs

COMMON="-o data.dataset.portion=1.0 \
-o data.dataset.sample_cache.enabled=true \
-o data.dataset.sample_cache.path=dataset/DeepSense6G/scenario31/sample_lmdb_cache/u_mask_beam_jepa_seq2_pred1_{split}.lmdb \
-o data.dataset.image_use_cache=true \
-o data.dataset.lidar_use_cache=true \
-o data.dataloader.pin_memory=true \
-o training.transfer.non_blocking=true \
-o training.epochs=50 \
-o training.validation.interval_epochs=5 \
-o data.dataloader.train_batch_size=32 \
-o data.dataloader.test_batch_size=32 \
-o data.dataloader.num_workers=4 \
-o data.dataloader.persistent_workers=true \
-o data.dataloader.prefetch_factor=1"

CUDA_VISIBLE_DEVICES=0 nohup conda run -n kd_mm_beam kd-sensing-train -c configs/fusion/u_mask_beam_jepa_smoke.yaml $COMMON -o output.run_name=u_mask_beam_jepa_full_lmdb_v5 > logs/u_mask_beam_jepa_full_lmdb_v5.gpu0.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup conda run -n kd_mm_beam kd-sensing-train -c configs/fusion/u_mask_beam_jepa_no_jepa.yaml $COMMON -o output.run_name=u_mask_beam_jepa_no_jepa_lmdb_v5 > logs/u_mask_beam_jepa_no_jepa_lmdb_v5.gpu1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup conda run -n kd_mm_beam kd-sensing-train -c configs/fusion/u_mask_beam_jepa_no_uncertainty.yaml $COMMON -o output.run_name=u_mask_beam_jepa_no_uncertainty_lmdb_v5 > logs/u_mask_beam_jepa_no_uncertainty_lmdb_v5.gpu2.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup conda run -n kd_mm_beam kd-sensing-train -c configs/fusion/u_mask_beam_jepa_concat_mlp.yaml $COMMON -o output.run_name=u_mask_beam_jepa_concat_mlp_lmdb_v5 > logs/u_mask_beam_jepa_concat_mlp_lmdb_v5.gpu3.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup conda run -n kd_mm_beam kd-sensing-train -c configs/fusion/u_mask_beam_jepa_weighted_sum.yaml $COMMON -o output.run_name=u_mask_beam_jepa_weighted_sum_lmdb_v5 > logs/u_mask_beam_jepa_weighted_sum_lmdb_v5.gpu0.log 2>&1 &