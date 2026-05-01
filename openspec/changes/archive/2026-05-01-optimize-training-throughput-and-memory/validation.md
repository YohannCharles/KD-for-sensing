## 验证记录

- `conda run -n kd_mm_beam python -m compileall -q src scripts`：通过。
- `conda run -n kd_mm_beam pytest -q tests/test_training_io_workflow.py tests/test_lidar_modality.py tests/test_gps_modality.py`：`108 passed`。
- `conda run -n kd_mm_beam pytest -q -p no:cacheprovider`：`197 passed`。
- `conda run -n kd_mm_beam python scripts/profile_training_io.py --config configs/image/student_no_kd.yaml --samples 1 --warmup 0 --device cpu --output outputs/profile/smoke_profile.json -o data.dataset.type=synthetic -o data.dataset.length=1 -o data.dataloader.train_batch_size=1 -o data.dataloader.test_batch_size=1 -o data.dataloader.num_workers=0 -o output.tensorboard.enabled=false`：通过；临时输出已清理。
- `conda run -n kd_mm_beam python scripts/train.py --config configs/image/student_no_kd.yaml --dry-run -o experiment.device=cpu -o output.dir=outputs/smoke_throughput -o output.tensorboard.enabled=false -o output.progress.enabled=false -o training.amp.enabled=false`：通过；临时输出和临时 registry checkpoint 已清理。
- `conda run -n kd_mm_beam python scripts/profile_training_io.py --config configs/fusion/image_radar_student_no_kd.yaml --samples 1 --warmup 0 --device cpu --output outputs/profile/fusion_smoke_profile.json -o data.dataset.type=synthetic -o data.dataset.length=1 -o data.dataloader.train_batch_size=1 -o data.dataloader.test_batch_size=1 -o data.dataloader.num_workers=0 -o output.tensorboard.enabled=false`：通过；临时输出已清理。
- `conda run -n kd_mm_beam python scripts/train.py --config configs/fusion/image_radar_student_no_kd.yaml --dry-run -o experiment.device=cpu -o output.dir=outputs/smoke_throughput -o output.tensorboard.enabled=false -o output.progress.enabled=false -o training.amp.enabled=false`：通过；临时输出和临时 registry checkpoint 已清理。

`openspec` CLI 当前依赖的 `node` 在本机执行 `openspec --version` / `openspec status` 时会卡住，之前观察到进程进入不可中断 I/O 状态。为避免继续产生卡住进程，本 change 采用文件级校验：确认 proposal、design、tasks、spec diff 和本验证记录均存在且非空。
