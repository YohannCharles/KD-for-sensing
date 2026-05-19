## 1. 模态契约与配置入口

- [x] 1.1 在 `src/kd_sensing/modalities.py` 中新增 `csi` 模态契约，包含 `use_csi`、`csi`、`csi_batch`、默认 encoder/model 字段和 CSI RMS artifact key
- [x] 1.2 更新模态标准化、dataset flag/default、batch input key 和 sample key 相关测试，验证 `csi` 固定排在 `mmwave` 之后
- [x] 1.3 更新 `src/kd_sensing/engine/modality_resolution.py`，支持 `experiment.task: csi` 和 fusion `modalities` 中的 `csi`
- [x] 1.4 更新默认配置或 canonical 配置生成逻辑，使 CSI-only 和包含 CSI 的 fusion 配置可被加载

## 2. CSI 数据加载与 RMS 统计

- [x] 2.1 新增 `src/kd_sensing/data/transform_ops/csi.py`，实现 `.npy/.npz` clean CSI 加载、complex 到 real/imag 转换、shape/finite 校验
- [x] 2.2 扩展 `SequenceSamples` 或等价样本路径解析，支持启用 CSI 时校验并保存 `csi1..csiN` 历史路径列
- [x] 2.3 扩展 DeepSense6G/MMW dataset 初始化与 `DeepSense6GModalityLoader`，启用 CSI 时返回 `sample["csi"]`
- [x] 2.4 实现 CSI 训练集全局 RMS normalizer，确保 train fit、test 复用，并在未启用 CSI 时零开销
- [x] 2.5 扩展 MMW Town10 准备或后处理路径，在启用 CSI 导出时从 `channel_path` 写出 `csi*` 列或输出可诊断失败原因

## 3. Batch 与训练路径

- [x] 3.1 在 `src/kd_sensing/engine/batch.py` 中新增 `prepare_csi_inputs`，按历史窗口截断并追加 `num_pred - 1` 个 zero padding 时隙
- [x] 3.2 更新 `prepare_fusion_inputs` 和 `forward_model`，将 `csi_batch` 传入 CSI-only 或 fusion 模型
- [x] 3.3 更新训练、验证、评估和诊断路径的输入映射测试，确认 CSI 标签仍来自 `target_beam[:, :num_pred]`

## 4. CSI Encoder 与注册表

- [x] 4.1 新增 `src/kd_sensing/models/csi.py`，实现 `PilotCSIChannelEstimator`，覆盖 physical 和 estimation SNR 两种噪声模式
- [x] 4.2 实现 frequency view、delay view、`CSIViewTokenizer`、`SymmetricViewFusion` 和 `PilotDualViewCSIEncoder`
- [x] 4.3 将 `PilotDualViewCSIEncoder` 注册为 `ENCODERS["pilot_dual_view_csi"]`，暴露 `output_dim` 并支持 `output_dim`/`d_model`/`feature_size`
- [x] 4.4 更新 `src/kd_sensing/models/modular.py` 默认 encoder 解析，使 `modalities: [csi]` 能构建 `pilot_dual_view_csi`
- [x] 4.5 更新 `src/kd_sensing/registries.py` 的默认组件导入，确保构建流程能发现 CSI encoder

## 5. 实验配置

- [x] 5.1 新增 `configs/csi/no_kd.yaml` 或等价 CSI-only baseline，使用 `modular_sequence` 和 `pilot_dual_view_csi`
- [x] 5.2 新增至少一个包含 CSI 的 fusion 示例配置，验证 CSI 可与 `gps`、`mmwave` 或其它模态共同进入 `modular_sequence`
- [x] 5.3 在配置中暴露 `csi_estimation`、`delay_taps`、`view_fusion`、`train_rms`、SNR 和 pilot 参数，并记录到运行 metadata

## 6. 测试与验证

- [x] 6.1 使用 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py` 验证模态契约和注册边界
- [x] 6.2 新增并使用 `conda run -n kd_mm_beam pytest tests/test_csi_modality.py` 验证 CSI loader、RMS、batch padding、pilot noise 方差和 encoder forward shape
- [x] 6.3 使用 `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py` 验证配置加载、dataset 构建和训练输入映射未回归
- [x] 6.4 使用 `conda run -n kd_mm_beam python scripts/train.py --config configs/csi/no_kd.yaml training.epochs=1 data.dataloader.num_workers=0 data.dataset.portion=0.02` 运行 CSI-only smoke training
