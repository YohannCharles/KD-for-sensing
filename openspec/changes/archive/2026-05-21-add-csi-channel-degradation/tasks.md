## 1. 配置与退化 profile

- [x] 1.1 在 `src/kd_sensing/data/transform_ops/csi.py` 或相邻模块中定义 `CSIDegradationConfig`、profile resolver 和有效参数导出结构，默认 `enabled: false`。
- [x] 1.2 实现 `clean`、`medium`、`hard` profile，并支持 YAML 覆盖 SNR、path dropout、dominant attenuation、delay/angle noise、antenna phase error、temporal shift 和 seed。
- [x] 1.3 在 dataset/data factory 配置传递路径中接入 `data.dataset.csi_degradation`，保持未配置时现有 clean CSI 行为不变。

## 2. CSI 退化算子实现

- [x] 2.1 实现 complex gain AWGN，按 clean CSI 或 path gain 功率与 `snr_db` 计算复噪声方差。
- [x] 2.2 实现弱路径优先 path dropout，并保证未配置 dominant path removal 时不直接删除全部最强路径。
- [x] 2.3 实现 dominant path attenuation、delay noise、delay quantization、AoA/AoD angle noise 和 antenna phase calibration error。
- [x] 2.4 在 path-level payload 可用时先执行 path-level 退化再派生等效 CSI；在 tensor-only payload 下只执行可用 tensor-level 退化并记录 skipped operators。
- [x] 2.5 保证所有退化后 CSI 输出仍为 finite `float32` real/imag 张量，形状满足 `[Nsc, Nant, 2]` 或 `[T, Nsc, Nant, 2]`。

## 3. Dataset 集成与 temporal shift

- [x] 3.1 修改 `load_csi_sequence()`/`read_csi_tensor()` 调用链，使其可接收退化配置、样本稳定 key 和 diagnostics 容器。
- [x] 3.2 在 dataset 内分离 clean CSI cache 与 degraded CSI cache，确保 `_prepare_csi_rms_normalizer()` 始终使用退化前 clean CSI。
- [x] 3.3 实现历史窗口内 CSI temporal shift，默认边界 clamp，不读取 future CSI 或 future beam 对应帧。
- [x] 3.4 使用 base seed、split、dataset index 和历史 CSI 路径摘要派生 NumPy RNG，确保多 worker 和重复读取结果一致。
- [x] 3.5 在 return metadata 或 run metadata 中记录 degradation profile、resolved parameters、seed、sample shift、fill mode 和 skipped operators。

## 4. 配置样例与实验入口

- [x] 4.1 新增 CSI-only medium degraded no-KD 配置，并显式启用 `data.dataset.csi_degradation.profile: medium`。
- [x] 4.2 新增至少一个包含 degraded CSI 的 fusion 示例配置，确保其它模态输入和 future beam label 不被 CSI degradation 修改。
- [x] 4.3 检查现有 `configs/csi/no_kd.yaml` 和 `configs/fusion/mmwave_csi_no_kd.yaml` 在未配置 degradation 时仍加载为 clean CSI。

## 5. 测试

- [x] 5.1 在 `tests/test_csi_modality.py` 增加 profile resolver、AWGN 方差、path dropout、dominant attenuation、angle/delay noise 和 antenna phase error 的小夹具测试。
- [x] 5.2 增加 deterministic 测试，验证相同 seed、split、样本 index 和路径集合重复读取 degraded CSI 数值一致。
- [x] 5.3 增加 RMS 测试，验证启用 degradation 时训练集 RMS 仍基于 clean CSI，test split 复用 train-fitted normalizer。
- [x] 5.4 增加 temporal shift 测试，验证 shift 不读取未来 CSI，边界 clamp 行为和 metadata 记录符合预期。
- [x] 5.5 增加配置加载测试，覆盖 CSI-only degraded 配置和 fusion degraded CSI 配置。

## 6. 验证与 OpenSpec

- [x] 6.1 运行 `conda run -n kd_mm_beam pytest tests/test_csi_modality.py -q`。
- [x] 6.2 运行 `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py tests/test_student_configs.py -q`。
- [x] 6.3 运行 `conda run -n kd_mm_beam pytest tests/test_mmw_town10_preparation.py -q`，确认 MMW sequence/CSI 路径契约未回归。
- [x] 6.4 运行 `openspec status --change add-csi-channel-degradation`，确认 proposal、design、specs 和 tasks 均为 done/apply-ready。
