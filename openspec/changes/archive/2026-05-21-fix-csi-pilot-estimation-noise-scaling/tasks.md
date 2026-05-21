## 1. 配置矩阵修正

- [x] 1.1 更新 `configs/csi/hardening_matrix/A1_mild_pilot_estimation.yaml`，将 `csi_estimation.mode` 从 `physical` 改为 `est_snr` 或 `estimation_snr`，并配置固定 `snr_db` 或 `train_snr_min_db/train_snr_max_db`
- [x] 1.2 更新 B 组 hardening-only 配置，显式设置 `csi_estimation.mode: none` 且不继承 A1 pilot noise
- [x] 1.3 更新 C 组 encoder-only 配置，显式设置 `csi_estimation.mode: none`，并保留各自 encoder 单变量
- [x] 1.4 更新 D 组 combined 配置，显式设置 `csi_estimation.mode: none`，并只组合声明的 hardening 与 encoder 变量
- [x] 1.5 检查 A2 destructive negative control 的配置命名和 metadata，确保它被分析脚本识别为 destructive 而不是 mild pilot
- [x] 1.6 同步更新配置生成器、README 或运行脚本中关于 A1/B/C/D 的说明

## 2. Pilot Estimator 诊断与校验

- [x] 2.1 确认 `PilotCSIChannelEstimator` 在 physical、estimation-SNR 和 disabled/no-noise 路径均输出 `sigma_e2`、`h_power_mean`、`noise_power_mean`、`h_hat_power_mean`、`noise_power_signal_ratio` 和 `pilot_identity_max_abs`
- [x] 2.2 为 estimation-SNR 训练采样路径记录 sampled `snr_db`，并保证 batch 维采样值能进入 diagnostics
- [x] 2.3 增加 mild pilot noise validity helper，根据配置 SNR 或允许阈值判断 `noise_power_signal_ratio` 是否失真
- [x] 2.4 在 debug diagnostics 或 run metadata 中写入 pilot noise validity 结果和 invalid reason
- [x] 2.5 保持 `mode: physical` 的原有方差公式不变，避免破坏现有物理模式契约

## 3. Sweep 分析 Gate

- [x] 3.1 更新 CSI hardening sweep 分析脚本，先执行 A0 parity、pilot noise scale、C1/C2 health 和 diagnostics availability gate
- [x] 3.2 在 `summary.csv` 中新增或填充 `pilot_noise_scale_valid`、`invalid_reason`、`full_sweep_status`、`a0_parity_status` 和 `debug_decision` 字段
- [x] 3.3 在 `ranked_candidates.csv` 中排除 invalid 或 pending-debug run
- [x] 3.4 对缺少必需 diagnostics 的旧 full sweep 目录输出 `invalid_due_to_missing_debug_diagnostics` 或等价状态
- [x] 3.5 对 noise ratio 明显失真的 mild pilot run 输出 `invalid_due_to_pilot_noise_scale`
- [x] 3.6 确保 gate 未通过时不输出 hardening 设计失败结论

## 4. 旧结果隔离与重跑入口

- [x] 4.1 标记旧 `outputs/csi_hardening_matrix_20260520_164406` 类结果为 invalid/pending，不删除原始产物
- [x] 4.2 提供修复后的短 debug gate 运行命令，命令必须使用 `conda run -n kd_mm_beam`
- [x] 4.3 提供修复后的 CSI-only A/B/C/D full sweep 运行命令，命令必须使用 `conda run -n kd_mm_beam`
- [x] 4.4 确保新 sweep 输出目录与旧 invalid sweep 输出目录隔离
- [x] 4.5 在 analysis artifact 中记录当前结果是否来自修复后的 pilot scaling 配置

## 5. 单元测试

- [x] 5.1 扩展 `tests/test_student_configs.py`，验证 A1 使用 estimation-SNR，B/C/D 显式关闭 pilot noise
- [x] 5.2 扩展 `tests/test_csi_modality.py`，验证 estimation-SNR 的 `noise_power_signal_ratio` 与配置 SNR 同量级
- [x] 5.3 扩展 `tests/test_csi_modality.py`，验证 physical 模式仍保持 `noise_var / (pilot_power * pilot_len)` 方差公式
- [x] 5.4 扩展 debug/analysis 相关测试，验证旧缺失 diagnostics 的 sweep 被标记 invalid
- [x] 5.5 扩展 analysis 测试，验证 invalid run 不进入 ranked candidates

## 6. 自动化验证

- [x] 6.1 运行 `conda run -n kd_mm_beam pytest tests/test_student_configs.py -q`
- [x] 6.2 运行 `conda run -n kd_mm_beam pytest tests/test_csi_modality.py -q`
- [x] 6.3 运行 `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py -q`
- [x] 6.4 运行 analysis 脚本相关测试或最小 fixtures，命令必须使用 `conda run -n kd_mm_beam`
- [x] 6.5 运行 `openspec status --change fix-csi-pilot-estimation-noise-scaling`

## 7. 实验验证

- [ ] 7.1 使用修复后的配置运行 5 个 debug gate 短跑，确认 A0 original、A0 clone、pilot disabled、C1 only、C2 only 全部健康
- [ ] 7.2 使用修复后的 A1 estimation-SNR 配置运行短跑，确认 `noise_power_signal_ratio` 落在目标 SNR 区间
- [ ] 7.3 运行修复后的 CSI-only A/B/C/D sweep，输出新的 `summary.csv` 和 `ranked_candidates.csv`
- [ ] 7.4 人工检查新 sweep 的候选曲线，确认是否存在 high-ceiling slow-learning CSI variant
- [ ] 7.5 只有新 CSI-only 候选成立后，再恢复 GPS+CSI 或 G2D-style 多模态验证
