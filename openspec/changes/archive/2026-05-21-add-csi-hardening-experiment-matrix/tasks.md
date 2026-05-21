## 1. 配置与契约归一化

- [x] 1.1 定义 `csi_hardening` 配置结构和默认值，覆盖 common phase、subcarrier phase slope、antenna calibration、antenna permutation、mode、seed 和 enabled。
- [x] 1.2 在配置加载或训练准备阶段支持 `data.dataset.csi_hardening` alias，并在 CSI encoder 未显式配置时复制到 teacher/student `encoders.csi.csi_hardening`。
- [x] 1.3 确保未配置 `csi_hardening`、warmup 或 `use_internal_gru: false` 时，现有 `configs/csi/no_kd.yaml` 与 degraded CSI 配置加载结果保持兼容。

## 2. CSI hardening 算子与 encoder 集成

- [x] 2.1 实现 normalized complex CSI 上的 common phase rotation，支持训练随机、eval off 和 eval fixed 行为。
- [x] 2.2 实现 subcarrier phase slope，按 subcarrier index 生成相位斜率并保持 `[B,T,Nsc,Nant]` shape。
- [x] 2.3 实现 fixed-by-seed antenna calibration complex gain，支持幅度范围和相位标准差配置。
- [x] 2.4 实现 fixed-by-seed antenna permutation，确保相同 seed 与 Nant 下 permutation 可复现。
- [x] 2.5 将 hardening 集成到 `PilotDualViewCSIEncoder.forward()` 的 RMS 归一化之后、`PilotCSIChannelEstimator` 之前。
- [x] 2.6 增加 hardening auxiliary diagnostics，至少记录 enabled 状态、input power 或 phase statistic，并保持 `return_aux` 兼容。

## 3. CSI encoder 架构消融能力

- [x] 3.1 为 `pilot_dual_view_csi` 增加 `use_internal_gru`，默认 `true`，设置为 `false` 时跳过内部 GRU 并输出 `[B,T,D]`。
- [x] 3.2 增加 `view_gate_warmup_epochs` 与 `view_gate_warmup_mode: mean`，warmup 期强制 mean fusion 并输出等价 gate diagnostics。
- [x] 3.3 增加 `delay_view_warmup_epochs` 与 `delay_view_warmup_mode: freq_only`，warmup 期只使用 frequency view。
- [x] 3.4 增加 `view_fusion: freq_only`，用于 C5 frequency-only 消融。
- [x] 3.5 扩展 `CSIViewTokenizer` 支持 `tokenizer.hidden_channels`、`tokenizer.dropout` 和 `tokenizer.use_second_conv`。
- [x] 3.6 在 trainer 每个 epoch 开始时递归调用模型 epoch setter，保证 CSI warmup 行为随 epoch 生效。

## 4. G2D 与 CSI+easy modality 支持

- [x] 4.1 调整 G2D teacher ensemble 构建逻辑，使其优先使用 `distillation.g2d.modalities` 或 student model modalities，而不是假设固定五模态集合。
- [x] 4.2 确保 `gps+csi` G2D teacher forward、teacher confidence、weak-to-strong ranking 和 diagnostics 均包含真实配置模态名。
- [x] 4.3 验证 SMP gradient masking 在 active modality 为 `csi` 时保留 CSI encoder、fusion/head 梯度并屏蔽 inactive encoder 梯度。
- [x] 4.4 为缺失 `gps` 或 `csi` teacher checkpoint 的情况提供包含模态名和解析来源的错误信息。

## 5. 实验配置矩阵

- [x] 5.1 新增 CSI-only 第一批配置：A0、A1、A2、B3、B4、B5、B6、C1、C2。
- [x] 5.2 新增 CSI-only 第二批组合配置：D1、D2、D3、D4，并确保 D 组不启用 destructive `csi_degradation`。
- [x] 5.3 新增多模态验证配置：E0 GPS-only、E1 GPS+clean CSI、E2 GPS+slow CSI、E3 GPS+slow CSI + CSI-prioritized warmup、E4 GPS+slow CSI + G2D-style。
- [x] 5.4 为配置矩阵提供 README 或注释，明确推荐运行顺序：第一批 A/B/C，第二批 D，第三批 E0-E3，第四批 E4。

## 6. Sweep 分析脚本

- [x] 6.1 新增 `scripts/analyze_csi_hardening_sweep.py`，支持 `--runs_root`、`--pattern`、`--clean_teacher_run` 和 `--out` 参数。
- [x] 6.2 实现从 `train_log.json`、`final_config.yaml` 或可用 metrics artifact 中提取验证准确率、ADBA 和 run metadata。
- [x] 6.3 实现 final last10、best、E50/E80/E90、ceiling gap、E90 ratio、`is_destructive` 和 `is_slow_high_ceiling` 计算。
- [x] 6.4 输出 `summary.csv`、`ranked_candidates.csv`、`learning_curves.png` 和 `ceiling_gap_vs_E90_ratio.png`。

## 7. 测试

- [x] 7.1 在 `tests/test_csi_modality.py` 增加 hardening 算子 shape、finite、确定性、默认关闭、RMS 后 estimator 前调用顺序和 auxiliary diagnostics 测试。
- [x] 7.2 在 `tests/test_csi_modality.py` 增加 `use_internal_gru: false`、view gate warmup、delay view warmup、`freq_only` 和 tokenizer 配置测试。
- [x] 7.3 在 `tests/test_student_configs.py` 增加 CSI hardening matrix 和 GPS+CSI G2D-style 配置加载测试。
- [x] 7.4 在 `tests/test_g2d_smp.py` 或 `tests/test_g2d_distiller.py` 增加包含 `csi` 的 G2D teacher confidence、ranking 和 SMP gradient mask 测试。
- [x] 7.5 新增或扩展分析脚本测试，使用临时 run logs 验证 E50/E80/E90、destructive 判定、slow-high-ceiling 判定和 CSV 输出。

## 8. 验证

- [x] 8.1 运行 `conda run -n kd_mm_beam pytest tests/test_csi_modality.py -q`。
- [x] 8.2 运行 `conda run -n kd_mm_beam pytest tests/test_student_configs.py tests/test_g2d_loss.py tests/test_g2d_distiller.py tests/test_g2d_smp.py -q`。
- [x] 8.3 运行 `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py -q`，确认训练配置与 RMS 注入路径未回归。
- [x] 8.4 运行 `conda run -n kd_mm_beam python scripts/analyze_csi_hardening_sweep.py --help`，确认 CLI 可用。
- [x] 8.5 运行 `openspec status --change add-csi-hardening-experiment-matrix`，确认 proposal、design、specs 和 tasks 均为 done/apply-ready。
