## 1. Debug 矩阵与配置基准

- [x] 1.1 定位原始 `A0_clean_full_teacher` 的权威配置来源，优先使用实际训练产出的 resolved/final config 作为 `A0_original` 基准。
- [x] 1.2 增加 `A0_clone_generated` 配置生成入口，确保通过当前 sweep/config 生成器生成且关闭 `csi_hardening`、`csi_degradation`、pilot noise、view/delay warmup，并设置 `use_internal_gru=true`、`view_fusion=symmetric_gate`。
- [x] 1.3 增加 `A0_clone_generated + pilot disabled through new path` debug 配置，用于验证 pilot estimator disabled 解析路径与 A0 clone 等价。
- [x] 1.4 增加 `C1_view_gate_warmup_only` 和 `C2_no_internal_gru_only` debug 配置，保持其它数据、优化器、scheduler、loss、representation core 和 beam head 字段与 A0 clone 一致。
- [x] 1.5 为 5 个 debug run 配置 10 到 20 epoch 的短训练参数和独立输出目录，避免覆盖已有 full sweep 结果。

## 2. Resolved Config 与启动摘要

- [x] 2.1 在训练启动后保存完整 `resolved_config.yaml`，确保内容包含 defaults、alias 归一化和命令行覆盖后的最终值。
- [x] 2.2 实现 `A0_original` 与 `A0_clone_generated` 的 config diff artifact，允许 run name、output dir、timestamp 和 seed 等身份字段不同。
- [x] 2.3 在 diff 中显式检查 optimizer、scheduler、loss、dataset split、normalization、train RMS path、`seq_len`、`num_pred`、`num_classes`、CSI encoder、representation core 和 beam head。
- [x] 2.4 在 run 启动日志中打印 modalities、dataset path、split paths、batch size、optimizer/lr/scheduler、max epochs、model type、CSI encoder type、`d_model`、`delay_taps`、`view_fusion`、`use_internal_gru`、pilot/hardening/degradation 开关。
- [x] 2.5 在 run 启动日志中打印 total params、trainable params，以及 csi_encoder、representation_core、beam_head、fusion 等模块的 trainable params。

## 3. CSI Forward 数据流诊断

- [x] 3.1 为 `pilot_dual_view_csi` 增加显式 debug flag，默认关闭，只在首个 train batch 和首个 val batch 记录详细统计。
- [x] 3.2 在 hardening 前、hardening 后和 pilot 后记录 shape、dtype、abs_mean、abs_std、abs_max、real_mean、imag_mean、nan_count 和 zero_ratio。
- [x] 3.3 在 freq view 和 delay view 记录 shape、mean、std 和 nan_count。
- [x] 3.4 记录 freq_feat、delay_feat、fused_feat、gru_out 和 final CSI feature 的 norm，并区分 train/val batch 来源。
- [x] 3.5 将 CSI 首 batch 诊断写入 run log、metadata artifact 或 TensorBoard text/scalar stream，并保持正常训练日志兼容。

## 4. Pilot、Hardening 与 View 不变量

- [x] 4.1 检查 `pilot_estimator.enabled=false` 解析路径，确保 `h_hat` 与输入归一化 CSI 完全一致，并记录 `max_abs(h_hat - h)`。
- [x] 4.2 在 A1 类 mild pilot estimation 路径中记录 sampled SNR、`sigma_e2_mean`、`h_power_mean`、`noise_power_mean`、`h_hat_power_mean` 和 `noise_power/signal_power`。
- [x] 4.3 检查 hardening 输出 shape 必须保持 `[B,T,Nsc,Nant]`、`nan_count=0`，并在 abs_mean 或 abs_std 无解释漂移超过 20% 时标记 warning。
- [x] 4.4 确保 fixed antenna permutation 和 antenna calibration 在同一 run 内 fixed by seed，不会每个 batch 重新采样。
- [x] 4.5 检查 `view_gate_warmup` 和 `use_internal_gru=false` 路径，确保非零 CSI 输入下 fused feature 与 final CSI feature norm 不为 0。

## 5. 训练健康指标

- [x] 5.1 在 debug 模式下按模块保存 epoch 初始参数快照或轻量 checksum，用于计算 param delta。
- [x] 5.2 每个 epoch 记录 `grad_norm_csi_encoder`、`grad_norm_representation_core`、`grad_norm_beam_head` 和存在时的 fusion grad norm。
- [x] 5.3 每个 epoch 记录 `param_delta_csi_encoder`、`param_delta_representation_core`、`param_delta_beam_head` 和存在时的 fusion param delta。
- [x] 5.4 当关键模块 grad norm 或 param delta 持续为 0 时，在日志中标记可能冻结、optimizer 参数组遗漏、梯度屏蔽或 loss 未连接。

## 6. Debug 判定与运行入口

- [x] 6.1 更新 debug README、脚本或命令说明，明确先跑 5 个 debug run，不继续解读完整 sweep。
- [x] 6.2 实现或记录判定逻辑：A0 original 高但 A0 clone 低时先修配置继承；A0 clone 高但 C1 低时修 view gate warmup；A0 clone 高但 C2 低时修 no-internal-GRU 路径。
- [x] 6.3 实现或记录判定逻辑：B3/B4 才低时修 hardening；A1 低但 B/C 高时修 pilot estimator 噪声计算。
- [x] 6.4 确保分析输出在 A0 clone parity 未通过前，将 full sweep 结果标记为待排查而不是 hardening 设计失败。

## 7. 测试

- [x] 7.1 使用 `conda run -n kd_mm_beam pytest tests/test_student_configs.py -q` 验证 5 个 debug 配置可加载且关键字段继承 A0 clone。
- [x] 7.2 使用 `conda run -n kd_mm_beam pytest tests/test_csi_modality.py -q` 验证 pilot disabled identity、hardening shape/finite/stat diagnostics、view warmup feature norm 和 no-internal-GRU 输出 shape。
- [x] 7.3 使用 `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py -q` 验证 `resolved_config.yaml`、config diff artifact、启动摘要和 debug metrics 日志写入。
- [x] 7.4 新增或扩展测试，验证 trainable params by module、grad norm 和 param delta 在 toy training step 中非零且可持久化。

## 8. 验证

- [x] 8.1 使用 `conda run -n kd_mm_beam python -m pytest tests/test_csi_modality.py tests/test_student_configs.py tests/test_training_io_workflow.py -q` 运行相关自动化测试。
- [x] 8.2 使用 `conda run -n kd_mm_beam` 启动 5 个 debug run 的 dry-run 或短跑命令，确认输出目录中存在 `resolved_config.yaml`、config diff、CSI 首 batch 诊断和 epoch 训练健康指标。
- [x] 8.3 人工检查 `A0_original` 与 `A0_clone_generated` 的 diff 和前 10 到 20 epoch 曲线；若 clone 未接近 original，暂停后续 hardening 结论并修配置。
- [x] 8.4 运行 `openspec status --change debug-csi-hardening-experiment-matrix`，确认 proposal、design、specs 和 tasks 均为 done/apply-ready。
