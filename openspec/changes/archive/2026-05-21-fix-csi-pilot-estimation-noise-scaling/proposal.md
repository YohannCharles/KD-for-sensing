## Why

`debug-csi-hardening-experiment-matrix` 的结果表明 CSI 生成配置、A0 clone、pilot disabled、C1/C2 单变量路径已经健康；旧 full sweep 中非 A0 变体低到随机水平的主因不是 hardening 设计失败，而是 `mode: physical` 的固定 `noise_var` 在 CSI RMS 归一化后成为极强破坏噪声。

如果继续按旧结果解释，会把“噪声标定错误”误判成“CSI hard-to-learn 构造失败”。现在需要修正 pilot estimation 噪声量级语义，废弃/隔离旧矩阵结果，并用 debug 已验证的配置基线重跑 CSI hardening 矩阵。

## What Changes

- 将需要“mild pilot estimation”的 CSI 配置从绝对物理噪声方差改为按归一化 CSI 信号功率标定的 estimation-SNR 模式，优先使用 `mode: est_snr` 或 `mode: estimation_snr`。
- 明确 `mode: physical` 的适用前提：`noise_var` 必须与 estimator 接收张量的尺度一致；在 encoder 已执行训练集 RMS 归一化后，不得把未归一化物理噪声方差直接作为 mild 噪声。
- 更新 CSI hardening matrix 的 A/B/C/D 配置：除明确测试 pilot estimation 的 A1 外，其它单变量 hardening/encoder 变体必须显式设置 `csi_estimation.mode: none`。
- 为 A1 增加 pilot 噪声量级诊断 gate：`noise_power/signal_power` 必须落在配置 SNR 对应范围内，明显大于 mild 区间时 run 应标记为 invalid。
- 将旧 `outputs/csi_hardening_matrix_20260520_164406` 一类结果在分析输出中标记为 `invalid_due_to_pilot_noise_scale` 或等价状态，避免被候选排序误用。
- 重新生成并重跑 CSI-only A/B/C/D sweep，要求 A0、A0 clone、C1、C2 先通过 debug parity 后才输出 hardening 结论。
- 更新分析脚本，使 A0 parity 未通过、pilot noise ratio 失真或关键单变量 run 掉到随机水平时，输出“待排查/无效”，而不是输出 hardening 设计失败。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `csi-modality-model`: 修正 pilot estimation 噪声标定契约，要求 mild pilot estimation 使用信号功率相对 SNR 或明确校准后的归一化物理方差，并记录噪声/信号比诊断。
- `experiment-workflow`: 增加 CSI hardening sweep 的有效性 gate 和无效旧结果隔离要求，确保 debug parity 通过后才解释候选结论。

## Impact

- 受影响代码：
  - `src/kd_sensing/models/csi.py`
  - `src/kd_sensing/engine/debug_diagnostics.py`
  - CSI hardening sweep 生成/分析脚本
  - 训练 run metadata、`train_log.json` 或分析 summary 写入路径
- 受影响配置：
  - `configs/csi/hardening_matrix/*.yaml`
  - `configs/csi/hardening_matrix/debug/*.yaml`
  - 可能包括 fusion CSI hardening validation 配置，如果它们复用 slow CSI 变体
- 受影响实验产物：
  - 旧 full sweep 结果必须从候选结论中隔离，除非重新按新配置 rerun。
- 受影响测试：
  - `tests/test_csi_modality.py`
  - `tests/test_student_configs.py`
  - `tests/test_training_io_workflow.py`
  - CSI hardening analysis 脚本相关测试
- 不新增外部依赖；所有 Python 验证命令使用 `conda run -n kd_mm_beam ...`。
