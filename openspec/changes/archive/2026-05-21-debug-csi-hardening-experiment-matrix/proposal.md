## Why

当前 `add-csi-hardening-experiment-matrix` 的非 A0 CSI 变体几乎全部停在 `accuracy_val≈0.14`、`val_adba≈0.50`，包括理论上不应改变 CSI 输入的 `C1_view_gate_warmup` 和 `C2_no_internal_gru`。这更像配置继承、CSI encoder 新路径、pilot/hardening 数据流或训练连接错误，而不是 CSI hardening 实验设计本身失败。

## What Changes

- 新增一个最小排查矩阵，先跑 `A0_original`、`A0_clone_generated`、`A0_clone_generated + pilot disabled through new path`、`C1_view_gate_warmup_only`、`C2_no_internal_gru_only`，暂停完整 sweep 结论解读。
- 为每个 run 保存 `resolved_config.yaml`，并在 run 启动时打印关键配置、模型结构、参数规模和 trainable params by module。
- 增加 A0 original 与 A0 generated clone 的配置 diff，除 run/output/seed 等允许字段外，关键字段不一致时必须先修复配置继承。
- 在 CSI forward 首个 train/val batch 增加可控 debug logging，覆盖 hardening 前后、pilot 后、freq/delay view、feature norm、GRU 输出和最终 CSI feature。
- 强化 pilot estimator 与 hardening 的不变量检查：`enabled=false` 时 `h_hat == h`，hardening 后 shape 保持 `[B,T,Nsc,Nant]` 且统计量不异常漂移。
- 增加每 epoch 的训练健康指标：CSI encoder、representation core、beam head 的 grad norm 和 param delta，用于发现冻结、loss 未连接或梯度被屏蔽。
- 不继续新增完整实验矩阵；本 change 的目标是确认并修复 A1/B/C/D 变体是否复现 A0 的 CSI 学习路径。

## Capabilities

### New Capabilities

- `csi-hardening-debug-validation`: 定义 CSI hardening 矩阵的最小诊断 run、配置一致性检查、CSI 数据流统计、pilot/hardening 不变量和训练健康判据。

### Modified Capabilities

- `csi-modality-model`: 补充 CSI encoder/pilot/hardening 路径的 debug diagnostics 与关键不变量要求。
- `experiment-workflow`: 要求实验 run 保存 resolved config、支持基准 clone diff，并在 debug 矩阵中先验证配置/训练路径再解释 sweep 结果。

## Impact

- 受影响代码：
  - `src/kd_sensing/models/csi.py`
  - `src/kd_sensing/engine/trainer.py`
  - `src/kd_sensing/engine/run_metadata.py`
  - `src/kd_sensing/config/io.py`
  - `src/kd_sensing/config/canonical.py`
  - CSI hardening sweep/config 生成入口
  - 训练日志与 TensorBoard 指标写入路径
- 受影响配置：
  - `configs/csi/hardening_matrix/*.yaml`
  - 需要新增或修正 debug-only configs/run specs
- 受影响测试：
  - `tests/test_csi_modality.py`
  - `tests/test_student_configs.py`
  - `tests/test_training_io_workflow.py`
  - 可新增针对 resolved config diff、pilot disabled identity、training health metrics 的单元测试
- 不新增外部依赖；所有 Python 验证命令使用 `conda run -n kd_mm_beam ...`。
