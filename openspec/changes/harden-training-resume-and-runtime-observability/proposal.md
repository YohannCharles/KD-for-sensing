## Why

当前训练恢复路径会在 `last.pth` 缺失或恢复角色字段不完整时静默退化为新训练，并且没有保存足以重建随机轨迹、AMP、扩展状态和日志历史的运行时状态。与此同时，checkpoint 写出、最终测试、指标聚合和性能观测存在原子性、来源标注或确定性缺口，可能让“已恢复”“已选模”和“已完成测试”的产物与实际执行不一致。

## What Changes

- 将 resume 改为 fail-closed 预检：在构建可变训练资源、覆盖运行配置或执行 optimizer step 前解析恢复路径，并校验 resume 角色必需字段。
- 引入版本化 `runtime_state`，保存并恢复 Python、NumPy、Torch CPU/CUDA RNG、DataLoader generator、GradScaler、training extension、history 和 epoch logs；用不可变 config、split 与 normalization fingerprint 阻止不兼容恢复，只允许显式 allowlist 中的运行控制字段变化。
- 明确跨 run 恢复、零剩余 epoch 和自定义 checkpoint selection 的语义，确保最终评估加载并记录实际选中的 checkpoint，而不是固定假设 `best.pth`。
- 让所有训练 checkpoint 与 sidecar 使用原子写出和内容摘要，并让每个文件记录自身准确的 selection provenance。
- **BREAKING**：新 checkpoint schema 不再把 validation loss 复制为 `test_loss`；历史 `test_loss` 仅由有版本标记和警告的 legacy 迁移路径读取。
- 将 `final_test_metrics` 与 validation/通用 `metrics.json` 写出分离，先补齐 split、checkpoint 和 selection provenance，再一次性发布最终测试结果，避免测试阶段覆盖验证指标。
- 修正训练指标按有效样本或 token 加权、跳过 validation 时不复制旧值、批量标量同步、persistent worker 生命周期和 timing/profile 行为。
- 让 synthetic dataset/index 与 DataLoader generator 从稳定身份派生，避免共享 generator 或构建顺序改变样本和 batch 顺序。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `original-code-compatibility`：收紧 resume 路径、角色 schema、不可变 fingerprint、跨 run 与 legacy checkpoint 恢复契约。
- `training-evaluation-runtime`：补齐版本化运行时状态、最终测试选模、加权指标、worker 生命周期、同步与 timing/profile 契约。
- `experiment-artifact-registry`：补齐全部 checkpoint 的原子发布、sidecar digest、逐文件 selection provenance 和最终测试产物边界。
- `dataset-loader-behavior`：补齐可恢复 DataLoader generator、persistent worker 和 synthetic index 的确定性契约。

## Impact

- 主要影响 `src/kd_sensing/engine/checkpointing.py`、`training_state.py`、`trainer.py`、`training_metrics.py`、`trainer_runtime_helpers.py`、`batch_step.py`、`data_factory.py`、训练 artifact writer 和 checkpoint registry helper。
- 数据侧影响 synthetic dataset/index、DataLoader/sampler generator 构建与状态交接；不读取或修改真实 `dataset/`。
- checkpoint 与 sidecar 的新 schema 需要 focused migration/compatibility tests；已有历史 checkpoint 保持只读兼容，新产物不保留含义错误的 `test_loss` alias。
- 不新增依赖、console script、实体实验配置或训练任务；验证使用 `kd_mm_beam` 下的 synthetic/fixture 测试，并保持产物位于测试临时目录或 ignored `outputs/`。
