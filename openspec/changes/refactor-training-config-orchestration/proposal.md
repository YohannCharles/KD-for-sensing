## Why

项目已经完成 `src/kd_sensing` 包结构、registry、模态契约和轻量导入边界的第一轮治理，但新的实验能力继续把横切逻辑压回 `engine.trainer.train()` 和 `config/io.py`。继续在这两个入口上叠加 Raymobtime、CSI、CRAF/MARF/G2D、诊断和兼容逻辑，会让新增实验的修改半径越来越大，也会放大文档、OpenSpec、安装入口和源码之间的漂移。

## What Changes

- 将训练入口收敛为生命周期编排，保留 `kd_sensing.engine.trainer.train` 公开入口和现有 CLI 行为，但把运行目录、checkpoint、history、TensorBoard、epoch log、训练 batch step 和最终 artifact 写出拆到职责明确的窄模块。
- 将配置加载流程拆成可审查 pipeline：`config/io.py` 负责 YAML/virtual config/命令行覆盖的入口协调，objective 默认值、模态推导、dataset 专属规则、迁移拒绝和 schema 校验进入独立 helper。
- 为训练输出兼容性增加 characterization 约束，确保重构前后的 `final_config.yaml`、`train_log.json`、`training_outputs.npz`、checkpoint sidecar、early stopping metadata 和 TensorBoard 标量 key 保持兼容。
- 收口 CLI/spec/docs 漂移，特别是 `kd-sensing-visualize-modalities` 兼容入口必须恢复为可用入口，或在同一变更中从 spec 和文档中移除该承诺。
- 增加架构边界测试，防止后续新增算法、数据集或分析功能继续把职责直接写回 `trainer.py`、`config/io.py` 或已拆分的运行时 helper。
- 不改变模型结构、loss 数值定义、默认配置路径、checkpoint 加载语义、实验指标名称或当前 Raymobtime/DeepSense6G/MMW/CSI 工作流的用户可见行为。

## Capabilities

### New Capabilities

- 无。本变更收敛既有训练、配置和入口职责，不引入新的实验能力。

### Modified Capabilities

- `project-architecture`: 增加训练编排、配置 pipeline、兼容 CLI 入口和架构边界测试的职责约束。
- `canonical-config-resolution`: 明确 `config/io.py` 与 config normalization、validation、dataset-specific rule 和 canonical recipe 的边界。
- `experiment-workflow`: 强化重构后的训练输出、checkpoint、日志、TensorBoard 和 CLI help 行为必须保持兼容。

## Impact

- 受影响源码：`src/kd_sensing/engine/trainer.py`、`src/kd_sensing/engine/training_extensions.py`、新增或调整的 `engine` 运行时/日志/checkpoint/artifact helper、`src/kd_sensing/config/io.py`、新增或调整的 `config` normalization/validation helper、`src/kd_sensing/cli/` 和 `pyproject.toml`。
- 受影响测试：训练 IO workflow、architecture boundaries、prediction objective、Raymobtime smoke、viewer manifest/兼容 CLI help、canonical config resolution 和可能新增的 characterization tests。
- 受影响文档/OpenSpec：README 或 `tools/visualization/README.md` 中的兼容入口说明，`openspec/specs/project-architecture`、`canonical-config-resolution` 和 `experiment-workflow` 的增量要求。
- 验证命令继续使用 `kd_mm_beam` 环境；本变更不提交 `outputs/`、`logs/`、cache、checkpoint 或本地数据产物。
