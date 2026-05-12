## Why

项目已经完成包结构、registry、模态契约和轻量导入边界治理，但新增 G2D、CRAF、MARF、互补分析和可视化功能后，复杂度继续集中到少数枢纽文件。当前 `engine/trainer.py`、诊断 core 和 fusion YAML 矩阵承担了过多横切职责，每次增加实验方法都需要同时改训练主循环、loss 接入、诊断日志、配置和测试，导致代码膨胀和回归面扩大。

本变更的目标是把“新增实验方法”从侵入式主流程修改，收敛为可注册、可组合、可验证的训练扩展点和配置 overlay，先控制增长路径，再考虑更大的重构。

## What Changes

- 在训练引擎中引入窄的训练扩展边界，用于承载 G2D、CRAF、MARF 等方法特有的 teacher 准备、额外 loss、梯度后处理和 epoch diagnostics。
- 将 `engine/trainer.py` 收敛为训练生命周期编排：构建对象、迭代 epoch/batch、调用扩展点、保存 checkpoint 和运行产物；方法细节不再继续堆进主循环。
- 将 batch 输入准备和 model forward 统一为共享 runtime helper，避免 trainer、validator、viewer prediction 和 teacher ensemble 分别维护任务分支逻辑。
- 将 CRAF/MARF 的训练期 extra loss 与 diagnostics 聚合移动到职责明确的模块，保留现有行为和指标键。
- 将 G2D teacher ensemble 的构建、checkpoint 解析和前向准备从 distillation 算法层移到 engine runtime 层，保持 `distillation/g2d.py` 专注于 loss、confidence 和 feature/logit 对齐。
- 让诊断可视化目录完成真实职责拆分，避免 `visualization/core.py` 继续作为大型实现聚合文件。
- 为高级 fusion 实验引入 base + method overlay + ablation overlay 的配置生成/解析约束，减少 CRAF/MARF/G2D 场景配置复制。
- 增加架构回归测试，限制新增方法直接修改训练主循环或依赖大型兼容聚合层。
- 不改变模型结构、loss 数值语义、checkpoint key、公开 CLI 或现有配置行为。

## Capabilities

### New Capabilities

- 无

### Modified Capabilities

- `project-architecture`: 增加训练扩展点、runtime helper、诊断真实拆分和依赖方向约束，控制新增实验方法的主流程侵入。
- `configurable-multimodal-fusion`: 增加高级 fusion 方法配置 overlay 约束，减少 CRAF/MARF/G2D 配置复制并保持实体 YAML 兼容。

## Impact

- 主要影响 `src/kd_sensing/engine/trainer.py`、`engine/validator.py`、`distillation/teacher_ensemble.py`、`distillation/g2d.py` 的依赖边界、`engine/batch.py` 或新增 runtime helper、`engine/craf_training.py`、`engine/marf_training.py` 和诊断可视化内部模块。
- 可能新增少量 engine 子模块，例如 `engine/forwarding.py`、`engine/training_extensions.py`、`engine/g2d_training.py`、`engine/craf_losses_runtime.py` 或同等职责模块。
- 影响 `configs/fusion/` 的高级方法配置组织方式，但现有实体 YAML 必须继续可加载。
- 测试需要覆盖训练扩展点行为等价、轻量导入边界、G2D/CRAF/MARF 关键回归、配置 overlay 解析和架构边界。
