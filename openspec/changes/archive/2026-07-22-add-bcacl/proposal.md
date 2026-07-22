## Why

四模态联合训练会让优势模态主导融合目标与编码器梯度，导致弱模态缺少独立的 Beam 判别证据；当优势模态缺失时，现有融合表示 Beam 原型修复也无法从弱模态中恢复不存在的判别信息。需要在不改变推理融合、不强制异构特征相等且不使用测试集统计的前提下，引入类别条件的非对称互补学习。

## What Changes

- 为 T2 增加默认关闭的 BCACL 训练能力：模态私有投影/分类头、共享 Beam 分类头、独立模态 Beam 原型库和 Beam 关系 KL。
- 支持固定教师与基于训练集原型质量的类别条件稀疏教师选择；所有迁移单向执行并停止教师梯度。
- 明确区分原始 `observed_mask` 与 synthetic dropout 后的 `fusion_mask`，只对真实观测模态计算单模态监督和迁移资格。
- 增加可恢复的 detached two-stage 训练：Phase 1 解耦训练编码器与 BCACL，Phase 2 默认冻结编码器并训练现有融合与融合恢复原型路径。
- 增加 U0--U5 独立消融、原型/质量/教师诊断持久化和完整 15 个非空模态组合汇总。
- 增加 single-seed、inner/development、claim-ineligible 的 smoke、固定教师与自动教师运行入口；不自动运行 outer test、multi-seed 或正式 claim。
- 保持现有融合结构、第一创新点的融合恢复原型、数据划分、预处理、随机种子、评估定义和 checkpoint 选择规则不变。

## Capabilities

### New Capabilities

- `beam-conditioned-asymmetric-complementary-learning`: 规定 BCACL 的单模态监督、模态原型、固定/自动教师、停止梯度、两阶段训练、诊断、消融和无泄漏边界。

### Modified Capabilities

- `u-mask-beam-jepa`: 增加默认关闭且不改变推理路径的 BCACL 训练分支，并保持融合恢复原型与模态原型严格分离。
- `training-evaluation-runtime`: 增加可恢复的两阶段训练状态、训练集 epoch 统计和 15-pattern 分组汇总，保持 outer test 与 claim 隔离。

## Impact

- 影响 `src/kd_sensing/models/`、`src/kd_sensing/losses/`、`src/kd_sensing/engine/`、当前 T2 配置、fixed-mask 评估汇总、BCACL focused tests 与本地实验脚本。
- 不新增依赖、公共 CLI、动态融合权重、MoE/门控、样本置信度加权、异构特征直接对齐或推理期分支。
- 新生成的 checkpoint、日志、原型统计和实验结果仅写入忽略的 `outputs/`，不纳入源码变更。
