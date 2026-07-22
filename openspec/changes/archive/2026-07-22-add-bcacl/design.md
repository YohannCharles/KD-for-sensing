## Context

当前 T2 的四模态编码器各输出 `[B,T,64]`，时序池化后形成 `[B,M,64]`；同一融合 Beam 原型头产生逐模态 logits，监督 Router 再融合这些 logits。训练期 `modality_temporal_mask[B,T,M]` 表示 synthetic dropout 后的融合可用性，启用 `preserve_unmasked_for_superset` 时，数据管线另存 dropout 前输入与基准可用性。当前没有模态私有分类头、独立模态原型库或通用的两阶段训练切换。

BCACL 必须增强单模态编码器，又不能改变当前推理融合和第一创新点的融合恢复原型；关闭时还必须保持既有模型 state dict 与 forward 数值一致。原型和教师质量只能来自训练 batch，validation/test 不得参与更新。

## Goals / Non-Goals

**Goals:**

- 在共享 64 类 Beam 关系空间中执行单向、停止教师梯度的互补迁移。
- 支持私有头、共享头、固定教师、质量教师和 detached two-stage 的独立消融。
- 通过现有 checkpoint/extension 机制保存并恢复 BCACL 参数、原型、质量与阶段身份。
- 持久化单模态、教师选择、逐类质量和损失诊断，并汇总 15 个非空模态组合。
- 保持配置关闭路径和 Phase 2 推理路径完全兼容。

**Non-Goals:**

- 不实现动态融合权重、置信度加权、门控、MoE 或样本级推理可靠性加权。
- 不直接对齐异构原始特征，不做全模态两两特征对比。
- 不替换、复用或改写现有融合恢复原型库。
- 不自动运行 outer test、multi-seed、大规模搜索或正式 claim 更新。

## Decisions

### 1. BCACL 作为 T2 的 opt-in companion module

仅当顶层 `bcacl.enabled=true` 时，trainer 把解析后的 BCACL 配置传给 `UMaskBeamJEPA`，模型才实例化独立的 `BCACLModule`。该模块包含每模态 `Linear + LayerNorm` 投影、可选私有线性头、可选共享线性头，以及名为 `modality_prototypes`、`quality_matrix` 的 buffers。关闭时模型不创建该属性，既有 state-dict key、参数初始化顺序、forward 与推理结果不变。

未采用把 BCACL 塞入现有 `BeamPrototypeBank` 的方案，因为这会混淆单模态统计原型与融合恢复原型的所有权，并破坏第一创新点的独立性。

### 2. observed 与 fusion 可用性复用 preserved-superset 数据契约

Phase 1 默认要求 `temporal_missing.preserve_unmasked_for_superset=true`。extension 从 preserved payload 读取自然可用的 `base_mask` 作为 `observed_mask[B,M]`，把当前 batch mask 保存为 `fusion_mask[B,M]`。当 `distill_from_pre_dropout_modalities=true` 时，Phase 1 forward 使用 preserved 原始输入和 observed mask；融合 dropout 仍保留在 `fusion_mask` 诊断中，不会被解释成自然观测。

未采用从零张量猜测模态是否存在的方案，因为零值可能是合法输入，也会把人工 dropout 错当成自然缺失。

### 3. Beam 关系损失只作用于学生路径

每个模态用自己的原型计算 `log_softmax(cos(normalize(z), normalize(P_m))/temperature)`。无效原型在 softmax 前屏蔽；没有有效类别时返回有限的均匀分布且不允许迁移。KL 使用 `F.kl_div(student_log_prob, teacher_prob.detach(), reduction="none")` 后按有效迁移项平均，保证教师不被学生梯度拉动。

固定教师由配置中的 dataset-type 映射解析，不在 loss 内硬编码 MMW/LiDAR 或 DeepSense6G/Image。自动模式按真实 Beam 类别、observed mask、初始化状态和质量差选择每个学生最多一个教师。

### 4. epoch 统计同时支持替换和 EMA

每个 Phase 1 batch 以 float32 累积归一化投影的 `[M,K,D]` 和与 `[M,K]` count；DDP 下在 epoch 末 all-reduce。`prototype.update=epoch` 用满足最小样本数的新均值替换原型，`ema` 则对同一训练集 epoch 均值做 EMA；样本不足保持旧值。质量的类内项由归一化向量和的范数精确得到，类间项默认取当前模态原型空间中最近的有效负类，避免假设 Beam 0/63 物理相邻。

未采用额外第二遍训练集扫描，避免将每个 epoch 的 I/O 和编码开销翻倍；聚合和向量范数足以计算定义中的平均 cosine 类内方差。

### 5. 两阶段使用两个显式运行

`bcacl.stage=phase1` 用 BCACL 基础损失完全替代融合基础损失，融合 Router 与融合恢复原型不接收梯度。`bcacl.stage=phase2` 从 Phase 1 checkpoint 做 model-only initialization，重置 optimizer/scheduler/RNG 轨迹并在建 optimizer 前冻结 encoders、encoder projections 和全部 BCACL 参数，只训练既有融合与融合恢复原型路径。默认不提供隐式全模型 joint refine；配置字段存在但必须显式启用。

未采用单次运行中途切换 `requires_grad`，因为现有 optimizer 在 extension setup 前构建，中途解冻会产生未注册 optimizer 参数并削弱精确 resume。

### 6. 诊断和实验边界

extension 将标量交给现有 epoch/TensorBoard 日志，并在 run directory 下追加 JSON/CSV：模态 CE/top1、4x4 迁移矩阵、逐 Beam 教师、样本计数、原型初始化率和质量。fixed-mask 评估器继续逐 pattern 输出原指标，并增加 Single/Double/Triple/All-14 macro 与 worst；不存在的通信效用字段不伪造。

实验 launcher 只生成 single-seed inner/development、`claim_eligible=false` 的 U0--U5 和 A--D 任务，检测当前空闲 GPU 后一 GPU 一任务，不抢占已有进程。

## Risks / Trade-offs

- [Phase 1 首个 epoch 尚无原型，固定教师 KL 暂不可用] → 首个 epoch 只训练私有/共享 CE，epoch 末初始化原型，从下一 epoch 开始迁移。
- [优势教师过强导致关系分布塌缩] → 默认较小 `lambda_bcacl`、质量 margin、top-1 稀疏迁移，并记录关系/教师统计。
- [低样本 Beam 质量不稳定] → `min_class_count` 和独立有效 mask 禁止其成为教师，质量使用 EMA。
- [两阶段 checkpoint 配置错配] → launcher 绑定 source SHA、role、schema、dataset 和 stage，model-only initialization 严格校验 key/shape。
- [工作树已有并行实验修改] → BCACL 使用独立模块、change、测试和输出目录；对共享 T2 文件只做局部 opt-in 修改。

## Migration Plan

1. 先合入默认关闭配置和 disabled-path 等价测试。
2. 合入 BCACL 模块与 Phase 1 focused tests，再合入 Phase 2 初始化/冻结测试。
3. 运行 U0 smoke 确认旧 checkpoint 严格加载和 forward 数值不变。
4. 依次运行 U1/U2、固定教师、自动教师；任何阶段失败均停止后续晋级。
5. 回滚只需关闭或删除顶层 `bcacl` 配置；既有 checkpoint 和推理路径不依赖 BCACL。

## Open Questions

- 现有仓库未提供数据集码本角度映射以证明 Beam 0/63 的物理邻接，因此首版只将 `prototype_nearest` 作为默认 hard negative；确认码本后再启用 beam-neighbor 模式。
- 信道功率矩阵并非所有当前 loader 都保证提供；评估只在 batch 已有对应字段时保留现有通信指标。
