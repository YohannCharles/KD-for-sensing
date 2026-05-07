## Context

当前仓库已经形成配置驱动实验体系：模型、loss、metric、distiller 通过 `kd_sensing.registries` 构建；训练和评估入口集中在 `src/kd_sensing/engine/trainer.py` 与 `validator.py`；fusion 数据输入通过 `engine.batch.prepare_fusion_inputs` 根据 `modalities` 准备；canonical fusion 配置可以按固定模态顺序生成多模态组合。

`模态失衡改进方案1.md` 的 CRAF 原方案包含从零目录结构、独立 dataset/collate、独立 losses 和训练脚本。直接照搬会绕开现有统一入口，并破坏已经完成的单模态、fusion、KD、cache、normalization artifact 和 scene/split 记录能力。本设计选择把 CRAF 作为一个新的 fusion 模型族和一组训练扩展接入现有体系。

现有模型输出约定主要是 `(pred, input_features, output_features)`，其中训练流程会截取最后 `num_pred + 1` 个时隙与 `prepare_labels()` 生成的标签对齐。CRAF 的 horizon query 需要适配这个语义：默认输出 `num_pred + 1` 个预测 slot，覆盖当前 beam anchor 与未来 beam，而不是引入另一套标签接口。

## Goals / Non-Goals

**Goals:**

- 在 `src/kd_sensing.models.fusion` 内新增可注册的 CRAF fusion 模型，支持现有五种模态的任意有效组合。
- 复用现有模态契约、batch 准备、feature extractor、配置加载、训练输出、验证指标和 OpenSpec 约束。
- 支持 reliability-aware token fusion、单模态辅助预测、confidence 特征、dataset reliability prior、reliability gate 和 Transformer fusion。
- 支持配置驱动的 modality dropout 与 counterfactual gate supervision，并在关闭时保持普通 no-KD 训练路径。
- 提供 baseline 配置和测试，使 CRAF 能与现有 early-concat fusion、token-only transformer 和单模态实验横向比较。
- 保持已有 `fusion_teacher`、`fusion_student`、单模态模型和 KD 配置默认行为不变。

**Non-Goals:**

- 不重写 DeepSense6G dataset、collate、预处理和 CSV 生成流程。
- 不引入新训练脚本或绕开 `kd_sensing.cli.train`、`kd_sensing.cli.evaluate`。
- 不在第一阶段实现 raw image/LiDAR/Radar 大模型 backbone 替换；CRAF 优先复用已有帧级 feature extractor 或轻量 projector。
- 不在第一阶段实现跨模态 teacher-student 蒸馏；CRAF 可先以 no-KD 和内部辅助 loss 训练。
- 不要求默认 canonical fusion 模式全部切换到 CRAF；CRAF 必须显式配置启用。

## Decisions

1. 将 CRAF 作为新的注册模型接入，而不是改造现有 `FusionModalityNet`。

   `FusionModalityNet` 和 `StudentModalityNet` 已经承担 legacy early-concat teacher/student 语义，并被多组配置和测试使用。新增 `craf_fusion`、`token_transformer_fusion` 等注册名可以让实验配置显式选择新方法，同时避免旧 checkpoint 和旧实验行为被隐式改变。替代方案是在现有 fusion student 中加开关，但会让一个类同时维护两套完全不同的 forward 和 loss 语义。

2. CRAF 复用现有 batch 输入签名，并在模型内部构造 modality mask。

   当前 dataset 没有统一返回 `available_modalities` 字段，fusion 配置也默认所选模态都存在。因此第一阶段的 `modality_mask` 默认为启用模态全 True；若 batch 或训练 helper 后续提供 mask，则通过 `force_modality_mask` 与默认 mask 合并。这样能支持训练时 modality dropout/counterfactual，又不阻塞现有数据读取。替代方案是先改 dataset/collate 增加每样本可用性字段，但这会扩大范围并影响所有模态测试。

3. CRAF 输出通过小型适配层兼容 dict 与三元组。

   CRAF 需要暴露 `logits`、`reliability`、`unimodal_logits`、`confidence` 和 `memory` 等诊断字段；现有训练循环只消费三元组。实现时新增窄 helper 统一提取 logits/input_features/output_features/auxiliary，旧模型仍走原三元组路径。替代方案是让 CRAF 只返回三元组，但会丢失可靠性监督和日志所需信息。

4. horizon query 的预测长度使用 `num_pred + 1`。

   项目现有标签包含当前 beam anchor 和未来 `num_pred` 个 target，训练和验证都按 `num_pred + 1` 计算指标。CRAF 的 `prediction_horizon` 默认从配置中的 `num_pred` 推导为 `num_pred + 1`，从而复用 `prepare_labels()`、Top-K 和 DBA。替代方案是只预测未来 `num_pred`，但需要改标签和指标语义，不利于与历史实验比较。

5. Counterfactual 监督放在训练 helper 中，不塞进 distiller。

   distiller 当前处理 teacher/student logits/features 的 KD 组合。CRAF 的反事实 forward 需要访问模型输出 dict、force mask、per-sample CE 和 reliability gate，和 KD distiller 的职责不同。新增训练 helper 或 engine 内部函数可以仅在 `training.counterfactual.enabled` 时运行。替代方案是新增 distiller 类型，但 no-KD CRAF 也需要该能力，会让 distillation 概念变得混乱。

6. Beam-aware soft label loss 作为独立 loss helper 注册或窄函数提供。

   现有 `loss.type` 仍可保持 focal/cross_entropy。CRAF 训练总 loss 由普通任务 loss、beam soft loss、unimodal auxiliary loss 和 gate loss 组合而成。实现时应先提供可测试的函数，再接入训练配置。替代方案是替换现有 task criterion，但会影响所有非 CRAF 实验。

7. 任务分阶段落地，先保证可运行和可比较，再扩展论文级消融。

   第一阶段完成模型构建、forward、基础训练、反事实 gate loss、核心日志和 smoke tests；第二阶段补齐更多 baseline 配置、prior EMA 可视化和实验矩阵。这样更贴合当前项目状态，避免一次性改动训练、数据和配置全链路。

## Risks / Trade-offs

- [Risk] CRAF 参数量和 Transformer 计算增加，训练吞吐下降。Mitigation：默认 `d_model`、层数和 head 数保守；优先用 synthetic/small batch smoke test 和现有 throughput metadata 观察影响。
- [Risk] 可靠性 gate 早期训练塌缩，强模态过早压制弱模态。Mitigation：使用 `min_gate`、warmup、counterfactual start epoch、unimodal auxiliary loss 和可关闭的 gate loss 权重。
- [Risk] 现有训练循环对 dict 输出适配不完整。Mitigation：集中实现输出解析 helper，并用旧模型和 CRAF 模型分别覆盖训练/验证 forward。
- [Risk] 数据集没有真实缺失模态标记，难以验证自然缺失场景。Mitigation：第一阶段明确支持配置和训练强制 mask；若未来 dataset 提供 `modality_mask`，再接入真实缺失字段。
- [Risk] CRAF 与 KD distiller 同时启用时 loss 组合复杂。Mitigation：第一阶段推荐 CRAF no-KD；KD 组合仅在输出适配稳定后再显式支持。
- [Risk] 新 baseline 配置过多导致维护成本上升。Mitigation：优先提供代表性 all-modalities、image+radar 和弱/强模态组合配置，其它组合依赖可参数化配置生成。

## Migration Plan

1. 先添加新模块、注册名和单元测试，不触碰既有默认配置。
2. 再接入训练输出适配和 CRAF no-KD 训练路径，用 synthetic 或小数据 smoke test 验证。
3. 然后添加 counterfactual loss、beam soft loss、reliability 日志和 CRAF baseline 配置。
4. 最后补充文档和实验建议，确认旧 fusion 配置测试仍通过。

回滚策略是删除或停用 CRAF 配置和注册名；由于默认配置不切换，旧模型训练路径不需要迁移。
