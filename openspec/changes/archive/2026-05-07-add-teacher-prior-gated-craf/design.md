## Context

当前项目已经具备配置驱动训练、组件注册、场景化输出目录、单模态 teacher/student、五模态 fusion、`craf_fusion`、`token_transformer_fusion`、fixed-prior sanity 配置、CE-only counterfactual target 和 CRAF 诊断日志。现有 CRAF 的主要缺口不是能否运行，而是 gate 学习仍依赖训练中的 noisy counterfactual signal，容易把弱模态 gate 推高。

方案 3 引入的关键变化是训练策略和初始化边界：先用单模态 teacher 验证指标确定模态可靠性 prior，再用 teacher encoder 初始化 CRAF 分支，并在 frozen encoder 上训练 fusion/gate/head。只有当 fusion 学稳后，Stage 3 才选择性解冻 GPS/mmWave encoder，以避免 image、LiDAR、radar 弱分支继续污染验证泛化。

## Goals / Non-Goals

**Goals:**

- 提供可复现的 Stage 1/2/3 训练入口，优先覆盖 Scenario 32 五模态实验。
- 让 Stage 2 初始 gate 接近 teacher prior，且 residual gate 最后一层零初始化。
- 支持手动 prior 和 metric prior 两种 teacher registry 生成方式。
- 在 Stage 2 默认冻结全部 teacher-initialized encoder，仅训练 fusion transformer、prediction head 和 prior residual gate。
- 在 Stage 3 默认只解冻 GPS/mmWave encoder，并使用独立小学习率参数组。
- 保留已有 CRAF、fixed prior、token transformer 和 counterfactual 路径，新增能力只在显式配置中启用。
- 提供测试覆盖，确认 gate 初始化、冻结/解冻、teacher loader、prior loss、配置加载和短训练 smoke test。

**Non-Goals:**

- 不把所有 canonical fusion 默认切换到 teacher-prior CRAF。
- 不默认启用 reliability-weighted KD、relative counterfactual 或 shuffle counterfactual。
- 不重写 dataset、collate、DeepSense6G split、LiDAR/mmWave 归一化和 cache 策略。
- 不引入新的训练入口框架；继续复用 `scripts/train.py`、`python -m kd_sensing.cli.train` 和现有 `trainer.py`。
- 不在第一批实现中全模态 encoder 解冻或大规模调参。

## Decisions

1. Stage 1 复用现有单模态 teacher 体系，必要时新增薄配置别名，而不是复制一套平行训练脚本。

   当前仓库已经有 `image_teacher`、`radar_teacher`、`gps_teacher`、`lidar_teacher`、`mmwave_teacher` 与 canonical no-KD 配置，并且这些模型输出 `(pred, features, output_features)`。方案 3 中的 `SingleModalTeacher` 目标是得到每模态可靠 teacher checkpoint 与可迁移 encoder；在本代码库中更稳妥的边界是优先复用已有 teacher 模型和 feature extractor。如果后续需要 transformer-style single-modal teacher，可作为同一 registry 下的新模型补充，但 Stage 2 loader 首批只依赖可稳定映射的 encoder/feature extractor 权重。

2. Teacher encoder 加载只加载可验证兼容的分支权重，并显式记录失败。

   `teacher_loader` 从 teacher registry 读取 checkpoint 后，将 teacher 的 `feature_extraction` 或对应 `<modality>_feature_extractor` 权重映射到 CRAF 的 `encoders.<modality>`。如果 shape 或 key 不匹配，loader 必须返回 missing/unexpected/shape mismatch 摘要，并写入训练日志；严格模式下直接失败，非严格模式下只跳过不兼容 key。默认不加载 teacher GRU、attention 和 classifier，因为 CRAF 的跨模态 transformer/head 与单模态 GRU 语义不同。

3. PriorResidualGate 独立于旧 ReliabilityEstimator 实现。

   现有 `ReliabilityEstimator` 已承担 sigmoid、softmax 和 fixed-prior 逻辑。Prior residual gate 的核心是 `logit(prior) + residual_logit`，且 residual 最后一层必须零初始化。把它做成独立模块或独立 gate class，可以避免把旧 gate 的 dataset prior 语义和新 prior logit 初始化混在一起。CRAF forward 需要兼容 gate 返回 dict，并将 `gate`、`prior`、`residual_logits` 和 `gate_logits` 放入 diagnostics。

4. Stage 2 默认不启用 counterfactual、unimodal aux 和 KD。

   当前失败模式说明 noisy counterfactual target 和弱单模态 auxiliary 可能把 gate 拉错。Stage 2 的目标是验证 teacher init + prior residual gate 是否能超过 fixed-prior sanity，因此默认 loss 只包括 task CE、beam soft 小权重和 prior regularization。counterfactual 和 KD 作为后续 ablation 保留配置入口，不能阻塞主流程。

5. Prior regularization 使用可用模态 mask，只约束存在模态。

   `prior_regularization_loss(gate, prior, modality_mask)` 必须只在 `modality_mask=True` 的位置计算 MSE 或 L1。这样它同时兼容完整五模态训练、modality dropout 和未来真实缺失模态。Stage 2 默认权重 0.05，Stage 3 默认权重 0.01。

6. Stage 3 的冻结与 optimizer 参数组由模型模块名解析。

   CRAF 已用 `encoders` 和 `feature_projections` 按模态组织分支。Stage 3 应通过 `finetune.unfreeze_modalities` 和 `finetune.freeze_modalities` 设置 `requires_grad`，并构建 fusion/head/gate/strong encoder/weak encoder 参数组。这样避免在训练循环里硬编码具体类名，也便于测试每个 param group 是否非空。

7. 评估模态组合通过 force modality mask 复用现有 CRAF forward。

   `craf_fusion` 已支持 `force_modality_mask`。验证阶段新增 `evaluation.modality_subsets` 后，应对同一个 dataloader 额外跑 GPS、mmWave、GPS+mmWave、strong_only、weak_only、all 等组合，并把结果写入 metrics/train log。非 CRAF 或不支持 force mask 的模型必须跳过该扩展，不影响原验证指标。

8. 配置布局跟随现有 `configs/fusion/` 和 `configs/<modality>/`。

   为减少 README 中新路径心智负担，单模态 teacher 继续优先使用 `configs/<modality>/teacher_no_kd.yaml`，同时新增 Scenario 32 teacher-prior 专用入口或覆盖配置。Stage 2/3 与消融配置放在 `configs/fusion/` 下，run name 使用 `scene32_stage2_teacher_init_prior_residual` 和 `scene32_stage3_selective_ft_gps_mmwave`。

## Risks / Trade-offs

- [Risk] 现有单模态 teacher encoder 与 CRAF encoder 命名或形状不完全一致。Mitigation：loader 先支持清晰 key mapping 和 shape check，严格模式用于主实验，非严格模式用于定位差异，并把每模态 load summary 写入日志。
- [Risk] 只加载 feature extractor 不等价于方案中的 transformer teacher encoder。Mitigation：首批实现先验证最小可迁移表示边界；如果收益不足，再新增 transformer-style single-modal teacher 并复用同一 loader contract。
- [Risk] prior regularization 太强会压制 sample-wise residual。Mitigation：Stage 2/3 分别使用 0.05/0.01 默认值，并记录 residual logit 均值与 gate-prior 偏移。
- [Risk] Stage 3 参数组漏掉模块会导致实际未微调。Mitigation：增加冻结/解冻测试，训练日志记录每模态 frozen 和 trainable parameter count。
- [Risk] 模态组合验证增加验证耗时。Mitigation：只在显式 `evaluation.modality_subsets.enabled=true` 时运行，默认组合数量固定且只记录 epoch 标量。
- [Risk] reliability-weighted KD 可能被弱 teacher 污染。Mitigation：默认关闭，并默认只允许 GPS/mmWave teacher 参与首批 KD ablation。

## Migration Plan

1. 新增 teacher registry 构建脚本和 registry JSON schema 测试，支持从已有 teacher 输出读取 metrics 和 checkpoint。
2. 新增 PriorResidualGate、prior regularization loss 和 teacher loader 单元测试。
3. 扩展 `craf_fusion` gate factory/diagnostics，接入 `gate_type: prior_residual_sigmoid`，保持旧 gate 配置不变。
4. 扩展训练初始化流程，在 optimizer 构建前加载 teacher encoder、应用冻结策略和记录 load/freeze 状态。
5. 扩展 optimizer builder，支持 Stage 3 参数组和选择性解冻。
6. 新增 Stage 2/3 配置和 teacher/prior 消融配置，并更新 README 实验顺序。
7. 增加定向测试和短训练 smoke test，随后运行完整测试套件。

回滚策略是使用已有 `token_transformer_all_modalities_no_kd`、`craf_all_modalities_fixed_prior_sanity` 或旧 `craf_all_modalities_stabilized_no_kd` 配置。新增逻辑均由显式配置启用，legacy 单模态、fusion 和 KD 配置不需要迁移。

## Open Questions

- Stage 1 首批主实验是否必须新增 transformer-style `single_modal_teacher`，还是先复用当前 GRU teacher 的 feature extractor 权重即可。默认按复用现有 teacher 执行，以降低实现风险。
- `metrics.json` 中 prior 计算使用 `val_acc_top1`、`val_acc_top3` 和 `val_adba` 的权重是否固定为 0.6/0.2/0.2。默认按方案 3 实现，并暴露配置覆盖。
