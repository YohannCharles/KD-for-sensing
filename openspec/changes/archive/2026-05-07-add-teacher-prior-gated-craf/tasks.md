## 1. Teacher Registry 与指标产物

- [x] 1.1 梳理现有单模态 teacher 输出，确认 best checkpoint、`train_log.json`、registry sidecar 和 metrics 字段的可用来源。
- [x] 1.2 实现 teacher metrics 导出或读取适配，保证每个单模态 teacher 至少提供 `modality`、`best_epoch`、`val_acc_top1`、`val_acc_top3`、`val_acc_top5`、`val_adba` 和 `train_acc_top1`。
- [x] 1.3 新增 `scripts/build_teacher_registry.py`，支持 teacher root、输出路径、scene、prior mode、manual prior、metric prior 权重和 prior min/max 参数。
- [x] 1.4 实现 manual prior 生成，默认 Scene32 使用 image 0.20、radar 0.20、gps 0.85、lidar 0.15、mmwave 0.90。
- [x] 1.5 实现 metric prior 生成，按 `0.6 * val_acc_top1 + 0.2 * val_acc_top3 + 0.2 * val_adba` 归一化并 clamp。
- [x] 1.6 为 teacher registry 构建脚本增加单元测试，覆盖缺失 metrics、模态不一致、manual prior 和 metric prior。

## 2. Prior Gate 与 Loss

- [x] 2.1 新增 `PriorResidualGate` 或等价 gate 模块，实现 `logit(prior) + residual_logit`、`min_gate`、modality mask 和 confidence 输入。
- [x] 2.2 将 residual MLP 最后一层权重和 bias 初始化为 0，确保未训练时 gate 接近 prior。
- [x] 2.3 扩展 CRAF gate 构建逻辑，支持 `gate_type: none`、`fixed_prior`、`prior_residual_sigmoid` 和旧 gate。
- [x] 2.4 扩展 CRAF forward diagnostics，输出 `gate`、`gate_logits`、`prior`、`residual_logits` 和按模态聚合所需字段。
- [x] 2.5 实现 `prior_regularization_loss(gate, prior, modality_mask, loss_type)`，支持 MSE 和 L1，并只在可用模态上计算。
- [x] 2.6 增加 PriorResidualGate 和 prior regularization 单元测试，覆盖初始化误差、不可用模态 mask、MSE/L1 和 diagnostics。

## 3. Teacher Encoder Loader

- [x] 3.1 新增 teacher loader 窄模块，负责读取 teacher registry、解析 checkpoint、执行模态到 CRAF encoder 的 key mapping。
- [x] 3.2 支持从现有单模态 teacher 的 `feature_extraction` 或等价 feature extractor key 加载到 `craf_fusion.encoders.<modality>`。
- [x] 3.3 实现 strict 与 non-strict 加载模式，记录 missing、unexpected 和 shape mismatch。
- [x] 3.4 实现 Stage 2 encoder 冻结策略，冻结加载成功的 encoder，保留 fusion transformer、head 和 gate 可训练。
- [x] 3.5 实现 Stage 3 选择性冻结/解冻 helper，按 `finetune.unfreeze_modalities` 和 `finetune.freeze_modalities` 设置 `requires_grad`。
- [x] 3.6 为 teacher loader 增加合成 checkpoint 测试，覆盖成功加载、shape mismatch、strict 报错、只加载配置启用模态和冻结状态。

## 4. 训练流程与 Optimizer

- [x] 4.1 在 `trainer.py` 中于 optimizer 构建前接入 teacher-prior 初始化流程，并将 teacher registry 路径写入 runtime/final config。
- [x] 4.2 扩展 checkpoint 加载流程，支持 Stage 3 从 Stage 2 best checkpoint 加载后再应用选择性 finetune 策略。
- [x] 4.3 扩展 optimizer builder，支持 Stage 3 参数组，分别配置 fusion、head、gate、strong encoder 和 weak encoder 学习率。
- [x] 4.4 确保 frozen 参数不进入 optimizer，并在训练日志中记录每个参数组的学习率和参数量。
- [x] 4.5 接入 prior regularization loss，默认 Stage 2 权重 0.05，Stage 3 权重 0.01，权重为 0 时完全关闭。
- [x] 4.6 保持 Stage 2/3 默认 counterfactual、unimodal auxiliary 和 KD 有效权重为 0。
- [x] 4.7 实现可选 reliability-weighted KD loss，默认关闭，显式启用时支持只使用 GPS/mmWave teacher。
- [x] 4.8 实现 relative context marginal 和 shuffle counterfactual ablation helper，确保 delta 只使用 CE 或 label-smoothed CE。

## 5. 诊断日志与验证子集

- [x] 5.1 扩展 epoch diagnostics，记录每模态 `craf/gate_mean/*`、`craf/prior/*` 和 `craf/residual_logit_mean/*`。
- [x] 5.2 扩展 teacher 初始化日志，记录每模态 load success、load summary、frozen 状态和 trainable parameter count。
- [x] 5.3 扩展 TensorBoard 写入逻辑，写出 gate、prior、residual、prior regularization loss 和 teacher load/freeze 关键标量。
- [x] 5.4 扩展验证流程，支持 `evaluation.modality_subsets`，用 force modality mask 评估 gps、mmwave、gps_mmwave、strong_only、weak_only 和 all。
- [x] 5.5 确保非 CRAF 或不支持 force mask 的模型跳过模态组合验证，并保持默认验证指标不变。
- [x] 5.6 增加日志与验证子集测试，确认 `train_log.json` 或等价 epoch metrics 包含新增字段。

## 6. 配置与文档

- [x] 6.1 新增或补齐 Scene32 image、radar、gps、lidar、mmwave teacher-prior Stage 1 配置入口。
- [x] 6.2 新增 `scene32_stage2_teacher_init_prior_residual` 主实验配置，启用 teacher encoder load/freeze、prior residual gate 和 prior regularization。
- [x] 6.3 新增 `scene32_stage3_selective_ft_gps_mmwave` 主实验配置，加载 Stage 2 checkpoint 并只解冻 GPS/mmWave encoder。
- [x] 6.4 新增消融配置：teacher init no prior、prior gate random encoder、teacher init fixed prior、teacher init prior residual。
- [x] 6.5 保留并回归 `token_transformer_all_modalities_no_kd`、`craf_all_modalities_fixed_prior_sanity` 和既有 CRAF stabilized 配置。
- [x] 6.6 更新 README 或实验说明，记录 Stage 1/2/3 命令、teacher registry 构建命令、主实验命名、日志判据和失败排查方式。

## 7. 测试与验证

- [x] 7.1 使用 `conda run -n kd_mm_beam pytest tests/test_craf_fusion.py` 或等价定向测试验证 CRAF gate、prior loss 和旧 CRAF 回归。
- [x] 7.2 使用 `conda run -n kd_mm_beam pytest tests/test_teacher_prior_craf*.py` 或等价新增测试验证 teacher registry、teacher loader、Stage 2 冻结和 Stage 3 解冻。
- [x] 7.3 使用 `conda run -n kd_mm_beam pytest tests/test_config*.py tests/test_student_configs.py` 或等价测试验证新增配置和 legacy 配置不回退。
- [x] 7.4 使用 `conda run -n kd_mm_beam python scripts/build_teacher_registry.py --help` 验证 teacher registry 脚本入口可用。
- [x] 7.5 使用 `conda run -n kd_mm_beam python -m kd_sensing.cli.train --config <stage2-smoke-config> --override training.epochs=1 data.dataloader.train_batch_size=2 data.dataloader.test_batch_size=2` 完成 Stage 2 短训练 smoke test。
- [x] 7.6 使用 `conda run -n kd_mm_beam python -m kd_sensing.cli.train --config <stage3-smoke-config> --override training.epochs=1 data.dataloader.train_batch_size=2 data.dataloader.test_batch_size=2` 完成 Stage 3 短训练 smoke test。
- [x] 7.7 使用 `conda run -n kd_mm_beam pytest` 运行完整测试套件，确认单模态、legacy fusion、KD、CRAF 和新增 teacher-prior 路径不回退。
- [x] 7.8 使用 `conda run -n kd_mm_beam openspec status --change add-teacher-prior-gated-craf` 确认 OpenSpec 状态可追踪。
