## Context

项目当前以 `src/kd_sensing` 包、组件注册表、`model.primary`、配置驱动训练和 objective metadata 为核心。默认 supervised/adaptation 工作流会准备 beam label、调用主模型、计算 beam 或多任务预测 loss，并记录 Top-K/DBA 等指标；KD/teacher/distiller 路线已经退役。现有 image 输入契约固定为 RGB/ImageNet `[B, T, 3, 224, 224]`，GPS 输入使用 relative-polar `[B, T, 3]`，fusion batch 准备由模态契约驱动。

GPS-conditioned JEPA 与现有 beam 分类不同：它的 target 来自 image latent 本身，训练主指标是 latent prediction loss，而不是 `target_beam` 分类准确率。因此该能力需要同时触及模型注册、训练 batch step、objective metadata、验证指标和运行产物，但必须保持现有 supervised beam、GPS v2、Top8、BGAM、CSI 和 Raymobtime workflow 的默认行为不变。

## Goals / Non-Goals

**Goals:**

- 提供可配置、可注册的 GPS-conditioned JEPA 预训练模型，内部包含 context encoder、EMA target encoder、GPS conditioner 和 predictor。
- 支持 `experiment.objective: gps_conditioned_jepa`，训练与验证只依赖 image/GPS 输入和 JEPA mask，不要求 beam label 或 beam logits。
- 复用当前训练入口、输出目录、checkpoint、resume、TensorBoard、runtime metadata 和 `kd_mm_beam` 命令约束。
- 支持随机 patch mask 和 GPS angle biased patch mask，初始实现优先保证可测、可复现、内存有界。
- 保存可复用的 context encoder 权重或 manifest，为后续 supervised beam/fusion fine-tuning 提供初始化输入。

**Non-Goals:**

- 不恢复 `distillation.*`、frozen teacher checkpoint、KD loss 或旧 teacher/student registry。
- 不新增顶层旧脚本入口；训练仍通过 `kd-sensing-train` 或包内 CLI。
- 不在本 change 中承诺 SOTA JEPA 架构、跨数据集大规模预训练调参或完整论文实验矩阵。
- 不改变现有 beam supervised objective、GPS v2、Top8、BGAM、CSI hardening、Raymobtime selection 的默认配置语义。

## Decisions

### Decision 1: 将 JEPA 作为一等 objective，而不是 beam loss 的附加项

实现新增 `gps_conditioned_jepa` objective metadata，默认主指标为 `val_jepa_loss`、mode 为 `min`，required targets 为空或声明为 `self_supervised_image_gps`，required outputs 为 JEPA latent prediction payload。训练和验证在该 objective 下跳过 beam target/loss/Top-K/DBA 路径。

备选方案是把 JEPA loss 作为 beam supervised 的 auxiliary loss。该方案短期改动少，但仍要求 `target_beam`、会污染早停指标，并无法支持无标签预训练，因此不采用。

### Decision 2: 使用训练扩展接入 JEPA loss，并补充 optimizer-step 后 EMA hook

当前 `TrainingExtension` 已支持 `before_forward`、`compute_base_loss`、`after_forward`、`after_backward` 和 `after_epoch`。JEPA 可以通过 extension 提供 base loss，使主训练循环不复制一套完整 epoch 逻辑。为了正确更新 target encoder，需要在 optimizer step 和 grad scaler update 之后新增 `after_optimizer_step` hook，由 JEPA extension 调用模型的 `update_target_encoder()`。

备选方案是在模型 forward 内更新 EMA。该方案会把训练副作用放入 forward，验证和 AMP 下更难推理，也不利于 resume 和单元测试，因此不采用。

### Decision 3: JEPA model 输出使用显式 self-supervised payload

新增或扩展 model output adapter，使 `gps_conditioned_jepa` objective 下的模型输出可以包含：

- `predicted_target_latent`: `[B, T, N_tgt, D]`
- `target_latent`: `[B, T, N_tgt, D]`，stop-gradient
- `target_mask` / `context_mask`: `[B, T, N]`
- `loss_mask`: `[B, T, N_tgt]`
- scalar diagnostics，例如 mask ratio、EMA decay、latent norm

训练/验证代码在该 objective 下不要求 beam logits。备选方案是让模型返回 dummy logits 以复用旧 adapter；这会让无意义的 accuracy、Top-K 和 beam loss 更容易混入日志，因此不采用。

### Decision 4: 初始 visual token encoder 保持轻量、可替换

新增 JEPA 专用 visual token encoder 配置，默认使用 patch embedding + 小型 Transformer/MLP block 产生空间 token；可选支持从 ResNet-18 feature map 提取 token，但不要求第一版下游 supervised 模型必须共享同一 encoder 类。context encoder 和 target encoder 使用同构配置，target encoder 初始化自 context encoder，并通过 EMA 更新。

备选方案是强行复用现有 `ResNet18ImageEncoder`。该 encoder 当前只输出帧级 `[B, T, D]` embedding，不保留空间 patch token，直接复用会把 JEPA 退化成帧级 latent matching，难以表达 context/target patch masking，因此不作为默认。

### Decision 5: GPS conditioning 默认采用 FiLM，保留 cross-attention 扩展口

默认 conditioner 使用 GPS-Rel-Polar `[B, T, 3]` 生成 `gamma/beta`，对 context latent 执行 FiLM 调制。配置保留 `conditioning.type`，第一版支持 `film` 和 `concat_mlp`，后续可扩展 cross-attention。GPS conditioning 不读取 raw latitude/longitude，不绕过现有 GPS 特征模式。

备选方案是把 GPS 作为普通 token 与 image patch token 拼接后交给一个大 Transformer。该方案更灵活，但对小样本和当前配置规模更重，第一版不采用。

### Decision 6: Patch sampling 先做 deterministic mask builder

新增 mask builder，输入 image token grid、GPS relative angle、epoch/step/seed，输出 context/target mask。`random` 模式用于 smoke 和消融；`gps_angle_biased` 模式按 GPS 方位在 patch grid 上构造权重并采样 target/context，要求同一样本内 context 与 target 非重叠，并在 diagnostics 中记录有效 mask ratio。

备选方案是在 dataset 阶段提前写出 mask。该方案会把训练随机性固化到 CSV 或 cache，增加数据产物边界复杂度，因此不采用。

### Decision 7: 预训练产物显式记录可复用权重

checkpoint 保存完整 `model.primary`，并额外在 run artifacts 中记录 context encoder state dict key、visual token encoder配置、GPS conditioner 配置、masking 配置和 image/GPS profile。可选导出 `context_encoder.pth` 或 manifest，供后续 fine-tuning change 或手动加载使用。第一版只保证产物可定位和可加载，不在所有 supervised 配置中自动启用迁移学习。

### Decision 8: 主实验采用 DeepSense6G paper-split 风格的多场景训练

canonical smoke 配置可以继续使用单场景 scene31 以便快速检查链路；主 JEPA 预训练和 GPS-biased mask ablation 配置采用更贴近 BeamBench/Image AE 论文协议的场景划分：训练拼接 scenes 32、33、34，验证/监控覆盖 scenes 31、32、33、34。多场景训练通过通用 DataFactory 拼接 DeepSense6G scene dataset，并记录 multi-scene metadata，避免把主实验误标成 scene31-only。

### Decision 9: 下游复用验证必须区分 model selection 与 final test

JEPA context encoder 的 supervised image+GPS 下游复用验证新增 BeamBench-fair low-memory 配置。该配置族使用 scenes 32、33、34 的训练 split 作为训练来源，并从训练 split 内部固定划出 validation 子集做 early stopping/checkpoint selection；训练完成后，runtime 重新加载 `best.pth` 并在 scenes 31、32、33、34 的 test split 上做一次 final test。最终 test 指标写入运行 metadata 的 `final_test_metrics`，并记录 `model_selection_split`，避免把 test set 同时用作选模和汇报。

该配置族的 DBA 使用 `evaluation.dba_distance_mode: linear`，以匹配 BeamBench helper 的非环形 beam index 距离；原有 circular DBA 保持默认，供需要环形阵列误差分析的现有实验继续使用。公平验证配置显式设置 `scheduler.type: none`，避免 warm restart 学习率导致中后期指标回落后仍与论文表格做直接比较。预测窗口固定为 `num_pred: 1`；历史输入长度保持当前 image+GPS supervised 工作流的 `seq_len: 8`，因为 BeamBench 原文只明确 prediction window 为 1，没有足够依据在本 change 中同步修改历史窗口长度。

### Decision 10: 2604.05668 对齐口径使用 S32-34 合并 80/10/10 stratified split

为与 arXiv:2604.05668 的主实验表格比较，新增独立 supervised image+GPS 配置族，采用 scenes 32、33、34 的官方 train/test CSV 合并后的 labeled 样本作为候选集合，并在每个 scene 内按 `future_beam1` 标签做固定 seed 的 `80/10/10` stratified train/validation/test split。该口径将历史输入长度改为 `seq_len: 5`，预测窗口仍为 `num_pred: 1`，并继续使用 linear DBA。

该配置族不替代 BeamBench-fair 配置。BeamBench-fair 仍用于评估 S31 未见场景泛化；2604 对齐配置仅用于复现该论文主表中 S32/S33/S34 well-represented scenes 的宏平均 DBA。GPS 标准化必须只在 80% train 子集上拟合，并复用于 validation/test，避免合并全量数据后产生 normalization leakage。

## Risks / Trade-offs

- [风险] JEPA path 横跨 trainer、validator、objective metadata 和 model output，改动面比单个模型大。→ 通过 objective gate 限定分支，并用现有 supervised 测试保护默认行为。
- [风险] patch token encoder 与现有 ResNet-18 supervised encoder 不完全同构，预训练迁移收益可能受限。→ 先保存明确的 context encoder artifact，后续再通过独立 change 定义自动 fine-tuning adapter。
- [风险] GPS angle biased mask 可能引入错误几何先验。→ 默认保留 random mask 消融，并在 metadata 中记录 mask mode、seed、context/target ratio。
- [风险] EMA 在 AMP/grad scaler 下更新时机容易错误。→ 新增 `after_optimizer_step` hook，并用单元测试验证 target encoder 在 optimizer step 后变化、且不接收梯度。
- [风险] 无 beam label 训练可能暴露当前 batch/label 准备中的隐式假设。→ 在 `gps_conditioned_jepa` objective 下显式跳过 `prepare_task_labels` 和 beam metric 计算，验证 synthetic unlabeled batch。

## Migration Plan

1. 增加 objective metadata、history/TensorBoard 字段和 early-stopping alias，保证配置解析能识别 `gps_conditioned_jepa`。
2. 增加 JEPA model、loss/mask helper、训练 extension 和验证路径。
3. 增加 canonical smoke 配置与文档，默认输出到 `outputs/sceneXX/<run_name>/`。
4. 补齐单元测试和 smoke 测试，再运行相关快速检查。
5. 若需要回滚，删除新增配置并禁用 `gps_conditioned_jepa` objective；现有 supervised/adaptation 配置不依赖该 objective。

## Open Questions

- 第一版是否需要同时提供 ResNet feature-map token encoder，还是先以 patch embedding encoder 完成训练闭环。
- 下游 beam/fusion fine-tuning 是否在本 change 内自动加载 JEPA context encoder，还是作为后续独立 change 处理。
- GPS angle biased mask 的 grid-to-angle 映射需要按 DeepSense6G camera intrinsic 做更精确校准，还是先使用归一化图像平面近似。
