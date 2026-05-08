## Context

项目当前的多模态融合已经具备较好的基础设施：`craf_fusion` 使用统一模态顺序、`encoders`/`feature_projections`/tokenizer、`force_modality_mask`、teacher registry 加载与 encoder 冻结、beam soft loss、prior regularization、TensorBoard diagnostics 和 subset validation。现有强 baseline 是 `scene32_teacher_init_no_prior_ablation`，说明 teacher encoder 初始化和 token fusion 本身有效；固定 prior 或 prior residual scalar gate 更偏解释性，clean all-modal Top-1 没有稳定拉开差距。

`模态失衡改进方案4.md` 提出的 MARF 方向是合理的，但需要按项目实际做三点调整：

- 文件位置使用包内路径：`src/kd_sensing/models/fusion/`、`src/kd_sensing/distillation/`、`src/kd_sensing/engine/`，而不是仓库根目录的 `models/`、`losses/`。
- 模型 forward 保持现有 fusion 调用契约：接收 `image_batch/radar_batch/gps_batch/lidar_batch/mmwave_batch` 和可选 `force_modality_mask`，而不是改为 `features` dict，避免重写 batch 准备、trainer、validator。
- 预测长度按现有标签语义实现为 `horizon = num_pred + 1`。Scene32 常用 `num_pred: 3` 时实际 logits 为 `[B, 4, 64]`，包含当前最后 beam slot 和 3 个未来 slot。

## Goals / Non-Goals

**Goals:**
- 新增可通过 `model.student.type: marf_fusion` 构建的 MARF fusion student，支持 image/radar/gps/lidar/mmWave 任意有效组合。
- Router 必须 sample-wise + horizon-wise，输出 `anchor_weights [B,H,K]` 和 `residual_weights [B,H,K]`，不可用模态必须被 mask 到 0。
- Prior 只能作为 teacher registry 驱动的弱 bias，支持关闭和缩放；不得把 GPS/mmWave 写死为主模态。
- 复用 teacher registry encoder loading/freeze 路径，MARF 主训练默认冻结单模态 teacher encoders，只训练 router/fusion/adapter/head。
- 训练支持 subset-aware loss：all-modal CE、subset CE、all-to-subset KD、beam soft、residual norm、prior regularization 和可选 anchor entropy。
- 验证和脚本能证明 `val/subset/all/top1` 与官方 `accuracy/val` 一致，并能评估 top-prior、random/top-prior、single-best、weak/low-prior 和扰动表现。

**Non-Goals:**
- 不实现全量 encoder fine-tune 作为默认路径；Stage 3 selective fine-tune 只保留为后续 ablation。
- 不改变 CRAF、token transformer、legacy fusion 配置的默认行为。
- 不引入新的外部深度学习依赖；使用现有 PyTorch、registry、trainer 和 validator。
- 不默认使用 focal gamma=2 作为 MARF 主任务 loss；MARF 配置默认使用 `cross_entropy` 和小 label smoothing。

## Decisions

### 1. 以 `MARFFusionNet` 作为新的 fusion student 注册名

实现放在 `src/kd_sensing/models/fusion/marf.py`，注册为 `marf_fusion`，并在 `src/kd_sensing/models/fusion/__init__.py` 与 `src/kd_sensing/models/__init__.py` 导出。模型暴露：

- `supports_force_modality_mask = True`
- `supports_marf_routing = True`
- `encoders: nn.ModuleDict`
- `set_reliability_prior(priors)` 兼容现有 `apply_teacher_priors()`，内部转发给 router prior buffer

备选方案是修改 `CRAFTTokenFusionBase` 或让 MARF 继承 CRAF。该方案会把 scalar reliability gate 与 MARF anchor/residual 路由绑在一起，训练分支和 diagnostics 容易混淆；因此采用独立模型类，并复用/抽取 CRAF 的 encoder、mask、confidence helper。

### 2. Router、AnchorFusion、ResidualAdapter 作为模型内部模块，但保持可单测

新增模块类：

- `ModalityRouter`: 输入 `modality_summary [B,K,D]`、`confidence [B,K,2]`、`modality_mask [B,K]` 和 prior buffer；输出 anchor/residual logits 与 weights。
- `AnchorFusion`: 使用每个 horizon 一个 learnable query，对按 anchor 权重缩放后的 `[B,K,T,D]` token 做 cross-attention，输出 `h_anchor [B,H,D]`。
- `ResidualAdapter`: 用共享 cross-attention 和模态 embedding 从每个模态 token 提取 `residual_delta [B,H,K,D]`，按 residual weights 汇总得到 `h_final`。

备选方案是继续使用现有 Transformer self-attention 并只改 gate。这个方案实现更小，但仍是模态级 scalar gate，不能表达“某个 horizon 由 A 主导、另一个 horizon 由 B 主导”的需求。

### 3. `horizon = num_pred + 1`，默认 `d_model = feature_size`

项目当前 `prepare_labels()` 会拼接当前最后 beam 和未来 `num_pred` 个 beam，因此 MARF 预测头必须输出 `num_pred + 1` 个 slot。配置允许 `d_model` 与 `feature_size` 不同，但 Scene32 初始配置应使用 `feature_size: 64`、`d_model: 64`，保证 teacher encoder 权重加载和显存风险可控；若后续提升 `d_model`，通过 `feature_projections` 做投影。

备选方案是按方案文档写死 `H=3,D=256`。这会与现有训练标签、测试和 checkpoint loading 不一致，不适合作为本仓库第一版。

### 4. Prior 是 router bias，不是固定规则

Router 保存 `[K]` prior buffer，`set_reliability_prior()` 从 teacher registry 更新它。启用 prior bias 时：

- `anchor_logits = anchor_delta_logits + prior_anchor_scale * logit(prior)`
- `residual_logits = residual_delta_logits + prior_residual_scale * logit(prior)`

`use_prior_bias: false` 时完全由样本特征决定。prior regularization 只约束 `anchor_weights.mean(dim=(0,1))` 与 prior 的距离，权重默认小于现有 CRAF prior regularization，避免退化为 fixed prior。

备选方案是保留 `fixed_prior` gate。它能快速体现强模态假设，但不能支持不同数据集优势模态变化，也违背“不写死强弱模态”的目标。

### 5. Subset-aware training 放在 trainer 层，不放进模型 forward

新增 `src/kd_sensing/engine/marf_training.py`，提供 prior-driven `ModalitySubsetSampler` 和 subset loss helper。Trainer 在主 forward 后，如果模型 `supports_marf_routing` 且 `training.subset_training.enabled`，执行若干次 subset forward：

- `all` forward 参与主 distiller/task loss。
- subset forward 使用 `force_modality_mask`。
- subset CE 使用同一 `task_criterion`。
- subset KD 使用 all logits 的 detach soft target。

备选方案是让模型 forward 内部循环所有 subset。这样会让模型知道训练策略，难以在 validator、debug 脚本和 ablation 里复用，也会破坏现有 `adapt_model_output()` 简洁契约。

### 6. Validation subsets 改为 prior-driven，并保留兼容名称

当前 `_modality_subset_definitions()` 写死 `strong = gps+mmwave`、`weak = image+radar+lidar`。MARF 需要从 teacher prior 或模型 prior 获取排序：

- `top_prior`: prior 最高的 top-k 模态。
- `single_best_prior`: prior 最高的单模态。
- `low_prior_only`/兼容 `weak_only`: prior 最低的一组模态，不再写死具体名称。
- `all`: 使用全部启用模态。

`all` subset 的 Top-1 必须与官方 validation Top-1 一致；若出现差异，`scripts/debug_eval_consistency.py` 要输出两个路径的 logits/labels 形状、样本数、batch 数和首批预测一致性。

### 7. 诊断输出沿用 dict diagnostics

MARF forward 返回至少：

- `logits`
- `token_features`
- `output_features`
- `anchor_logits/anchor_weights`
- `residual_logits/residual_weights`
- `h_anchor/h_final`
- `residual_delta`
- `prior`
- `effective_modality_mask`
- `unimodal_logits/confidence`
- `modalities`

Trainer 写入 `marf/anchor_mean/<modality>`、`marf/residual_mean/<modality>`、`marf/anchor_h<h>/<modality>`、subset loss 和 MARF extra loss。CRAF diagnostics 保持原路径，避免旧测试和已有 TensorBoard 标量语义变化。

## Risks / Trade-offs

- Router 过早塌缩到高 prior 模态 -> 使用小 prior regularization、可关闭 prior bias、anchor entropy 默认 0，并先跑 no-prior-bias ablation。
- Residual adapter 污染 anchor 表示 -> residual scale 默认 0.2，增加 residual norm loss，提供 no-residual ablation。
- 训练开销增加 -> subset forward 默认每 batch 2 个，先支持配置关闭；smoke test 用 synthetic 和小 batch。
- `all` subset 与官方 validation 不一致 -> 先实现 debug consistency 脚本并纳入测试，所有结论必须在一致性通过后再看。
- 复用 CRAF private helper 可能增加维护成本 -> 第一版可抽取 `fusion/common.py`；若实现阶段只需少量 helper，可先局部复用并补测试约束。
- Label smoothing 配置依赖当前 PyTorch `CrossEntropyLoss` 参数 -> 配置加载和 smoke test 必须覆盖 `type: cross_entropy`、`label_smoothing: 0.03`。

## Migration Plan

1. 保留现有 CRAF 和 teacher-prior 配置不变，新增 `marf_fusion` 和 MARF 配置，不替换任何默认入口。
2. 先实现模型 forward、mask、prior bias 和 diagnostics，跑单元 shape/mask/softmax 测试。
3. 接入 teacher registry loading/freeze，跑已有 teacher-prior 相关测试和 MARF encoder freeze 测试。
4. 接入 subset-aware training 和 prior-driven validation，先跑 synthetic smoke，再跑 `debug_eval_consistency.py`。
5. 再运行 Scene32 主实验和 ablation：no subset、subset training、no residual、no prior bias。
6. 回滚策略是删除 `marf_fusion` 配置入口或将实验配置切回现有 `scene32_teacher_init_no_prior_ablation`；旧模型和旧配置不受影响。

## Open Questions

- `top_prior_k` 默认取 2 还是按可用模态数比例取值；第一版按配置默认 2。
- `weak_only` 名称是否保留为兼容别名；第一版保留，但语义改为 low-prior subset，并在日志中记录实际模态列表。
- `d_model` 是否在正式 Scene32 实验中升到 128/256；第一版以 64 对齐现有 teacher encoder 与资源约束，后续作为独立 ablation。
