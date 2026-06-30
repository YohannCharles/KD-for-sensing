## Context

项目当前的模型扩展默认走 `modular_sequence` 组件路径；只有 active OpenSpec artifact 明确说明原因时，才允许新增完整 `MODELS.register(...)`。U-MaskBeamJEPA 需要在同一次 forward 内同时构建 full-modal teacher latent、available-set context、Gaussian JEPA predictor、reliability-gated fusion 和 student logits，输出字段也要被 opt-in 损失消费，因此第一版登记为 whole-model exception，而不是把多个互相耦合的阶段硬塞进单个 representation core。

现有 runtime 已提供 `prepare_task_batch`、`run_model_step`、`adapt_model_output`、difficulty operator、training extension 和 prediction loss 钩子。新方案必须复用这些路径：模型输出以 dict 返回，`logits` 被 `ModelOutput` 识别，其余 `teacher_logits`、`u_star`、`mu_B`、`logvar_B`、`modality_mu_B`、`modality_logvar_B`、`modality_reliability`、`global_reliability` 进入 diagnostics，供 U-MaskBeamJEPA loss extension 读取。

项目模态契约使用 canonical 名称 `image`、`radar`、`gps`、`lidar`、`mmwave`、`csi`。用户 prompt 中的 `vision` 在本项目中落为 `image`；第一版默认模态为 `["image", "radar", "lidar", "gps"]`，不新增伪模态或 `vision` alias。

## Goals / Non-Goals

**Goals:**

- 新增可 registry 构建的 `u_mask_beam_jepa` 模型，支持 image/radar/lidar/gps 四模态 latent 聚合和缺失模态 mask。
- 新增最小 loss 接入：beam CE、teacher CE、global Gaussian latent NLL 和 per-modality Gaussian NLL，并暴露 loss/reliability/global reliability 诊断。
- 新增训练 missing mask 采样 helper 和 pattern mask helper，支持 p_missing float 或 per-modality list，保证每个样本至少一个可用模态。
- 新增 smoke 配置和 ablation 配置，覆盖 teacher、JEPA loss、modality uncertainty、global uncertainty、fusion type、context type。
- 保持普通 supervised baseline 不需要新增 metadata 字段或模型专用训练循环分支。

**Non-Goals:**

- 不生成原始 RGB/LiDAR/radar，也不预测每个缺失模态的完整 latent 集合。
- 不引入 diffusion、flow matching、evidential uncertainty 或新的 KD/distiller registry。
- 不新增根目录 `train_u_mask_beam_jepa.py`、`eval_missing_patterns.py` 或旧式顶层 `models/`、`losses/`、`data/` 目录。
- 不把 `vision` 注册成新模态；文档和配置使用 canonical `image`。
- 不把 missing/corruption 测试绑定到真实 `dataset/`。

## Decisions

1. **实现路径：whole-model exception，而不是普通 component baseline。**
   - 方案：新增 `src/kd_sensing/models/u_mask_beam_jepa.py`，注册 `u_mask_beam_jepa`，并加入 `import_default_components()`。
   - 原因：teacher、predictor、fusion 和 loss diagnostics 需要共享同一组 latent 与 mask；拆成独立 core/head 会迫使训练侧绕路传中间字段。
   - 替代方案：只新增 `REPRESENTATION_CORES`。放弃原因是 teacher logits、Gaussian target 和 ablation 输出契约会污染普通 `modular_sequence`。

2. **复用现有 encoder/projector 思路，模型内部只接收或构建 d_model latent。**
   - 方案：模型构造支持 `encoders`/`projectors` 配置，优先复用现有 registry 组件；若第一版 smoke 只给 synthetic latent，可支持 `latent_inputs` 测试路径。
   - 原因：用户 prompt 假设 `encoder_i(x_i) -> z_i`，项目实际需要通过 canonical batch key 与 encoder registry 取得 latent。
   - 替代方案：手写四个 raw modality encoder。放弃原因是重复现有组件且会扩大配置面。

3. **mask 来源：训练采样走 helper，评估 pattern 走显式 mask，difficulty metadata 作为兼容输入。**
   - 方案：新增 `src/kd_sensing/data/missing_mask.py`，提供 `sample_missing_mask`、`make_pattern_mask` 和轻量 `apply_modality_corruption`；训练 extension 在 `before_forward` 注入 `missing_mask`，eval 可通过配置指定 available pattern。
   - 原因：现有 `ModalityMissingOperator` 是 zero-fill corruption；U-MaskBeamJEPA 需要模型级 available-set mask。二者语义相邻但不应互相替代。
   - 替代方案：直接复用 difficulty operator 的 per-field valid mask。放弃原因是它服务输入损坏/zero-fill，不能稳定表达四模态 set mask 和 ablation pattern。

4. **loss 接入：使用 training extension / prediction loss 扩展，不新增 distiller。**
   - 方案：新增 `src/kd_sensing/losses/u_mask_beam_jepa.py`，从 `ModelOutput.diagnostics` 读取 teacher/latent 字段，返回 tensor components 与 scalar diagnostics；训练配置 opt-in 启用。
   - 原因：现有 registry 已移除 KD，teacher CE 在这里是同模型 full-modal auxiliary supervision，不恢复 distillation registry。
   - 替代方案：注册一个 `LOSSES` 名称替换 base CE。放弃原因是当前训练循环以 beam base loss + prediction/extension loss 为主，替换 base loss 会增加分支。

5. **fusion/ablation 第一版只实现必要分支。**
   - 方案：`reliability_gated_cross_attention` 为主路径；`concat_mlp` 和 `weighted_sum` 做最小可用对照。`context_type` 第一版支持 `set_transformer_simplified` 和等价 beam-query Transformer，`mask_transformer` 作为配置拒绝或后续任务。
   - 原因：ablation 需要能跑，但不需要为未验证路径写大而全框架。
   - 替代方案：一次性实现所有上下文变体。放弃原因是测试面和维护面过大。

6. **输出契约保持 `adapt_model_output` 兼容。**
   - 方案：模型 forward 返回 dict，`logits` 为 `[B, K]` 或现有 runtime 可接受的 `[B, 1, K]`，`output_features` 存 fused latent，诊断字段保留 tensors；`u_star` 在输出前 detach。
   - 原因：训练、验证、评估已经统一通过 `ModelOutput` 消费输出。
   - 替代方案：返回自定义 dataclass。放弃原因是会绕过现有 adapter。

7. **真实训练性能：LMDB 样本 cache 与 validation 降频都保持 opt-in。**
   - 方案：新增 `deepsense6g_sample_lmdb_cache` 预处理器生成 split-level LMDB；训练侧通过 `data.dataset.sample_cache` 显式读取。新增 `training.validation.interval_epochs`，第 1 个、间隔命中和最后 1 个 epoch 运行 validation。
   - 原因：U-MaskBeamJEPA 四模态训练的数据瓶颈来自大量小文件读取和每 epoch 全量 validation；这两个开关能减少 I/O 和评估时间，不改变默认训练语义。
   - 替代方案：把所有样本塞进内存或默认开启 cache。放弃原因是会扩大内存峰值，并让普通训练依赖本地缓存状态。

## Risks / Trade-offs

- **Risk: 全 0 mask 导致 attention 无有效 key/value。** → 在 mask helper 和模型 forward 双重检查；默认保证至少一个模态可用，否则抛出包含样本索引的错误。
- **Risk: 新 whole-model exception 扩大 registry 表面。** → 限定为一个注册名、一个模型文件、focused tests 和架构摘要 metadata；不新增别名。
- **Risk: teacher CE 被误解为恢复 legacy KD。** → 配置和 metadata 标记为 `same_model_full_modal_teacher_auxiliary`，不使用 `logits_kd`、`rkd` 或 distiller registry。
- **Risk: `vision` 与项目 `image` 命名冲突。** → 所有 artifact、配置和测试使用 `image`；只在注释中说明用户 prompt 的 vision 对应 image。
- **Risk: raw modality shape 差异导致第一版实现过大。** → 第一版 smoke 用 synthetic tensors 和现有 encoder registry；真实数据接入只通过已有 batch/runtime，不写 dataset-specific 分支。
- **Risk: reliability bias 数值不稳定。** → clamp loss 侧 `logvar`，以 per-modality `logvar_i_B` 的 softplus 均值计算 `[B, M, 1]` reliability，attention bias 使用 `log(r + eps)`，并在测试覆盖 finite 输出和 backward。
- **Risk: LMDB 依赖在现有环境缺失。** → `lmdb` 作为 opt-in extra；未安装时只在启用 sample cache/预处理器时给出安装提示，普通训练不受影响。

## Migration Plan

1. 新增模型、loss、mask helper 和最小配置，不修改现有默认 config。
2. 将 `u_mask_beam_jepa` 加入默认组件导入和模型架构摘要覆盖。
3. 用 synthetic focused tests 验证 registry build、forward、loss backward、mask helper、ablation 开关和 metadata。
4. 需要真实训练时，通过现有 `kd-sensing-train --config configs/fusion/u_mask_beam_jepa_s32.yaml` 运行 Scenario 32 opt-in 配置；输出仍写 ignored `outputs/`。scene 31 smoke config 保留为快速验证。
5. 回滚时删除新增模型/loss/helper/config/test 和 registry import 行；现有模型与训练入口不受影响。

## Open Questions

- 第一版真实训练默认用哪些现有 encoder 组合：当前建议先用项目已有 image/radar/lidar/gps lightweight encoder 或 synthetic smoke 配置，等具体数据集与显存预算确定后再调参。
- `mask_transformer` 是否需要立即实现：当前建议第一版配置拒绝该值，等 set-transformer 简化版跑通后再加。
- eval missing pattern 的 CLI 表达是否复用 difficulty profile 还是新增小字段：当前建议先用配置字段 `evaluation.missing_pattern.available_modalities`，少动 CLI。
