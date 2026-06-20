## Context

当前 `jepa-visual-architecture-sweep` 已经覆盖 patch 粒度、overlap tokenizer、conv stem、local token mixing、CNN tokens、多尺度 tokens、pooler/core ablation 和非 Transformer 对照。最近 follow-up 结果显示 `patch14_stage1_gps_query`、`overlap_k16_s8_stage1`、`overlap_k20_s10`、`resnet18_layer4_tokens` 和 `resnet18_layer3_layer4_tokens` 的性能接近，但这些结果仍混合了几个变量：

- patch 分辨率和 token 数。
- CNN 局部归纳偏置。
- JEPA Stage 1 checkpoint 是否重训。
- ResNet 是否 ImageNet 预训练、冻结或全量微调。
- pooler/core 是否改变。
- checkpoint selection 使用 primary、best_top1 还是 last。

`patch14_stage1_gps_query` 是一个轻量 patch-ViT-style JEPA tokenizer，并不是 TinyViT；它参数量远小于 ResNet18 token anchor。`resnet18_layer4_tokens` 只有 49 tokens，但 backbone 参数和计算量显著更大，不能把它的收益简单归因为 token 少或分辨率低。结合小场景数据可能不足以让浅层 ViT 学到 CNN 局部先验的担忧，本 change 需要提供一次覆盖 CNN、CNN+Transformer hybrid、JEPA Stage 1、ImageNet 预训练/冻结和 teacher-guided stabilization 的完整实验矩阵。

本设计仍复用现有训练、评估、diagnostics、runtime output 边界和 `loss.teacher_guidance` 机制。它不恢复旧 KD、HiST、Top8 selector、camera residual、GPS residual 或绕过 `src/kd_sensing` 的入口。

## Goals / Non-Goals

**Goals:**

- 定义并生成“全部候选”的实验矩阵，覆盖 CNN tokens、CNN+Transformer hybrid、patch/overlap 分辨率邻域、JEPA Stage 1、pooler/core ablation、预训练/冻结策略和 teacher-guided 候选。
- 为每个候选写出严格可比 metadata：variant family、token source、token count、token grid、image size、effective stride、checkpoint policy、pretraining source、freeze policy、teacher source、params、trainable params、compute proxy、split/seed/metric provenance 和 checkpoint selection。
- 生成 dependency-aware job manifest，使 Stage 1 pretraining、supervised downstream、teacher 训练/复用、best/best_top1/last re-evaluation 能被同一个 runner 顺序调度。
- 提供完整 bash runner，支持 GPU 0-3、最多 8 个进程并行、按 output-root scoped cleanup、安全 resume/skip completed runs。
- 汇总 Top-1/3/5、DBA、相邻 beam distance、params/compute Pareto、family best、checkpoint-policy best 和 strict claim eligibility。
- 所有运行产物默认写入 ignored `outputs/analysis/cnn_hybrid_jepa_visual_prior_sweep/`。

**Non-Goals:**

- 不在 proposal 阶段选择最终主线；本 change 只提供完整实验方案和可复现脚手架。
- 不引入 SAM、DETR、CLIP、大型 foundation vision backbone、Mamba 或新增重依赖作为第一版必跑项。
- 不恢复 retired `distillation` 配置或旧 teacher-student KD route；teacher 相关候选只能走当前 `loss.teacher_guidance`。
- 不提交 checkpoint、日志、cache、metrics CSV、TensorBoard 文件或生成 YAML 到源码之外的 runtime 产物。
- 不修改容器启动、认证或系统 profile 文件来实现长任务运行。

## Decisions

### Decision 1: 扩展现有 capability，而不是新建平行 sweep

本 change 修改 `jepa-visual-architecture-sweep` capability。新的全矩阵是现有 architecture sweep 的第二阶段扩展，继续使用同一 strict comparability metadata、diagnostics summary 和 output boundary。

原因是这次问题仍是“视觉架构/局部先验/JEPA reuse 哪个更好”，不是一个新的任务定义。沿用现有 capability 可以直接复用 `validate_sweep_manifest`、summary writer、architecture boundary tests 和之前的结果解释口径。

### Decision 2: 实验矩阵用声明式轴生成

源码中维护一个审计友好的 source manifest，例如 `configs/diagnostics/cnn_hybrid_jepa_visual_prior_sweep_manifest.yaml`。生成器从 manifest 展开 stage jobs、downstream jobs、teacher-guided jobs 和 re-eval jobs，生成到 runtime output root：

- `generated_configs/pretraining/*.yaml`
- `generated_configs/downstream/*.yaml`
- `generated_configs/eval/*.yaml`
- `jobs/stage1.tsv`
- `jobs/downstream.tsv`
- `jobs/teacher_guided.tsv`
- `jobs/reeval.tsv`
- `jobs/all.tsv`

每个 job MUST 记录 `variant_id`、`family`、`stage`、`depends_on`、`gpu_policy`、`max_parallel_group`、`command`、`output_dir`、`metrics_path` 和 `resume_policy`。

### Decision 3: 全量候选族和轴

第一版 full mode MUST 覆盖以下候选族。若某个候选需要新增组件但 apply 阶段尚未实现，生成器 MUST 在 manifest 中保留候选定义并把 `availability` 标为 `requires_component`，不能静默删除。

| family | required axes | purpose |
| --- | --- | --- |
| existing_controls | `gps_only_control`、`patch16_mean_baseline`、`patch16_gps_query_pool`、`patch14_stage1_gps_query`、`overlap_k16_s8_stage1`、`overlap_k20_s10`、`resnet18_layer4_tokens`、`resnet18_layer3_layer4_tokens` | 锚定已有结果 |
| patch_resolution_stage1 | patch size 16、14、12、10、8；image size 默认 224；pooler 默认 GPS-query K2 frame | 判断继续提高 patch 分辨率是否有效 |
| overlap_stage1 | kernel/stride：12/6、14/7、16/8、20/10、24/12；pooler 默认 GPS-query K2 frame | 判断 overlap 是负作用还是只是不在最佳邻域 |
| cnn_supervised_tokens | backbone：ResNet18、ResNet34；stage：layer3、layer4、layer3+layer4；pretrained：scratch、ImageNet；freeze：full_ft、freeze_backbone_projection、unfreeze_layer4 | 分离 CNN 局部先验、参数量和 ImageNet 预训练 |
| cnn_jepa_tokens | backbone：ResNet18、ResNet34；stage：layer4、layer3+layer4；pretrained：scratch、ImageNet；Stage 1 + downstream | 测试 CNN token source 是否能作为 JEPA visual encoder |
| hybrid_tokenizers | conv_stem_s16、conv_stem_s8、local_patch16、local_patch14、cvt_patch16、cvt_patch14、conv_stem_patch14、conv_stem_patch12 | 测试 CNN local stem/token mixing + Transformer 全局建模 |
| pooler_core_ablation | mean、gps_query_k1_frame、gps_query_k2_frame、gps_query_k4_frame、gps_query_k2_tokens、gps_query_k4_tokens、hybrid_residual_query、token_aware_core | 判断瓶颈是否在 pooling/fusion |
| teacher_guided_stabilization | teachers：resnet18_l4_scratch、resnet18_l4_imagenet_unfreeze_layer4、resnet34_l4_imagenet；students：patch14、patch12、overlap_k16_s8、conv_stem_s16、local_patch14；temperature：2、4；weight：0.1、0.3、0.5 | 用当前 `loss.teacher_guidance` 检验 CNN teacher 是否能稳定 patch/hybrid student |
| compute_controls | matched_trainable_params、matched_token_count、frozen_backbone_controls、random_feature_controls | 避免把“大模型/预训练”误读为架构收益 |
| seed_confirm | seeds 17、23、42；作用于 full mode 中所有 primary 候选，至少作用于每个 family top candidates | 估计小数据方差 |

“不管优先级，跑全部”在本 change 中定义为：full mode 默认展开上述所有 declared axes，并生成完整 job manifest。实现允许用户用 `--families` 或 `--dry-run` 缩小范围，但默认方案必须是全量。

### Decision 4: Stage 与 checkpoint policy 显式化

每个候选 MUST 声明：

- `stage_plan`: `supervised_only`、`stage1_then_downstream`、`teacher_then_student`、`reeval_only` 或组合。
- `checkpoint_policy`: `exact_reuse`、`partial_reuse`、`pos_interpolate`、`fresh_stage1_required`、`supervised_only_anchor`、`teacher_guided_student`。
- `checkpoint_selection`: `primary`、`best_top1`、`last`，其中 full summary MUST 同时保留 primary 和 best_top1 re-evaluation。

Stage 1 candidates MUST 先生成 pretraining job，并把 downstream job 的 `depends_on` 指向对应 checkpoint placeholder 或已存在 checkpoint。Supervised anchors MUST 不伪装成 JEPA checkpoint reuse。Teacher-guided candidates MUST 使用 `loss.teacher_guidance`，并记录 teacher checkpoint/logits/probability source、temperature、weight 和 stop-gradient policy。

### Decision 5: 可比性和 claim gate 分两层

Full run 可以包含 smoke、failed、skipped、unavailable 和 strict rows，但最终推荐判断只能读取 `strict_comparable: true` 且满足以下条件的行：

- split、scene set、seed、history window、GPS source window、prediction horizon、beam label space、metric profile、normalization artifact、difficulty digest 一致。
- checkpoint selection 明确。
- Stage 1/downstream/teacher provenance 完整。
- 非 smoke/lowmem-only。
- metrics 文件存在且包含主指标。

Summary MUST 同时输出“全量结果表”和“strict ranking”。全量表用于排查，strict ranking 用于架构判断。

### Decision 6: Runner 只做项目级作业编排

新增 shell runner 应该是项目脚本或生成脚本，例如 `outputs/analysis/.../run_full_sweep.sh` 或 source-managed thin script + runtime manifest。它 MUST：

- 使用 `conda run -n kd_mm_beam`。
- 默认 GPU 列表为 `0,1,2,3`。
- 默认最多 8 个项目进程同时运行。
- 尊重 job dependencies。
- 将 stdout/stderr 写入 output root 下的 logs。
- 支持 output-root scoped cleanup。
- 不修改 `/root/.container_env`、profile、SSH、systemd 或系统启动项。

## Risks / Trade-offs

- [Risk] 全量矩阵训练成本很高。→ Mitigation：full mode 仍生成全部 job；runner 支持 resume、skip completed、failed-only retry 和 dry-run manifest，用户可随时查看剩余作业。
- [Risk] ImageNet 预训练下载或缓存不可用。→ Mitigation：manifest 记录 `pretraining_source` 和 `availability`；不可下载时标为 `skipped_unavailable`，不污染 strict ranking。
- [Risk] 旧 KD 路线被误恢复。→ Mitigation：spec 和 tests 明确禁止 retired `distillation` key；只允许 `loss.teacher_guidance`。
- [Risk] 高 token 数候选显存爆炸。→ Mitigation：每个候选声明 `token_budget`、batch size、AMP、gradient accumulation 和 fail-fast policy。
- [Risk] CNN anchor 的提升来自参数量或预训练，而不是局部先验。→ Mitigation：加入 compute_controls、matched params/token count/freeze variants，并在 summary 中输出 Pareto。
- [Risk] 小数据方差导致单 seed 结论不稳。→ Mitigation：full mode 包含 seeds 17、23、42；summary 按 mean/std 和 per-seed best 同时报告。
- [Risk] Stage 1 checkpoint 与 downstream checkpoint selection 混乱。→ Mitigation：`stage_plan`、`checkpoint_policy`、`checkpoint_selection` 和 `depends_on` 必须进入每行 metadata。

## Migration Plan

1. 新增 full sweep manifest schema 和 validator，复用并扩展现有 `jepa_visual_architecture_sweep` strict comparability 逻辑。
2. 新增 config/job generator，从 source manifest 展开全量 candidates、runtime configs、job TSV 和 run script。
3. 补齐必要模型 opt-in 组件：缺失的 patch sizes、overlap 邻域、hybrid tokenizers、CNN JEPA token source、pooler/core variants 和 teacher-guidance config adapters。
4. 新增 runner 与 summary analyzer，支持 dependency-aware full sweep、resume、cleanup、metrics aggregation、Pareto 和 claim gate。
5. 新增 focused tests：manifest schema、config generation、retired KD guard、job dependencies、runner dry-run、summary/claim gate。
6. 运行 OpenSpec 校验、配置加载 smoke、模型 forward smoke、架构边界测试和 summary 单元测试。

Rollback 方式：删除新增 manifest/generator/runner/summary/tests 和本 change artifacts；runtime output root 可直接删除。现有 `patch14_stage1_gps_query`、`overlap_k16_s8_stage1`、`resnet18_layer4_tokens` 等结果目录不需要迁移。

## Open Questions

- full mode 是否把 seeds 17、23、42 应用于所有候选，还是先全候选 seed17 再对 top candidates 扩 seeds；本 proposal 默认按用户要求全跑，因此全部候选都展开三 seed。
- ImageNet pretrained ResNet 是否允许首次运行自动下载权重；若运行环境无网络，生成器应把相关 job 标记为 unavailable 或要求本地 cache。
- `resnet50`、ConvNeXt-Tiny、MobileNetV3 等更大外部 backbone 是否进入第二轮 change；本 change 第一版只把 ResNet18/34 作为必跑 CNN backbone，以控制依赖和解释复杂度。
