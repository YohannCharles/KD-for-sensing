## Context

仓库当前模型层已经具备稳定边界：`kd_sensing.registries` 提供轻量 registry，`modular_sequence` 组合 encoder/projector/core/head，`training_strategy_metadata()` 记录训练策略，训练启动阶段通过 `module_trainability_report()` 写出粗粒度参数统计。近期 TinyViT、CNN/hybrid JEPA sweep 和 scene-conditioned meta-offset 同时增加候选，使“我现在到底用了哪个 encoder、总模型多少参数、视觉/context encoder 多少参数、哪些参数冻结、哪些参数只是估算”变成日常决策问题。

现状的统计来源有三类，但没有统一口径：

- 训练 startup summary：基于真实 `nn.Module`，可统计 total/trainable 参数，但角色分组较粗，主要服务训练健康检查。
- 模型 metadata：ResNet/TinyViT/JEPA 等组件各自暴露部分 freeze、checkpoint、token metadata，但字段名、层级和参数量口径不统一。
- JEPA visual sweep manifest：已有 params/compute 估算器，适合候选矩阵和 Pareto，但它不是从真实模型实例递归统计，也没有被训练 startup summary 复用。

用户特别指出的基准口径必须被第一版覆盖：

| 模型 | total params | image encoder params | visual/context encoder params |
| --- | ---: | ---: | ---: |
| `patch14_stage1_gps_query` | 0.197M | 0.117M | 0.088M |
| `resnet18_layer4_tokens` | 11.32M | 11.24M | 11.21M |
| `resnet18_layer3_layer4_tokens` | 14.13M | 14.05M | 14.02M |

这说明 summary 不能只列 registry 名称，也不能只 build 当前 YAML；它必须同时服务真实实例、sweep 候选、run artifact 和设计期比较。

## Goals / Non-Goals

**Goals:**

- 新增统一模型架构与参数摘要 schema，覆盖真实模型实例、`modular_sequence` 子组件、whole-model exception 和 sweep 候选。
- 把现有 startup 参数统计、模型 `training_strategy_metadata()` 和 JEPA sweep params/compute metadata 收敛到同一输出语义。
- 支持用户用单条命令或 Python helper 查看当前配置的整体模型、各模态 encoder、视觉/context encoder、core/head 参数量和关键 metadata。
- 支持 dry-run/preflight 发现配置不兼容，例如 TinyViT 复用 ResNet 的 `unfreeze_stages: [layer4]`、潜在下载 22k checkpoint、unused classifier head 参数等。
- 保持 registry 和模型构造边界不变；summary 是观测层，不是第二套构造层。
- 默认不读取真实 `dataset/`，不训练，不下载 checkpoint，不提交任何运行产物。

**Non-Goals:**

- 不重构 `Registry`、不合并 `ENCODERS/MODELS/PROJECTORS/...`，不改变 registry 名称或默认组件导入策略。
- 不把所有模型强制迁移到 `modular_sequence`；whole-model exception 只需提供可摘要 metadata。
- 不实现精确 FLOPs/MACs 计算；首版只提供 compute proxy 和 token/attention proxy，真实 FLOPs 可作为后续扩展。
- 不把 TinyViT、JEPA 或 ResNet 候选设为新的默认模型；summary 只揭示结构和参数，不改变训练选择。
- 不为真实数据训练或 sweep 自动启动任务；summary CLI 只做配置/manifest/run artifact 解析和可选模型 build。

## Decisions

### Decision 1: 新增观测层，而不是重构 registry

实现一个窄模块，例如 `src/kd_sensing/models/architecture_summary.py`，提供：

- `summarize_model_architecture(model, *, cfg=None, source=None, include_named_parameters=False)`
- `summarize_model_config(model_cfg, *, build=True, allow_download=False, overrides=None)`
- `summarize_sweep_candidate(record)` 或 sweep 内部 adapter
- `render_architecture_summary(summary, format="markdown|json|csv")`

Rationale: registry 已经负责按名称构建组件，现有契约强调 registry 轻量导入和 canonical 名称边界。把参数/架构观察逻辑塞进 registry 会扩大导入面，也会混淆“可构造组件列表”和“某个配置实例的实际结构”。

Alternative: 在每个 encoder/model 自己实现完整参数摘要。这样短期直观，但会让 ResNet、TinyViT、JEPA、BEV、Vision-Position 等重复代码，并且难以保证字段一致。

### Decision 2: 实例级统计是事实源，候选级统计是声明源

summary 必须区分两种来源：

- `source.kind: instance`：从真实 `nn.Module` 的 `named_parameters()`、`named_modules()` 和 `requires_grad` 统计得到，作为“当前构建出来的模型”的事实源。
- `source.kind: candidate`：从 sweep manifest 或配置生成器 metadata 得到，作为“尚未 build 或不适合 build 的候选”的声明源。

两者使用同一字段名，但必须记录 `parameter_count_source`：

- `actual_module`
- `declared_candidate_metadata`
- `startup_summary_artifact`
- `mixed_actual_and_declared`

Rationale: `patch14_stage1_gps_query` 和 full sweep 候选需要在训练前进入矩阵；TinyViT/ResNet 当前配置需要真实 build 后看冻结和未使用参数。混在一个无来源字段里会造成误读。

### Decision 3: 输出 schema 以参数 role 和语义 role 双轴组织

顶层 schema 建议：

```yaml
schema_version: 1
source:
  kind: instance | candidate | artifact
  config_path: ...
model:
  registry_type: modular_sequence
  class: ModularSequenceModel
  architecture_category: component_baseline
  enabled_modalities: [image, gps]
parameters:
  total_params: 197000
  trainable_params: 109000
  frozen_params: 88000
  effective_params: 197000
  excluded_params: 0
components:
  encoders.image:
    registry_type: jepa_context_image
    semantic_role: image_encoder
    parameter_role: image_encoder
    total_params: 117000
    visual_context_encoder_params: 88000
    trainable_params: ...
    metadata: {...}
  representation_core:
    ...
warnings: []
comparability:
  token_count: ...
  compute_proxy: ...
  checkpoint_policy: ...
```

字段说明：

- `total_params`: module 里所有去重参数数量。
- `trainable_params`: `requires_grad=True` 参数数量。
- `frozen_params`: `requires_grad=False` 参数数量。
- `effective_params`: 排除明确未参与 downstream forward 的语义排除参数后的参数数量。
- `excluded_params`: 语义排除参数总量，并在 `excluded_parameter_groups` 记录原因。
- `components.*.semantic_role`: `image_encoder`、`visual_context_encoder`、`projector`、`representation_core`、`head`、`auxiliary`、`unused_downstream_head` 等。
- `components.*.parameter_role`: 方便表格按 image encoder params、visual/context encoder params、core/head params 聚合。

Rationale: 用户关心的不只是 PyTorch 参数总数，还关心“image encoder params”和“visual/context encoder params”这种架构语义口径。TinyViT unused ImageNet head 也需要 `total_params` 与 `effective_params` 同时可见。

### Decision 4: 模块角色优先从结构约定推断，再用组件 metadata 补充

实例级统计的 role 推断顺序：

1. `modular_sequence` 明确属性：`encoders.<modality>`、`projectors.<modality>`、`representation_core`、`heads.<name>`、`geometry_prior`、`reranker`。
2. 组件 `training_strategy_metadata()`：`registry_type`、`variant`、`backbone_dim`、`token_count`、`checkpoint_policy`、`freeze_policy`、`consumes_reliability_metadata`。
3. 已知模型属性：`context_encoder`、`visual_encoder`、`backbone`、`projection`、`pooler`、`adapter`、`beam_head`。
4. fallback：按 top-level child module 分组，标记 `semantic_role: unknown_component`，不影响总参数。

Rationale: 先利用项目现有稳定结构，避免让每个模型必须手写全量 summary；同时保留 fallback 覆盖 whole-model exception。

### Decision 5: warning 是 summary 的一等输出

首版必须输出 warnings 数组，最少覆盖：

- `incompatible_encoder_option`: 例如 TinyViT 配置继承 ResNet `unfreeze_stages: [layer4]`。
- `potential_checkpoint_download`: 例如 22k TinyViT 未提供 `checkpoint_path` 且 `allow_download=true`。
- `unused_parameter_group`: 例如 downstream encoder 不使用的 ImageNet classifier head 仍存在于 module tree。
- `declared_vs_actual_param_mismatch`: manifest 声明参数量与实际 build 统计差异超过阈值。
- `unknown_component_role`: 无法映射语义角色但仍纳入总参数的模块。

Rationale: 这个工具的价值不只是报数，更是提前暴露结构调整时的隐性错误。

### Decision 6: CLI 是薄入口，默认 dry-run/无副作用

新增包内 CLI 可命名为 `kd-sensing-model-architecture-summary` 或更短 `kd-sensing-model-summary`。默认行为：

```bash
conda run -n kd_mm_beam kd-sensing-model-summary \
  --config configs/image/supervised.yaml \
  -o model.primary.encoders.image.type=tinyvit_5m_scratch_rgb \
  -o model.primary.encoders.image.unfreeze_stages=[] \
  --format markdown
```

支持输入：

- `--config <yaml>` + `-o key=value`
- `--model-config-json <json>` 用于测试或外部调用
- `--sweep-manifest <yaml/json>` + `--variant-id <id|all>`
- `--startup-summary <path>` 只读既有 run artifact

默认输出 stdout；用户指定 `--output` 时只写入显式路径，文档推荐 ignored `outputs/analysis/model_architecture_summary/`。

Rationale: CLI 让用户快速比较架构，不需要启动训练；同时保持实现逻辑在窄 helper 中，CLI 只是 parser/renderer。

### Decision 7: 训练 startup summary 向新 schema 兼容迁移

`build_startup_summary()` 可继续保留 `parameters.total_params`、`parameters.modules.*` 旧字段，同时新增或引用：

- `architecture_summary`
- `parameters.schema_version`
- `parameters.effective_params`
- `parameters.excluded_parameter_groups`

TensorBoard 现有 `model/total_params`、`model/trainable_params` 不变；可追加关键组件 scalars，但不删除旧 tag。

Rationale: 训练日志和测试已经消费旧字段，直接替换会制造无意义破坏。增量字段能让新工具复用旧 artifact，也能保护下游分析脚本。

### Decision 8: JEPA sweep 先 adapter，后收敛估算器

第一阶段在 sweep summary 中增加 adapter，把现有 `params_metadata` 映射到统一 schema，并补充 `parameter_count_source=declared_candidate_metadata`。随后可把 sweep 内 `_params_metadata()` 的输出字段改为直接生成统一 schema 的 `components` 和 `parameters` 子集。

必须添加 fixture 覆盖：

- `patch14_stage1_gps_query`: total 约 0.197M、image encoder 约 0.117M、visual/context encoder 约 0.088M。
- `resnet18_layer4_tokens`: total 约 11.32M、image encoder 约 11.24M、visual/context encoder 约 11.21M。
- `resnet18_layer3_layer4_tokens`: total 约 14.13M、image encoder 约 14.05M、visual/context encoder 约 14.02M。

Rationale: 这些数值是用户当前正在用来判断架构规模收益的口径，必须被自动测试锁住，而不是写在一次性说明里。

## Risks / Trade-offs

- [Risk] 参数统计口径过多，用户不知道该看 `total_params` 还是 `effective_params`。→ Mitigation: schema 中强制记录字段定义、source 和 excluded groups；Markdown renderer 默认同时显示 total/trainable/effective，并把 excluded reason 展开。
- [Risk] whole-model exception 内部结构差异大，role 推断不准确。→ Mitigation: fallback 保证总数正确；focused tests 只要求已知当前模型的关键 role；后续模型可通过 metadata hints 增强。
- [Risk] build 配置可能触发 checkpoint 下载或重依赖导入。→ Mitigation: summary build 默认 `allow_download=false` 或等价安全策略；遇到需要网络的 pretrained 路径输出 warning/error，用户显式 opt-in 才允许。
- [Risk] JEPA sweep 的声明参数与真实实例参数存在偏差。→ Mitigation: 强制记录 `parameter_count_source`；支持 optional actual-build 校验；超过阈值写 warning，不静默覆盖。
- [Risk] TinyViT unused ImageNet head 可能导致现有 tests 期望变化。→ Mitigation: 首版 summary 先揭示并可选排除；是否删除/替换 TinyViT head 可在 TinyViT bugfix 或本 change 实现阶段根据 tests 单独处理。
- [Risk] CLI 新增会触碰 entrypoint allowlist。→ Mitigation: 如果新增 console script，必须同步 `docs/maintainer_context_index.yaml` owner metadata 和架构边界测试；也可先提供 `python -m kd_sensing.cli.model_architecture_summary` 包内入口。

## Migration Plan

1. 新增 summary helper 和 focused tests，先覆盖纯 Python API 和 synthetic module，不接 CLI。
2. 接入真实模型实例：ResNet image、image+GPS modular、TinyViT scratch、JEPA context image 或等价轻量 fixture。
3. 将 `module_trainability_report()` 迁移为调用或包装新 helper，同时保留旧字段。
4. 接入 JEPA visual sweep adapter，锁定 `patch14_stage1_gps_query` 与 ResNet token 候选的参数口径。
5. 添加 CLI/renderer、文档和维护上下文索引；确认默认无 dataset/训练/checkpoint 副作用。
6. 运行 focused tests、OpenSpec validate、配置加载、sweep focused tests 和架构边界测试。

Rollback: 删除新增 summary helper、CLI、tests 和文档即可；startup summary 可回退到旧 `module_trainability_report()`，现有模型训练和评估行为不受影响。

## Open Questions

- TinyViT downstream 未使用的 ImageNet classifier head 应在实现阶段直接替换为 `nn.Identity()`，还是只在 summary 中作为 `excluded_parameter_group` 处理？倾向先 summary 揭示，再视 focused tests 和 checkpoint loading 兼容性决定是否修复实现。
- CLI 名称采用 `kd-sensing-model-summary` 还是更明确的 `kd-sensing-model-architecture-summary`？倾向较短名称，但需要与现有 console script 风格一致。
- `patch14_stage1_gps_query` 的 0.197M 参数口径来自当前用户实现和 sweep 记录；实现时应以 source-managed manifest fixture 还是真实 build fixture 作为权威？倾向两者都保留：manifest fixture 锁声明口径，actual-build smoke 锁当前构造路径。
