## Context

当前 JEPA 下游链路已经形成一条可复用主线：

```text
image_batch ──▶ jepa_context_image ──▶ [B,T,N,D] patch tokens ──▶ pooling ──▶ [B,T,D]
gps_batch   ──▶ gps_mlp + projector ────────────────────────────────┘
                                                                  │
                                                                  ▼
                                                modular_sequence fusion core + beam head
```

`fair_gps_biased` 是当前 Image+GPS+JEPA 主线 baseline，GPS-query pooling 已作为派生配置证明了“image encoder 依赖 projected GPS feature”这条路径可行。问题是：当前实现仍把 Stage 1 JEPA 主模型、downstream image encoder、GPS-query pooler、checkpoint 加载和 metadata 逻辑集中在少数模块中；如果继续试验 K-token pooling、motion-aware query、轻量 adapter 或不同冻结/学习率策略，隐式契约会越来越多。

本设计围绕“让后续插模块更像换配置，而不是改训练器”展开。训练入口、dataset、Stage 1 JEPA objective、checkpoint schema 和现有 `fair_gps_biased` baseline 均保持稳定。

## Goals / Non-Goals

**Goals:**

- 把 JEPA downstream image encoder 内部拆成 context token encoder、pooler 和可选 adapter 的清晰边界。
- 让 mean pooling、现有 GPS-query pooling 和后续 pooler/adapter 都能通过配置构建，并保持输出兼容现有 `[B,T,D]` fusion 路径。
- 正式化 conditioned encoder 契约，使任意 encoder 都能声明依赖 projected/encoded/raw 条件特征，并由 `ModularSequenceModel` 负责依赖排序和 shape 校验。
- 为 `fair_gps_biased` 派生实验提供参数组 optimizer 能力，支持 context encoder、GPS encoder、pooler/adapter、fusion core 和 head 的差异化学习率、weight decay 和冻结策略。
- 将 JEPA downstream 结构 metadata 由子模块声明并聚合写出，减少 config 手写解析漂移。

**Non-Goals:**

- 不改变 GPS-conditioned JEPA Stage 1 预训练目标、mask sampler、EMA target encoder 或 checkpoint 格式。
- 不把训练主循环改成 JEPA 专用分支；supervised beam 仍走现有 `model.primary`、loss、metric、checkpoint workflow。
- 不替换或重命名 `fair_gps_biased` mean-pooling baseline；新结构只服务派生实验。
- 不恢复 KD/distillation、HiST/Hist、Top8 selector、GPS residual、camera residual 或旧 fusion 兼容入口。
- 不要求新增外部依赖；第一版仍使用 PyTorch 和现有 registry 机制。

## Decisions

### Decision 1: 下游 pooler/adapter 独立于 Stage 1 JEPA 主模型

新增或整理 JEPA downstream 相关模块时，把 Stage 1 自监督模型和 Stage 2 supervised reuse 逻辑分开。`GPSConditionedJEPA` 继续负责 context/target encoder、GPS conditioner、predictor、mask sampler 和 EMA；`JepaContextImageEncoder` 只负责加载 context encoder 权重、生成 patch tokens、调用 downstream pooler/adapter 并输出下游特征。

推荐结构是将 pooler/adapter 放到独立窄模块，例如 `kd_sensing.models.jepa_downstream` 或 `kd_sensing.models.jepa_pooling`。`models/jepa.py` 可以继续导出公开符号，但不继续承载所有 pooler 实现。

替代方案是继续在 `JepaContextImageEncoder.__init__` 中为每种 pooling 添加分支。这个方案短期最快，但每新增一个模块都会扩大同一个类的职责，也会让 metadata、测试和配置验证越来越难收敛。

### Decision 2: pooler/adapter 使用配置构建，默认兼容 mean pooling

`jepa_context_image` 的默认行为保持为 mean pooling。新配置可以显式声明：

```yaml
model:
  primary:
    encoders:
      image:
        type: jepa_context_image
        pooler:
          type: gps_query_attention
          k_queries: 4
          num_heads: 4
          condition_source: projected_gps
        adapter:
          type: identity
```

为兼容现有配置，`pooling: mean` 和 `pooling: gps_query_attention` MAY 继续作为薄 alias 解析为等价 `pooler` 配置。第一版 pooler 输出仍默认为 `[B,T,D]`，使 existing projector、fusion core 和 beam head 不需要改变。

替代方案是第一步直接输出 `[B,T,K,D]` 或完整 `[B,T,N,D]` 给 fusion core。这个方向值得保留，但会改变 encoder/projector/core/head 的统一 shape 契约。更稳的路线是先抽象 pooler 边界，再用单独任务扩展 token-valued representation。

### Decision 3: conditioned encoder 契约由模型声明、模块化模型执行

encoder 可以通过轻量属性或协议声明：

- `required_context_modalities`: 依赖哪些已启用模态。
- `context_feature_source`: 使用 `projected`、`encoded` 或 `raw` 条件特征。
- `context_feature_kwargs`: 每个依赖注入到 encoder forward 的 kwarg 名称。
- 可选 `context_feature_shapes` 或 `condition_dim` metadata，用于更早暴露维度错误。

`ModularSequenceModel` 负责：

1. 构建所有启用模态 encoder/projector。
2. 按依赖关系执行 encoder，先处理无依赖模态。
3. 对 projected/encoded/raw 条件特征做 batch/time shape 校验。
4. 对缺失依赖、未启用模态、自依赖或循环依赖抛出清晰错误。

替代方案是让每个 conditioned encoder 自己从 batch 读取所需模态。这个方案会绕过统一 normalization/projector 契约，并把 dataset 字段知识渗入模型模块。

### Decision 4: 参数组 optimizer 位于 `engine.optim`，训练器只消费结果

为 `fair_gps_biased` 调参提供参数组能力，但实现仍放在 `kd_sensing.engine.optim`。训练器继续调用 `build_optimizer(cfg, primary_model)` 和 `optimizer_param_group_summary()`，不理解 JEPA 细节。

建议配置形态：

```yaml
training:
  optimizer:
    type: adam
    parameter_groups:
      - name: jepa_context_encoder
        module_patterns: ["encoders.image.context_encoder"]
        lr: 0.0001
        weight_decay: 0.0001
      - name: jepa_pooler
        module_patterns: ["encoders.image.pooler", "encoders.image.adapter"]
        lr: 0.001
      - name: gps_encoder
        module_patterns: ["encoders.gps", "projectors.gps"]
        lr: 0.00075
      - name: fusion_head
        module_patterns: ["representation_core", "heads"]
        lr: 0.00075
```

参数组匹配必须可诊断：未匹配 pattern、重复匹配参数、剩余未分组 trainable 参数都要有明确行为。默认可将未分组 trainable 参数放入 `main` 组，除非配置 `require_all_matched: true`。

替代方案是在模型内部冻结或手动返回 parameter groups。这样会把优化策略绑到模型实现里，不利于同一模型在不同实验口径下复用。

### Decision 5: runtime metadata 从模块声明聚合

保留现有 `jepa_downstream_metadata(cfg)` 输出字段，但逐步让 `ModularSequenceModel` 和子模块提供 `training_strategy_metadata()` 或等价只读方法。artifact writer/run metadata 可以优先读取模型声明的 metadata；在模型未构建的 config-only 路径中，继续使用配置解析作为 fallback。

metadata 至少记录：

- JEPA checkpoint path、state dict prefix、freeze encoder。
- pooler type、adapter type、condition source、k queries、num heads、是否返回 attention。
- 是否启用 token-valued representation。
- optimizer parameter group 名称、学习率、参数数量。
- `fair_gps_biased` baseline 与派生实验的 ablation 标识。

替代方案是继续在 `run_metadata.py` 逐字段解析 YAML。这个方案对当前两个配置还可控，但后续 pooler/adapter 组合增加时很容易漏字段。

### Decision 6: 派生配置只覆盖实验变量

继续沿用 `configs/fusion/experiments/jepa_image_gps/` 的分区：主 baseline 保持 `image_gps_jepa_gps_biased_best_*`，派生配置只覆盖 pooler/adapter、参数组、冻结策略、run name 和 metadata ablation。BeamBench-fair 与 2604-style 配置不得混用 checkpoint、label space、split protocol 或学习率 recipe。

替代方案是提升一个新的 root-level canonical config。现阶段还处于方法探索期，放在 experiments 子目录更符合当前配置边界。

## Risks / Trade-offs

- [Risk] pooler registry 和 adapter 边界过早抽象，实际只用到一种 GPS-query。→ Mitigation：第一版只抽出当前已存在的 mean/GPS-query 行为，并用 focused tests 保护兼容；不引入复杂 token-valued API。
- [Risk] 参数组 pattern 配错导致关键模块没训练或重复更新。→ Mitigation：构建 optimizer 时输出匹配 summary；测试覆盖未匹配、重复匹配和 `require_all_matched`。
- [Risk] conditioned encoder 支持 raw/encoded/projected 三种来源后，依赖路径变复杂。→ Mitigation：默认推荐 `projected`，raw 仅用于明确声明的轻量条件；错误信息必须包含 dependency、source 和 shape。
- [Risk] runtime metadata 既支持模型声明又支持 config fallback，短期存在两条路径。→ Mitigation：以模型声明为权威，config fallback 仅用于构建前 metadata 和历史配置；测试要求两者核心字段一致。
- [Risk] K-token pooling 后续仍会要求改 core/head 契约。→ Mitigation：本 change 只预留 metadata 和 pooler 边界；真正输出 `[B,T,K,D]` 时另开 OpenSpec change。

## Migration Plan

1. 新增 JEPA downstream pooler/adapter 窄模块和默认 mean/GPS-query pooler 构建路径，保留现有 `pooling` 字段兼容解析。
2. 整理 `JepaContextImageEncoder`，让其调用 pooler/adapter，同时保持现有 checkpoint 加载、freeze 和 `[B,T,D]` 输出。
3. 正式化 `ModularSequenceModel` conditioned encoder helper，补充 raw/encoded/projected 来源、错误信息和 focused tests。
4. 扩展 `engine.optim` 支持配置化 parameter groups，并保持无参数组配置时的现有单 `main` 组行为。
5. 扩展 runtime metadata 收集，优先聚合模型/子模块 metadata，保留 config fallback。
6. 新增派生配置和 README 说明，明确它们只作为 `fair_gps_biased` baseline 的对照实验。
7. 运行 OpenSpec strict validate 和相关 focused tests。

Rollback 方式是移除新增 pooler/adapter 注册、参数组配置和派生 YAML；保留现有 `pooling: mean` 与 `pooling: gps_query_attention` 路径时，已有 baseline 和 GPS-query 配置不需要迁移。

## Open Questions

- 是否第一版就加入 `adapter` 配置，还是只先抽 `pooler`，把 adapter 留作第二个小 change。
- 参数组 pattern 使用模块名前缀、正则、glob，还是只支持安全的 dotted-prefix 列表。
- runtime metadata 是否需要写入 attention entropy/peakiness 的训练聚合字段，还是先只记录结构配置。
- 如果后续 K-token pooling 要接入 `next_beam_query_transformer`，是扩展现有 core 输入契约，还是新增专用 token-valued JEPA core。
