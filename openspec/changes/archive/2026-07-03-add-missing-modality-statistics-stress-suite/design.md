## Context

当前 U-MaskBeamJEPA eval matrix 和 Scene31 fresh eval 能输出 pattern-level 指标，但论文 claim 仍缺两层证据：多 seed 统计显著性和系统化 stress-curve。缺失模态研究如果只展示 fixed pattern 的单次结果，容易被审稿人质疑随机性、条件选择偏差和不完整扰动。

本设计把“统计证据”和“stress suite”分离：统计模块消费已有 summary/eval 输出，stress suite 定义真实评估条件并复用 difficulty pipeline。二者共同产出 claim-oriented summary，但不自动升级 claim。

## Goals / Non-Goals

**Goals:**

- 为 method/seed/pattern 汇总 mean、std、bootstrap CI、paired delta、win/loss count 和可选显著性检验。
- 为缺失模态鲁棒性定义 canonical stress suite，覆盖 missing、unavailable、noise、async 和 severity sweep。
- 扩展 eval matrix 输出 strict comparability fields 和 seed aggregation 输入。
- 让 AMBER-lite/full、RMBP-MM、U-MaskBeamJEPA 等本地 baseline 能在同一 stress manifest 下比较。
- 所有扰动复用 difficulty pipeline，不改变 target、split 或 sample identity。

**Non-Goals:**

- 不实现新的模型结构、fusion 模块或训练 loss。
- 不替代 predictive JEPA robustness 的专属 benchmark。
- 不把 synthetic/smoke metrics 写成真实 claim。
- 不引入重型统计依赖；优先使用 numpy/scipy/pandas 已有依赖。

## Decisions

1. **统计汇总先消费 CSV/JSON，不嵌入训练 loop。**
   训练不应为了统计显著性改变 runtime；summary 工具从 eval/fresh-eval 产物读取 rows，更容易重跑和审计。

2. **默认 bootstrap CI，显著性检验可选。**
   bootstrap 对样本量和分布假设更宽松；paired permutation 或 Wilcoxon 作为可选项，仅在 paired seed/method 完整时输出。

3. **stress suite 使用 manifest 表达。**
   Manifest 声明 model groups、conditions、severity、difficulty profile、strict fields 和输出路径。这样可用于 Scene31、RBMA、AMBER、RMBP-MM，而不写死到某个模型。

4. **缺失模态 stress 与 predictive stress 分工。**
   本 change 的 canonical 条件面向模态缺失和可用性；predictive JEPA 的 history/current-frame 语义仍由 `predictive-jepa-robustness` 管。

5. **claim gate 只输出状态，不改文档。**
   stress summary 可以输出 `claim_ready=true/false` 和原因；正式 claim registry 仍人工维护。

## Risks / Trade-offs

- **Risk:** 多模型输出字段不一致。  
  **Mitigation:** 定义 normalized metric schema；缺字段时保留原始字段并标记 `metric_missing`。

- **Risk:** severity sweep 计算成本高。  
  **Mitigation:** manifest 支持 smoke、quick、formal 三档；formal 才要求完整 severity 和多 seed。

- **Risk:** paired test 因 seed 不对齐失效。  
  **Mitigation:** 输出 unpaired summary，同时 warning 说明 paired evidence unavailable。

- **Risk:** difficulty operator 对某模态不可用。  
  **Mitigation:** stress row 标记 `unavailable` 或 fallback，不能静默当作 clean。

## Migration Plan

1. 定义统计 summary schema 和 stress manifest schema。
2. 扩展 eval matrix 输出 required comparability fields。
3. 实现统计聚合器和 synthetic fixture tests。
4. 实现 missing-modality stress suite manifest normalizer 和 difficulty pipeline adapter。
5. 更新主线文档和 claim registry 升级规则说明。
6. 回滚时删除新增 summary/stress 工具；已有训练与 eval matrix clean/fixed pattern 行为保持不变。

## Open Questions

- 首版是否要把 Wilcoxon/permutation 都实现，还是先实现 bootstrap CI 和 paired delta。
- canonical severity values 是否固定为 `[0, 0.25, 0.5, 0.75, 1.0]`，还是允许每个 operator 单独声明。
- formal stress suite 是否要求所有 baseline 都重新 forward，还是允许读取已审计 per-condition metrics。
