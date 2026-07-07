## ADDED Requirements

### Requirement: Predictive GPS query explanatory visualizations 属于 diagnostics bundle
Predictive GPS query 的 attention、gate、PCA、t-SNE 或类似解释性图表 MUST 由 predictive JEPA robustness diagnostics bundle 或其 owner CLI mode 产出。项目 SHOULD 不保留独立 visualization CLI 作为 current entrypoint，除非它承载独立 claim gate。

#### Scenario: explanatory figures 由 bundle 输出
- **WHEN** 协作者需要生成 predictive GPS query explanatory visualizations
- **THEN** 推荐入口 MUST 是 predictive robustness diagnostics bundle 的显式 mode 或 equivalent owner command
- **AND** 输出 manifest MUST 标明这些图是解释性补充，而不是独立通过/失败 claim gate

#### Scenario: 删除独立 visualization CLI
- **WHEN** bundle 已覆盖旧 visualization CLI 的输入、输出路径和图表 metadata
- **THEN** 旧独立 CLI 或 console script MAY 删除
- **AND** docs、tests 和 inventory MUST 指向 bundle mode

### Requirement: Predictive claim 证据不得依赖被删 wrapper 路径
Predictive JEPA robustness 的 claim-facing evidence MUST 引用稳定 benchmark、metrics、diagnostics bundle 或 retained artifacts，而不是引用即将删除的 wrapper path。

#### Scenario: claim notes 更新 wrapper 引用
- **WHEN** claim notes 或 paper-facing docs 提到 predictive GPS query visualization path
- **THEN** 它们 MUST 指向 diagnostics bundle output 或 retained artifact manifest
- **AND** 删除 wrapper 后 claim evidence chain MUST 仍可追溯
