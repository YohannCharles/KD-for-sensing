## ADDED Requirements

### Requirement: H4 U-Mask training profile 可追溯且可选择
系统 MUST 定义 `umask_h4_v1` 与 `legacy_h0_v1` 两个 U-Mask training profile。H4 MUST 物化 AdamW、`lr=5e-4`、`weight_decay=3e-4`、`T_0=40`、`T_mult=1`、`eta_min=1e-6` 的 cosine warm restart；legacy H0 MUST 物化 Adam、`lr=5e-4`、`weight_decay=1e-4` 和 disabled scheduler。新的 T2/S1 mainline launcher MUST 显式请求 H4，历史 H0 screening 与 BPA/CMA ablation MUST 显式请求 legacy H0；AMBER-Full 与 RMBP-MM MUST 不接受 U-Mask profile 覆盖。

#### Scenario: 主线 T2 生成 H4 配置
- **WHEN** MMW mainline launcher 为 T2 或 S1 构建配置并显式请求 H4 profile
- **THEN** resolved config MUST 使用 `umask_h4_v1` 的 optimizer、weight decay 与 scheduler
- **AND** 运行 profile MUST 在 protocol provenance 中完整记录

### Requirement: RouterNoPattern 作为受控的 U-Mask 开发主线架构 profile
系统 MUST 定义 `umask_router_nopattern_v1` 与 `umask_router_pattern_v1` 两个 router architecture profile。新的 T2/S1 mainline launcher MUST 显式选择 `umask_router_nopattern_v1`，它只将 `model.primary.router_use_pattern_features` 设为 `false`；legacy H0 screening、BPA/CMA ablation 和 tracked base recipe MUST 保持 `umask_router_pattern_v1`。该选择 MUST 记录 canonical values 和 SHA256，并随 checkpoint/evaluation provenance 传播。

#### Scenario: T2/S1 mainline 使用同一 router profile
- **WHEN** MMW mainline launcher 为 T2 或 S1 构建 H4 配置
- **THEN** resolved config MUST 使用 `umask_router_nopattern_v1`
- **AND** T2 与 S1 MUST 共享该 router architecture profile，避免配对比较混入 router pattern 差异
- **AND** RouterNoPattern MUST 在固定 inner-mask 与多 seed/outer evidence 前保持 development-only 状态，不得升级为论文 claim

#### Scenario: legacy ablation 保持 pattern-on
- **WHEN** launcher 为 T2 BPA/CMA ablation 或 legacy hyperparameter screening 构建配置
- **THEN** resolved config MUST 使用 `umask_router_pattern_v1`
- **AND** 不得隐式继承 mainline RouterNoPattern setting

#### Scenario: H0 mechanism control 保持不变
- **WHEN** launcher 为 T2 BPA/CMA ablation 或 legacy hyperparameter screening 构建配置
- **THEN** resolved config MUST 使用 `legacy_h0_v1`
- **AND** 不得从 H4 mainline profile 继承 optimizer 或 scheduler

### Requirement: 设计筛选使用受限、可归因的分阶段矩阵
系统 MUST 提供独立的 MMW T2 design-screening launcher，使用固定 15-domain、5-to-1、40 epoch、`last.pth`、共同 inner split 和共同 batch。每个候选 MUST 有唯一 id、matched control、允许差异清单、resolved config fingerprint 和 `development_only=true` provenance；outer test MUST 不用于选择。

#### Scenario: 第一波 config-only screen
- **WHEN** 用户请求第一波 eight-GPU 筛选
- **THEN** launcher MUST 生成 H4 control、同步 encoder output 的 d_model、router capacity/feature 和 GPS MLP capacity 的单因素候选
- **AND** 每个候选 MUST 使用相同 inner split、seed、batch、epoch 与 mask protocol

#### Scenario: 候选晋级
- **WHEN** single-seed inner summary 完成
- **THEN** 只有满足预注册 J 与保护门槛的候选才可标记为晋级
- **AND** 未晋级候选 MUST 不进入组合或 outer evaluation

#### Scenario: development candidate 完成训练
- **WHEN** H4 design candidate 完成其 40 epoch training
- **THEN** trainer MUST NOT 构造或迭代 outer test loader，也不得发布任何 outer-test metrics
- **AND** `final_test_metrics` 如存在，MUST 仅记录未执行状态而不能包含 test evidence
- **AND** candidate 只能记录 inner train/validation 的 development evidence

### Requirement: BPA、CMA 与额外 loss 的比较边界明确
设计筛选 MUST 将 BPA temperature、sigma、fused weight 与 modality weight 作为可单独变化的候选。CMA MUST 从 BPA-disabled matched control 出发并保持 BPA/CMA 互斥；same-model superset KL 可作为已有额外 loss 的候选，但系统 MUST 不在本 change 中临时引入无预注册假设的新 loss。

#### Scenario: CMA sensitivity candidate
- **WHEN** launcher 构建 CMA weight 或 temperature 候选
- **THEN** BPA auxiliary weights MUST 为零且 CMA provenance MUST 标记 objective replacement
- **AND** 汇总不得将该行解释为 H4 BPA 上的叠加收益
