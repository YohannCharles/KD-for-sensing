## MODIFIED Requirements

### Requirement: H5/P1 跨方法统一数据划分
H5/P1 temporal matrix launcher MUST 为所有方法生成相同的 Scene31-34 数据范围，并 MUST 对 train、validation 和 test 使用相同的场景列表、split protocol、group-safe split strategy、split seed、split source 和 split fractions。方法基配置中的 Scene31-only 字段 MUST 被公共划分覆盖。重叠 temporal window MUST 按稳定 sequence identity 整组分配；逐样本分层拆分 MUST NOT 用于可比较 temporal evidence。

#### Scenario: AMBER 与 RMBP-MM dry-run 使用 Scene31-34
- **WHEN** 用户对 `amber_full` 和 `rmbp_mm` 运行 launcher dry-run
- **THEN** 两个生成配置的 `scenes`、`train_scenes`、`validation_scenes` 和 `test_scenes` MUST 均为 `[31, 32, 33, 34]`
- **AND** 两个配置 MUST 使用相同的 `stratified_80_10_10` group-safe sequence split strategy

#### Scenario: 不同方法共享相同 split contract
- **WHEN** launcher 同时生成 U-Mask、AMBER 和 RMBP-MM 配置
- **THEN** 所有生成配置的场景列表、split seed、source splits、fractions 和 group identity policy MUST 完全一致

#### Scenario: Temporal split 身份不相交
- **WHEN** H5/P1 workflow 生成 train、validation 和 test split artifact
- **THEN** 任意两个 split 的 sequence group identity MUST 两两不相交
- **AND** 任意两个 split 实际引用的 sample、历史输入帧和 target 帧 identity MUST 两两不相交
- **AND** 发现交集时 workflow MUST 在训练前失败并报告交集类型和有限示例

#### Scenario: 逐样本 temporal split 被拒绝
- **WHEN** H5/P1 或其它重叠 temporal window 配置请求逐样本 label-stratified split
- **THEN** 配置解析或 dataloader 构建 MUST 拒绝该策略用于 validation/test evidence
- **AND** 错误 MUST 指向 group-safe sequence strategy
