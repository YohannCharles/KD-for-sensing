## ADDED Requirements

### Requirement: H5/P1 跨方法统一数据划分
H5/P1 temporal matrix launcher MUST 为所有方法生成相同的 Scene31-34 数据范围，并 MUST 对 train、validation 和 test 使用相同的场景列表、split protocol、split strategy、split seed、split source 和 split fractions。方法基配置中的 Scene31-only 字段 MUST 被公共划分覆盖。

#### Scenario: AMBER 与 RMBP-MM dry-run 使用 Scene31-34
- **WHEN** 用户对 `amber_full` 和 `rmbp_mm` 运行 launcher dry-run
- **THEN** 两个生成配置的 `scenes`、`train_scenes`、`validation_scenes` 和 `test_scenes` MUST 均为 `[31, 32, 33, 34]`
- **AND** 两个配置 MUST 使用 `stratified_80_10_10` 和 `stratified_by_target_beam_per_scene`

#### Scenario: 不同方法共享相同 split contract
- **WHEN** launcher 同时生成 U-Mask、AMBER 和 RMBP-MM 配置
- **THEN** 所有生成配置的场景列表、split seed、source splits 和 fractions MUST 完全一致
