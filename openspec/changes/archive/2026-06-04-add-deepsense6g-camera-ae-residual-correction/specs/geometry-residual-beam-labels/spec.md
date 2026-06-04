## ADDED Requirements

### Requirement: GPS-prior local residual delta class
系统 MUST 支持以 GPS prior top1 为参照的 signed circular residual delta class。该 class MUST 使用 circular signed residual，默认 local delta radius 为 `8`，并提供 overflow class 表示超出 local window 的 residual。

#### Scenario: signed circular residual wrap-around
- **WHEN** `num_beams=64`、`target=1` 且 `gps_pred_top1=63`
- **THEN** `signed_circular_residual(target, gps_pred_top1, num_beams=64)` MUST 返回 `2`
- **AND** 该 residual MUST 能映射到 local delta class

#### Scenario: negative signed residual wrap-around
- **WHEN** `num_beams=64`、`target=63` 且 `gps_pred_top1=1`
- **THEN** `signed_circular_residual(target, gps_pred_top1, num_beams=64)` MUST 返回 `-2`
- **AND** 该 residual MUST 能映射到 local delta class

#### Scenario: residual to delta class
- **WHEN** residual 位于 `[-R, R]`
- **THEN** `residual_to_delta_class(residual, radius=R)` MUST 返回稳定的 class id
- **AND** `delta_class_to_residual(class_id, radius=R)` MUST 还原对应 residual

#### Scenario: overflow class
- **WHEN** residual 的绝对值大于 `R`
- **THEN** `residual_to_delta_class` MUST 返回 overflow class `2R+1`
- **AND** `delta_class_to_residual` 对 overflow class MUST 返回 `None` 或配置声明的 special value
- **AND** diagnostics MUST 能统计 overflow count
