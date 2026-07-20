## ADDED Requirements

### Requirement: 历史 local validation 与 strict outer evidence 必须不可混合
MMW multiseed evidence MUST 区分历史 `local_validation` 与 `mmw_twc_outer_v1`。strict outer rows MUST 携带 protocol id、split manifest SHA256、outer-evidence role、mask cache checksum 和 fixed seed set；summary MUST 不能将不同 evidence protocol 的行合并或配对。

#### Scenario: 拒绝混合历史与 strict 行
- **WHEN** summary 同时读取历史 local validation 和 strict outer evidence
- **THEN** 它 MUST 对跨 protocol 聚合抛出 identity error
- **AND** 不得生成混合均值或 paired delta

### Requirement: baseline fidelity 限制必须随 strict evidence 传播
AMBER-Full 和 RMBP-MM 的 strict outer result MUST 继续记录 local adaptation scope、缺失的 paper input/training stage 和 `paper_equivalent=false`；T2/S1 也 MUST 记录 H4/router profile 与 T2 temporal-consistency state。

#### Scenario: 导出主表来源说明
- **WHEN** strict summary 导出 method comparison
- **THEN** 每个 baseline MUST 包含 fidelity metadata
- **AND** claim-facing export MUST 不得省略 `paper_equivalent` 或 method adaptation scope

### Requirement: strict execution matrix 必须收敛为六个 current 方法
系统 MUST 运行 T2、S1、MaskTrain-CLS、AMBER-Full、RMBP-MM、AMR-Net-4M-Adapted 六个 method cell。Pattern-weighted CE 在 seed1 开发筛选未显示稳定收益后 MUST 从 current runtime、strict launcher 和正式完整性检查中退役；其历史结果 MAY 作为非 claim 的最小审计记录保留，但 MUST 不参与 strict summary。

#### Scenario: 六方法完整性
- **WHEN** strict summary 聚合固定五 seed 结果
- **THEN** 缺失任一 current method/seed MUST 使六方法执行矩阵 fail closed
- **AND** retired Pattern-weighted 记录的存在或缺失 MUST 不影响正式完整性
