## MODIFIED Requirements

### Requirement: MMW current surface 只包含 U0、两个 baseline 与限定轨迹协议消融

MMW canonical recipes MUST 仅为 `configs/mmw/u0.yaml`、`amber_full.yaml` 和 `rmbp_mm.yaml`，共享 `_base.yaml`。MMW public launcher、evaluator 和 summary MUST 仅接受 `U0`、`amber_full`、`rmbp_mm`。trajectory-disjoint 本地研究工具 MAY 复用 Candidate12 公共 encoder/fusion 并仅提供 M0 线性、M1 普通 prototype、M2 topology prototype、M3 topology prototype 加 random-balanced、M4 availability-balanced topology consistency 五个固定方法及 M4 的三个固定因果消融；这些方法 MUST NOT 成为 canonical recipe 或 public CLI route。

#### Scenario: 加载 MMW canonical recipe

- **WHEN** 用户加载任一 MMW canonical recipe
- **THEN** 配置 MUST 不包含 BCACL、CMSBL、capacity/reference、nested capacity、recovery 或历史 ablation 字段

#### Scenario: 运行轨迹协议五个方法

- **WHEN** 本地 trajectory-disjoint runner 接受方法名
- **THEN** 只允许 M0--M4 五个固定方法及 M4-a/M4-b/M4-c 三个因果消融
- **AND** MUST NOT 恢复 Router、PAMR、attention、Transformer、reconstruction、motion refinement 或 topology assignment
