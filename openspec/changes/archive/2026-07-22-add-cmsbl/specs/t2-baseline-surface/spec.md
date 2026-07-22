## MODIFIED Requirements

### Requirement: CMSBL 是唯一 active T2 研究扩展

系统 MUST 将 MMW 的 T2、S1、AMBER-Full、RMBP-MM，DeepSense6G Scene31--34 T2，以及 BCACL U2/CMSBL 训练依赖视为 current source surface。PGCD、PCER、动态 Router、PR-SQDF、missing residual、feature/prototype fusion、availability fallback、prototype diagnostics 和 BT-SCL MUST 只保留历史说明与 archive，不保留 source、YAML、script、test 或 compatibility stub。

#### Scenario: 标记 current 路径

- **WHEN** 代码、YAML、CLI、script、测试或文档被标记为 current
- **THEN** inventory MUST 能说明其 T2/baseline、BCACL U2 或 CMSBL owner
- **AND** 无法建立该路径时该项 MUST 从 current surface 删除

### Requirement: current 方法必须有最小 tracked recipe

系统 MUST 在 `configs/mmw/` 中只提供 shared base、T2、S1、AMBER-Full、RMBP-MM，在 `configs/deepsense6g/` 中只提供 shared base 与 T2。CMSBL/BCACL MUST 默认关闭，且 retained recipe MUST 在没有 outputs、数据和 checkpoint 的 clone 中解析。

#### Scenario: 干净 clone 配置 dry-run

- **WHEN** 用户加载任一 retained recipe
- **THEN** loader MUST 完成 parse、normalization 与 validation
- **AND** 不得读取 capacity stats、cache 或历史 resolved config
