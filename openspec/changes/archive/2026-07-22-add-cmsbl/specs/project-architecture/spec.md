## MODIFIED Requirements

### Requirement: 包导入图收敛到 CMSBL 主线闭包

`kd_sensing` MUST 只保留 T2、S1、AMBER-Full、RMBP-MM、BCACL U2、CMSBL 与双数据集所需 owner。共享 owner MUST 不导入 PCER、PGCD、候选 Router、PR-SQDF、feature/prototype quick search、availability fallback、BT-SCL 或其他 retired family。

#### Scenario: 导入 core surface

- **WHEN** 用户导入 config、registry 或训练入口
- **THEN** 导入 MUST 不读取数据、权重、outputs 或 cache
- **AND** current recipe load 与 synthetic forward MUST 不要求 retired module

### Requirement: 本地产物不参与源码收口

源码删除和 OpenSpec 归档 MUST 不删除、移动或改写 `dataset/`、`outputs/`、`outputs/cache/`、logs 或 checkpoint。

#### Scenario: 执行主线收口

- **WHEN** 维护者删除 retired source 和 tests
- **THEN** 本地产物路径与内容 MUST 保持不变
