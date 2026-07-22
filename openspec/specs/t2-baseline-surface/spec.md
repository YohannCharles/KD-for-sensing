# t2-baseline-surface Specification

## Purpose

定义 MMW T2/baseline 与受限 DeepSense6G T2 的 current 研究闭包，使代码、YAML、CLI、script、测试和文档均能追溯到当前 owner。
## Requirements
### Requirement: CMSBL 是唯一 active T2 研究扩展

系统 MUST 将 MMW 的 T2、S1、AMBER-Full、RMBP-MM、DeepSense6G Scene31--34 T2，以及 BCACL U2/CMSBL 视为 current source surface。PCER、PGCD、动态 Router、PR-SQDF、missing residual、feature/prototype fusion、availability fallback、prototype diagnostics 和 BT-SCL MUST 只保留历史说明与 archive。

#### Scenario: 标记 current 路径

- **WHEN** 代码、YAML、CLI、script、测试或文档被标记为 current
- **THEN** inventory MUST 能说明其 T2/baseline、BCACL U2 或 CMSBL owner
- **AND** 无法建立该路径时该项 MUST 从 current surface 删除

### Requirement: current 方法必须有 tracked canonical recipe

系统 MUST 在 `configs/mmw/` 中提供 T2、S1、AMBER-Full、RMBP-MM recipe 及其 shared base，并在 `configs/deepsense6g/` 中提供受限 Scene31–34 T2 recipe 及其 shared base。launcher MUST 不把 output、checkpoint 或历史 resolved config 当作源码输入。

#### Scenario: 干净 clone 配置 dry-run

- **WHEN** 用户在没有本地 `outputs/` 的 clone 加载任一 current recipe
- **THEN** loader MUST 能完成 parse、normalization 与 validation
- **AND** recipe MUST 声明其 MMW 或 DeepSense6G 四模态协议

### Requirement: 退役路线只保留历史说明

系统 MUST 以集中历史说明和 dated OpenSpec archive 记录退役路线的用途、范围和替代关系。系统 MUST 不保留 retired source module、实体 YAML、console script、thin wrapper、alias、migration guard 或 compatibility stub。

#### Scenario: 用户引用退役路径

- **WHEN** 用户尝试加载已退役配置或导入已退役模块
- **THEN** 普通文件不存在或普通 unknown-name 错误即可
- **AND** 系统 MUST 不自动迁移、映射或构建替代运行路径

### Requirement: AMBER-Full token padding 不得改变可用性语义
AMBER-Full 在对齐 modality spatial-token 维度时 MUST 为 padding token 生成不可用 mask；fusion attention、auxiliary loss 和 pooled diagnostics MUST 忽略 padding token。

#### Scenario: GPS token 少于 image spatial tokens
- **WHEN** image/radar/lidar 有多个 spatial token 而 GPS 只有一个 token
- **THEN** GPS 的补齐 token MUST 在 attention key-padding mask 中为 true
- **AND** GPS pooled feature MUST 只平均其真实 token
