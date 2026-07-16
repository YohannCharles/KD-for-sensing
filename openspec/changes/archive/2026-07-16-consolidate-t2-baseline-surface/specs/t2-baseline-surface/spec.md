## ADDED Requirements

### Requirement: T2/baseline 是唯一 current 研究 surface
系统 MUST 只将 MMW `T2`、`S1`、`AMBER-Full` 和 `RMBP-MM` 及其传递运行依赖视为 current source surface。任何不服务这四方法训练、评估、预处理、固定掩码或 active T2 BPA/CMA/hyperparameter protocol 的代码、配置、入口和测试 MUST 退役。

#### Scenario: current path 可追溯到四方法
- **WHEN** 代码、YAML、CLI、script 或测试被标记为 current
- **THEN** inventory MUST 说明其服务的 T2/baseline owner 或 active T2 task
- **AND** 无法建立该路径时该项 MUST 从 current source surface 删除

### Requirement: 四方法必须有 tracked canonical recipe
系统 MUST 在 tracked `configs/mmw/` 中提供 T2、S1 和 RMBP-MM recipe，并保留 AMBER-Full 的 tracked recipe。所有 T2/baseline launcher MUST 从这些 recipe 或其 tracked shared base 解析配置，MUST NOT 把 `outputs/`、checkpoint 或历史 resolved config 作为源码输入。

#### Scenario: 干净 clone 生成四方法配置
- **WHEN** 用户在没有任何本地 `outputs/` 的 clone 中执行 T2/baseline launcher dry-run
- **THEN** launcher MUST 能解析四方法的 tracked recipe
- **AND** 不得因缺少历史 resolved YAML 而失败

### Requirement: 退役路线只保留历史说明
系统 MUST 以集中历史说明和 dated OpenSpec archive 记录退役路线的用途、范围和替代关系。系统 MUST 不保留 retired source module、实体 YAML、console script、thin wrapper、alias、migration guard 或 compatibility stub。

#### Scenario: 用户引用退役路径
- **WHEN** 用户尝试加载已退役配置或导入已退役模块
- **THEN** 普通文件不存在或普通 unknown-name 错误即可
- **AND** 系统 MUST 不自动迁移、映射或构建替代运行路径
