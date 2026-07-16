## MODIFIED Requirements

### Requirement: T2/baseline 与双数据集是唯一 current 研究 surface

系统 MUST 将 MMW 的 T2、S1、AMBER-Full、RMBP-MM，以及 DeepSense6G Scene31–34 的 T2 四模态数据路径和其传递运行依赖视为 current source surface。无法追溯到这些训练、评估、预处理、fixed-mask 或 active T2 BPA/CMA/hyperparameter protocol 的项目项 MUST 退役。

#### Scenario: 标记 current 路径

- **WHEN** 代码、YAML、CLI、script、测试或文档被标记为 current
- **THEN** inventory MUST 能说明其 MMW T2/baseline owner、DeepSense6G T2 owner 或 active T2 task
- **AND** 无法建立该路径时该项 MUST 从 current surface 删除

### Requirement: current 方法必须有 tracked canonical recipe

系统 MUST 在 `configs/mmw/` 中提供 T2、S1、AMBER-Full、RMBP-MM recipe 及其 shared base，并在 `configs/deepsense6g/` 中提供受限 Scene31–34 T2 recipe 及其 shared base。launcher MUST 不把 output、checkpoint 或历史 resolved config 当作源码输入。

#### Scenario: 干净 clone 配置 dry-run

- **WHEN** 用户在没有本地 `outputs/` 的 clone 加载任一 current recipe
- **THEN** loader MUST 能完成 parse、normalization 与 validation
- **AND** recipe MUST 声明其 MMW 或 DeepSense6G 四模态协议
