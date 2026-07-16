## MODIFIED Requirements

### Requirement: current recipes 是唯一 canonical 配置面

配置加载 MUST 仅将 `configs/mmw/t2.yaml`、`s1.yaml`、`amber_full.yaml`、`rmbp_mm.yaml` 及其 tracked shared base，以及 `configs/deepsense6g/t2.yaml` 及其 tracked shared base 视为 current canonical inputs。

#### Scenario: 干净 clone 解析 recipe

- **WHEN** 用户加载任一 retained recipe
- **THEN** loader MUST 在没有 `outputs/`、checkpoint 和本地数据时完成 parse、normalization 与 validation
- **AND** 配置 MUST 声明 MMW 或受限 DeepSense6G 四模态协议
