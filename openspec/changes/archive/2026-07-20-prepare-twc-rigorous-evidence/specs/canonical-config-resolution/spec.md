## MODIFIED Requirements

### Requirement: current recipes 是唯一 canonical 配置面
配置加载 MUST 将 `configs/mmw/` 下 T2、S1、MaskTrain-CLS、AMBER-Full、RMBP-MM、AMR-Net-4M-Adapted recipe 及 shared base，以及 `configs/deepsense6g/` 下 T2、MaskTrain-CLS、AMBER-Full、RMBP-MM、AMR-Net-4M-Adapted recipe 及 shared base视为 current canonical inputs。所有 recipe MUST 在无 `outputs/`、数据和 checkpoint 时完成解析。

#### Scenario: clean clone 解析新增 recipe
- **WHEN** 用户加载任一新增 baseline recipe
- **THEN** loader MUST 不读取历史 resolved YAML 或 retired config
- **AND** recipe MUST 声明四模态、dataset protocol、window、loss 和 paper-equivalence 边界
