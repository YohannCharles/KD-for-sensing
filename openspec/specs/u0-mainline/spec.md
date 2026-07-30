# U0 Mainline Specification

## Purpose

定义稳定 MMW U0、AMBER-Full、RMBP-MM 与 DeepSense6G baseline 的最小运行闭包和公共边界。

## Requirements

### Requirement: MMW current surface 只包含稳定 canonical baseline

MMW canonical recipes MUST 仅为 `configs/mmw/u0.yaml`、`amber_full.yaml` 和 `rmbp_mm.yaml`，共享 `_base.yaml`。MMW public launcher、evaluator 和 summary MUST 仅接受 `U0`、`amber_full`、`rmbp_mm`。

#### Scenario: 加载 MMW canonical recipe

- **WHEN** 用户加载任一 MMW canonical recipe
- **THEN** 配置 MUST 不包含已退役路线或历史实验字段

### Requirement: U0 使用最小四模态训练闭包

U0 MUST 使用 `image`、`radar`、`gps`、`lidar`、masked-mean temporal pooling、supervised router、prototype/BPA 和同模型 superset consistency。已删除训练模块 MUST 不创建参数、buffer、loss、checkpoint state 或 optimizer group。

#### Scenario: 构建 U0

- **WHEN** 训练入口构建 U0 模型与 loss extension
- **THEN** 仅 U0 所需模块可以参与 forward、backward 和 checkpoint

### Requirement: 保留 DeepSense6G、AMBER-Full 和 RMBP-MM

DeepSense6G T2、AMBER-Full 和 RMBP-MM MUST 保持可配置、可构建和可训练/评估。它们使用共享四模态 batch contract，但不得依赖已删除的历史 owner。

#### Scenario: 无本地产物解析保留 recipe

- **WHEN** 在没有 outputs、cache 或 checkpoint 的环境加载保留 recipe
- **THEN** 配置解析和 synthetic model construction MUST 成功
