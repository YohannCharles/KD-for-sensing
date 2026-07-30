# U0 Mainline Specification

## Purpose

定义稳定 MMW U0、AMBER-Full、RMBP-MM、DeepSense6G baseline，以及 trajectory 与 CSI/TSPC 隔离本地研究路线的最小运行闭包和公共边界。

## Requirements

### Requirement: MMW current surface 包含稳定 canonical baseline 与限定本地研究面

MMW canonical recipes MUST 仅为 `configs/mmw/u0.yaml`、`amber_full.yaml` 和 `rmbp_mm.yaml`，共享 `_base.yaml`。MMW public launcher、evaluator 和 summary MUST 仅接受 `U0`、`amber_full`、`rmbp_mm`。trajectory-disjoint 的 M0--M4 与因果消融 MUST 保持为本地研究工具，不得成为 canonical recipe 或 public CLI route。

#### Scenario: 加载 MMW canonical recipe

- **WHEN** 用户加载任一 MMW canonical recipe
- **THEN** 配置 MUST 不包含已退役路线或历史实验字段

#### Scenario: 运行 trajectory baseline

- **WHEN** 本地 trajectory-disjoint runner 接受方法名
- **THEN** 只允许协议声明的固定 M0--M4 与因果消融
- **AND** public CLI 与 canonical recipe MUST 保持不变

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

### Requirement: CSI/TSPC 作为隔离的本地研究基础保留

CSI、稀疏导频、Radio、TSPC/TSPC-V2 与相关 baseline MAY 保留为本地研究面，但 MUST 不扩展 public CLI 或 canonical recipe。它们 MUST 只使用过去帧 sensing/CSI、train-only codebook/cache、共享 beam prototype、完整 RE/window 核算和封存 test，并保持 Full 与 CSI-off 的硬旁路。

#### Scenario: 运行 CSI/TSPC 本地实验

- **WHEN** 本地工具构建 CSI/TSPC batch 或加载 cache
- **THEN** future channel、outer test 和 validation-fitted state MUST 不进入训练或推理
- **AND** source identity、trajectory split 与 cache provenance MUST 可审计

### Requirement: 旧 checkpoint route 必须 fail closed

评估 MUST 拒绝 metadata 中带有已退役训练路线标识的 checkpoint，避免历史状态被误当作 current U0 证据。

#### Scenario: 评估旧 checkpoint

- **WHEN** checkpoint metadata 声明已退役训练 route
- **THEN** 评估 MUST 在加载模型参数前失败
