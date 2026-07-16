# canonical-config-resolution Specification

## Purpose

定义 MMW T2/baseline 的 tracked recipe 解析边界，保证干净 clone 只依赖版本控制中的输入，而不依赖任何本地训练输出或历史配置。

## Requirements

### Requirement: T2/baseline recipes 是唯一 canonical 配置面

配置加载 MUST 仅将 `configs/mmw/t2.yaml`、`s1.yaml`、`amber_full.yaml`、`rmbp_mm.yaml` 及其 tracked shared base 视为 current canonical inputs。

#### Scenario: 干净 clone 解析 recipe

- **WHEN** 用户加载任一 retained recipe
- **THEN** loader MUST 在没有 `outputs/`、checkpoint 和本地数据时完成 parse、normalization 与 validation
- **AND** 配置 MUST 声明 MMW 四模态协议

### Requirement: 不提供退役配置兼容

loader MUST 不从 output YAML、virtual matrix、旧路径或 migration guard 推导当前 recipe。

#### Scenario: 请求旧配置

- **WHEN** 用户请求不存在的退役 YAML
- **THEN** loader MUST 返回普通缺失文件或校验错误
- **AND** 不得映射到 retained recipe
