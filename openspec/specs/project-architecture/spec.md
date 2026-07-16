# project-architecture Specification

## Purpose

定义 T2/baseline 的最小 MMW 包边界，使配置、数据、模型、loss、训练、评估与预处理只保留四方法闭包所需的 owner 和导入关系。

## Requirements

### Requirement: 包导入图收敛到 T2/baseline 闭包

`kd_sensing` MUST 只保留 T2、S1、AMBER-Full、RMBP-MM 所需的数据、模型、loss、训练、评估、预处理和通用 owner。共享 owner MUST 不无条件导入 retired family。

#### Scenario: 导入 core surface

- **WHEN** 用户导入 config、registry 或训练入口
- **THEN** 导入 MUST 不读取数据、权重或输出目录
- **AND** 四方法的 config load 与 synthetic forward MUST 不要求 retired module

### Requirement: 本地产物与源码分离

`dataset/`、`outputs/`、logs、cache 与新 checkpoint MUST 是本地边界，不得成为 tracked source、canonical config 或 package import 的依赖。

#### Scenario: 干净 clone

- **WHEN** 维护者在无本地产物的 clone 加载 retained recipe
- **THEN** 解析和 package import MUST 成功
- **AND** 不得依赖历史 resolved output
