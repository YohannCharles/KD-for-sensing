# project-architecture Specification

## Purpose

定义 MMW T2/baseline、受限 DeepSense6G T2、BCACL U2 与 CMSBL 的最小包边界。

## Requirements

### Requirement: 包导入图收敛到 CMSBL 主线闭包

`kd_sensing` MUST 只保留 T2、S1、AMBER-Full、RMBP-MM、BCACL U2、CMSBL 与双数据集所需 owner。共享 owner MUST 不导入 PCER、PGCD、候选 Router、PR-SQDF、feature/prototype quick search、availability fallback、BT-SCL 或其他 retired family。

#### Scenario: 导入 core surface

- **WHEN** 用户导入 config、registry 或训练入口
- **THEN** 导入 MUST 不读取数据、权重或输出目录
- **AND** config load 和 synthetic forward MUST 不要求 retired module、outputs 或 cache

### Requirement: 主线收口不得修改本地产物

源码删除和 OpenSpec 归档 MUST 不删除、移动或改写 `dataset/`、`outputs/`、`outputs/cache/`、logs 或 checkpoint。

#### Scenario: 删除 retired source

- **WHEN** 维护者执行主线收口
- **THEN** 本地产物路径与内容 MUST 保持不变

### Requirement: 本地产物与源码分离

`dataset/`、`outputs/`、logs、cache 与新 checkpoint MUST 是本地边界，不得成为 tracked source、canonical config 或 package import 的依赖。

#### Scenario: 干净 clone

- **WHEN** 维护者在无本地产物的 clone 加载 retained recipe
- **THEN** 解析和 package import MUST 成功
- **AND** 不得依赖历史 resolved output
