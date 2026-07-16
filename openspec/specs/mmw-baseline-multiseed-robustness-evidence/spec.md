# mmw-baseline-multiseed-robustness-evidence Specification

## Purpose

定义 T2/baseline 的 MMW 固定协议和多 seed 证据边界，确保四方法比较共享 recipe、split、mask identity、checkpoint 预算和可审计 provenance。

## Requirements

### Requirement: 四方法多随机种子公平训练

系统 MUST 在同一 MMW 15-domain、四传感器、domain-balanced sampler、缺失增强、40 epoch 和固定 `last.pth` 协议下比较 T2、S1、AMBER-Full、RMBP-MM。每个方法 MUST 从 tracked `configs/mmw/` recipe 或 shared base 解析；seed MUST 控制训练随机性，但 MUST NOT 改变 split 或样本 inventory。

#### Scenario: launcher 不读取历史 resolved config

- **WHEN** launcher 为任一四方法生成 seed 配置
- **THEN** 输入 MUST 全部来自 tracked `configs/mmw/`
- **AND** launcher MUST 不读取 `outputs/` YAML、checkpoint 或 final config

#### Scenario: seed 不改变验证样本

- **WHEN** launcher 分别生成同一方法的多个 seed
- **THEN** 它们 MUST 使用相同的 domain inventory、split 路径和固定数据 seed
- **AND** 训练、sampler 与 temporal missing 的行为 seed MUST 仅随实验 seed 改变

### Requirement: 固定 mask 与样本身份严格配对

all-weather 与 task-output evidence MUST 在比较行之间复用 split、sample identity、mask identity、metric 定义与 checkpoint epoch。

#### Scenario: 汇总结果

- **WHEN** summary 聚合比较行
- **THEN** 缺失或不一致的 identity MUST 使该比较失败
- **AND** development screening MUST 标记为非正式 claim evidence
