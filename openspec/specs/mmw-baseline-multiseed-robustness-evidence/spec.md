# mmw-baseline-multiseed-robustness-evidence Specification

## Purpose

定义 T2/baseline 的 MMW 固定协议和多 seed 证据边界，确保四方法比较共享 recipe、split、mask identity、checkpoint 预算和可审计 provenance。

## Requirements

### Requirement: 四方法多随机种子公平训练

系统 MUST 支持在同一 MMW 15-domain、四传感器、40 epoch、固定 `last.pth` 协议下比较 T2、S1、AMBER-Full 和 RMBP-MM。每个方法 MUST 从 tracked `configs/mmw/` recipe 解析，seed MUST 不改变 split 或 sample inventory。

#### Scenario: launcher 生成 seed 配置

- **WHEN** launcher 为四方法生成 seed 配置
- **THEN** 输入 MUST 全部来自 tracked recipe
- **AND** 不得读取 `outputs/` YAML、checkpoint 或历史 final config

### Requirement: 固定 mask 与样本身份严格配对

all-weather 与 task-output evidence MUST 在比较行之间复用 split、sample identity、mask identity、metric 定义与 checkpoint epoch。

#### Scenario: 汇总结果

- **WHEN** summary 聚合比较行
- **THEN** 缺失或不一致的 identity MUST 使该比较失败
- **AND** development screening MUST 标记为非正式 claim evidence
