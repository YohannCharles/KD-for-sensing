## ADDED Requirements

### Requirement: 导航文档必须提示 Scene31 表面积检查
非平凡改动触碰 Scene31 local/manual workflow 时，维护导航 MUST 要求先检查 active/archived OpenSpec 状态、project surface inventory、真实 tracked YAML/runner 清单和最小验证命令。

#### Scenario: 修改 Scene31 workflow 前检查权威来源
- **WHEN** 维护者准备修改 `configs/scene31/` 或 `scripts/run_scene31_*.sh`
- **THEN** 导航文档 MUST 指向 `scene31-next-round-experiment-workflow` spec、project surface inventory 和 `tests/test_scene31_next_round.py`
- **AND** 文档 MUST 提醒不要把 generated YAML、ignored outputs 或 completed change 误当作 current source requirement
