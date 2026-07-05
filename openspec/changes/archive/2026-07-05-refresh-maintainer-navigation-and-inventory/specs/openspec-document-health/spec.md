## ADDED Requirements

### Requirement: Inventory 规模基线必须与真实仓库口径同步
项目表面积 inventory SHALL 记录当前源码、测试、脚本、配置和 OpenSpec 规模基线时说明统计口径，并在发现明显漂移时同步更新。规模数字 MUST 作为趋势和审计上下文，不得被解释为机械拆分、删除或放宽测试的唯一依据。

#### Scenario: 规模数字漂移被修复
- **WHEN** 维护者发现 inventory 中的 Python 文件数、配置数量、OpenSpec spec 数量或扫描日期明显落后于当前 tracked 文件系统
- **THEN** 本次文档健康修复 MUST 更新 inventory 的基线说明或改为更准确的可复核口径
- **AND** 更新 MUST 继续排除 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 和 ignored runtime artifacts

#### Scenario: 数字不替代右尺寸化判断
- **WHEN** inventory 记录某个源码、配置或 OpenSpec 数量
- **THEN** 文档 MUST 说明这些数字只是趋势信号
- **AND** 后续拆分、合并、保留或删除判断 MUST 继续依据 owner 职责、public surface、生命周期分类、调用边界和 focused validation
