## ADDED Requirements

### Requirement: Scene31 表面积漂移必须纳入热点治理
项目 MUST 将 Scene31 local/manual YAML、generator、runner、summary 和 checkpoint selection 工具纳入热点 inventory，并记录当前 tracked 数量、owner、生命周期、输出边界和收敛条件。

#### Scenario: inventory 记录当前 Scene31 表面积
- **WHEN** 开发者审阅项目表面积 inventory
- **THEN** inventory MUST 记录 `configs/scene31/` 中 tracked YAML、manifest、template、generator、shell runner 和 summary 脚本的当前分类
- **AND** inventory MUST 区分长期保留样例、manifest-backed local/manual 输入和应本地生成的 YAML

#### Scenario: 未登记 Scene31 表面积扩张被拒绝
- **WHEN** 新增 Scene31 YAML、shell runner 或 summary/generator 脚本
- **THEN** 架构边界检查 MUST 要求更新 inventory、删除重复入口或登记 local/manual 保留理由
