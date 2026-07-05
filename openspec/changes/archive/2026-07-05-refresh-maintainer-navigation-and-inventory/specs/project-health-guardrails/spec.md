## ADDED Requirements

### Requirement: 架构边界检查的脚本分类红点必须通过 inventory 修复
项目健康护栏 MUST 继续拒绝未分类 tracked `scripts/` Python 或 shell 文件。发现未分类脚本时，修复 MUST 更新 project surface inventory、删除脚本或迁移为正式 owner 入口，不得通过放宽测试或新增重复 allowlist 掩盖漂移。

#### Scenario: 未分类脚本失败
- **WHEN** `scripts/` 下存在 tracked `.py` 或 `.sh` 文件
- **THEN** 架构边界测试 MUST 能发现未在 project surface inventory 或等价 current 文档中登记的脚本
- **AND** 失败信息 MUST 指向缺失登记的相对路径

#### Scenario: 登记后检查恢复
- **WHEN** 未分类脚本被登记为 research diagnostic、dataset preparation、config generator、figure helper 或 local/manual helper
- **THEN** 架构边界测试 MUST 在不读取真实 `dataset/`、不启动训练、不写入 runtime artifacts 的情况下通过该分类检查
