## ADDED Requirements

### Requirement: Cleanup apply 必须验证路径与 manifest 完整性
runtime cleanup apply MUST 在任何删除前验证 manifest schema/rules version、非空候选路径、project root、scan root、允许根 containment、路径状态和 Git tracked 状态。任何必要状态无法确认时 MUST fail closed。

#### Scenario: 空候选路径
- **WHEN** cleanup manifest 的候选缺少 `path`、为空或只包含空白
- **THEN** apply MUST 拒绝整个 manifest 或该候选
- **AND** MUST NOT 将其解析为当前目录

#### Scenario: 项目根或 scan root 候选
- **WHEN** 候选 resolved path 等于项目根、scan root、文件系统根或受保护数据根
- **THEN** apply MUST 拒绝删除
- **AND** report MUST 记录 protected-root 原因

#### Scenario: 路径逃逸
- **WHEN** 候选通过 `..`、绝对路径或符号链接解析到允许根之外
- **THEN** apply MUST 拒绝删除
- **AND** 允许根之外的文件 MUST 保持不变

#### Scenario: Git 状态不可用
- **WHEN** `git ls-files`、项目根解析或 tracked-state 检查失败
- **THEN** apply MUST 中止删除阶段
- **AND** 系统 MUST NOT 把 tracked 集合当作空集合继续

#### Scenario: Manifest 版本或状态漂移
- **WHEN** manifest schema/rules version 不受支持，或候选 size、mtime、type、protection 状态与扫描时不兼容
- **THEN** apply MUST 跳过或拒绝候选
- **AND** execution report MUST 记录具体不匹配字段

#### Scenario: 候选文件系统类型漂移
- **WHEN** scan manifest 记录候选为普通文件、目录、符号链接或其他类型，但 apply 前 `lstat` 类型发生变化
- **THEN** apply MUST 在路径解析和删除前拒绝该候选
- **AND** execution report MUST 同时记录扫描类型和当前类型
