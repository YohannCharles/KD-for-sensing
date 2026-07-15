## ADDED Requirements

### Requirement: MMW archive extraction 必须防止路径与资源逃逸
MMW archive preparation MUST 在删除或覆盖任何目标目录前完成 member 路径、数量、解压总大小、压缩比和完整 archive digest 校验。解压结果 MUST 先写入受控临时目录，再原子替换目标。

#### Scenario: ZIP member path traversal
- **WHEN** archive member 是绝对路径、包含 `..`，或 resolved destination 位于 extraction root 之外
- **THEN** preparation MUST 拒绝 archive
- **AND** 现有目标目录 MUST 不被删除或修改

#### Scenario: Archive 资源上限
- **WHEN** member 数、声明解压总大小、单文件大小或压缩比超过受控上限
- **THEN** preparation MUST 在写入 member 前失败
- **AND** error MUST 记录命中的上限类型

#### Scenario: 完整 digest 控制复用
- **WHEN** archive SHA256、算法版本或目标 inventory 与 extraction marker 不一致
- **THEN** runtime MUST 不复用旧 extraction
- **AND** MUST 在安全预检通过后重新生成受控 extraction

#### Scenario: 安全原子替换
- **WHEN** archive 全部 member 校验和临时解压成功
- **THEN** runtime MUST 原子发布新的 extraction root
- **AND** 任一失败 MUST 清理临时目录并保留原目标
