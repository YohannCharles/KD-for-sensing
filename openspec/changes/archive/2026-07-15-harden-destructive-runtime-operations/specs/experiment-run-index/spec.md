## ADDED Requirements

### Requirement: Run index 进程命令行必须脱敏
run index MUST 保留 `/proc` command line 的 argv 边界，并在写入公共 JSON、run summary 或 card 前统一脱敏。系统 MUST NOT 保存原始 credential、token、password、URI userinfo 或敏感 config override。

#### Scenario: 读取 proc cmdline
- **WHEN** run index 读取 `/proc/<pid>/cmdline`
- **THEN** parser MUST 按 NUL 分隔保留 argv list
- **AND** 含空格路径 MUST 保持为单个 argv

#### Scenario: 敏感参数脱敏
- **WHEN** argv 包含 token、password、secret、credential、URI userinfo 或敏感点式 override
- **THEN** public resources、summary 和 card MUST 只包含脱敏值
- **AND** raw cmdline MUST NOT 被写入 artifact

#### Scenario: 普通参数保持可诊断
- **WHEN** argv 不包含敏感值
- **THEN** public artifact MUST 保留可识别的 executable、config path 和非敏感参数
- **AND** 展示字符串 MUST 使用 argv-safe join 语义
