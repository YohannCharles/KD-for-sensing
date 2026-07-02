## ADDED Requirements

### Requirement: Ponytail 瘦身回流必须被健康护栏发现
项目健康检查 SHALL 覆盖 ponytail audit follow-up 中已经判定为低价值的回流类型，包括跟踪运行/清理产物、未登记脚本、内部 package facade 回流导入和过大的治理镜像 fixture。

#### Scenario: 跟踪清理产物被拒绝
- **WHEN** 新增或保留的跟踪文件表现为根目录清理 manifest、审计 manifest、生成的 package metadata 或其他可再生成运行产物
- **THEN** 健康护栏 MUST 失败，并指出该文件应删除、移入 ignored 本地产物位置或登记为真实源码资产

#### Scenario: 未登记脚本被拒绝
- **WHEN** `scripts/` 下新增脚本未出现在 current inventory、README/OpenSpec 当前工作流或明确脚本 allowlist 中
- **THEN** 健康护栏 MUST 失败，并要求删除、合并到现有入口或补齐登记依据

#### Scenario: 内部 facade 回流导入被拒绝
- **WHEN** 内部源码或测试重新通过已收缩的包级 facade 导入 owner helper
- **THEN** 健康护栏 MUST 失败，并提示改为直接导入 owner 模块；公开 CLI/API 兼容 facade 不应被误判

#### Scenario: 治理 fixture 不得镜像退役历史全集
- **WHEN** 测试 fixture 只为保活历史 removed guard 名称而复制大表，且没有当前迁移价值说明
- **THEN** 健康护栏 MUST 要求 fixture 收缩为当前行为断言或删除
