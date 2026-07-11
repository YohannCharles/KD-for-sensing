## REMOVED Requirements

### Requirement: 研究运行 happy path
**Reason**: Preview CLI 只是 OpenSpec、doctor、run index 和 dashboard 的再包装，其中 doctor/dashboard 同时退役。
**Migration**: 直接运行 README/AGENTS 中的 focused validation 和显式训练命令。

#### Scenario: Preview command 删除
- **WHEN** 用户查看 package CLI
- **THEN** `kd-sensing-research-preview` MUST 不存在
- **AND** current docs MUST 不依赖该命令

### Requirement: 静态 evidence preview QA
**Reason**: HTML dashboard 退役；paper table/CSV/claim validation 由各自 retained owner 测试覆盖。
**Migration**: 使用 paper export、Scene31-34 final analysis 和 claim docs 的 focused tests。

#### Scenario: Evidence QA 留在 owner
- **WHEN** retained owner 输出 paper artifact
- **THEN** 该 owner 的 focused test MUST 验证 schema/caveat
- **AND** 不要求通用 preview QA

### Requirement: 实验预算 manifest
**Reason**: 通用预算 manifest 没有 runtime consumer，长跑参数已由 launcher dry-run/manifest 记录。
**Migration**: 使用具体 launcher 的 `--dry-run`、config manifest、输出 root 和停止条件。

#### Scenario: 长跑使用领域 launcher
- **WHEN** 用户准备启动多 seed 或长时间训练
- **THEN** MUST 先运行对应领域 launcher dry-run 或检查 config manifest
- **AND** 不要求 research preview budget schema

### Requirement: Run recipe 和环境 fallback
**Reason**: 环境与 CLI fallback 已由 README、AGENTS 和 package CLI help 管理。
**Migration**: 使用 `conda run -n kd_mm_beam ...` 和对应 `python -m` 诊断路径。

#### Scenario: 环境说明仍可用
- **WHEN** console script 安装异常
- **THEN** 文档 MAY 提供包内诊断命令
- **AND** 不需要 preview owner

### Requirement: Research run preview 验证
**Reason**: Preview capability 整体退役，不再维护专属测试。
**Migration**: 使用 architecture、CLI/config、compile 和 retained owner focused tests。

#### Scenario: Preview tests 删除
- **WHEN** capability implementation 删除
- **THEN** `tests/test_research_run_preview.py` MUST 不再存在
- **AND** quick verification MUST 不引用它
