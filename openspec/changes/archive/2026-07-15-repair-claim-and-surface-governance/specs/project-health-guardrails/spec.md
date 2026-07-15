## ADDED Requirements

### Requirement: Full 与 compile verification 必须覆盖真实 owner surface
`verify-full` MUST 执行全量 pytest；script/package compile 与 lifecycle guard MUST 扫描受控 owner roots 的 on-disk Python/source entrypoints，而不是只读取 Git tracked 列表。扫描 MUST 排除 dataset、outputs、logs、cache、checkpoint 和其它本地产物。

#### Scenario: 运行 verify-full
- **WHEN** 开发者运行 `make verify-full`
- **THEN** quick、CLI/config、compile 和 `conda run -n kd_mm_beam pytest -q` MUST 全部执行
- **AND** 任一阶段失败 MUST 使命令非零退出

#### Scenario: 未跟踪 owner script 语法错误
- **WHEN** `scripts/` 中存在 on-disk 未跟踪 Python 文件且语法非法
- **THEN** compile verification MUST 失败并报告路径

#### Scenario: 本地产物不被扫描
- **WHEN** outputs、logs、dataset 或 cache 中存在 Python/Markdown 运行产物
- **THEN** source lifecycle/compile guard MUST 不把它们当作源码入口

### Requirement: 最小 CI 必须复用仓库验证入口
项目 MUST 提供最小 CI，在声明的 Python/conda 环境中安装当前 package，并复用 OpenSpec strict、quick、CLI/config、compile 和 full test 入口。CI 文档 MUST 与实际 workflow 一致。

#### Scenario: CI workflow 存在
- **WHEN** 维护者检查 CI 配置和环境文档
- **THEN** workflow MUST 使用仓库现有验证入口而非复制测试清单
- **AND** 文档 MUST 不声称不存在的 CI、coverage、lint 或 type gate 已启用
