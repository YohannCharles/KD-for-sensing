## ADDED Requirements

### Requirement: 架构边界测试必须右尺寸化
架构边界测试 MUST 验证长期结构事实，而不是复制完整维护索引、文档短语、脚本 allowlist 或 OpenSpec prose。测试 SHOULD 保持可读、可维护，并优先从权威来源直接读取事实：`pyproject.toml`、真实路径、OpenSpec lifecycle inventory、tracked files、AST/import probes 和小型 retired token 常量。

#### Scenario: 删除大型治理镜像
- **WHEN** `tests/test_architecture_boundaries.py` 维护与 pyproject、inventory、README 或 OpenSpec 重复的长 allowlist
- **THEN** 本 change MUST 删除该镜像或改为从权威来源直接推导
- **AND** 测试 MUST 不要求维护完整源码目录清单、完整 package CLI 数据库或完整 hotspot budget 表

#### Scenario: 保留结构性失败
- **WHEN** 当前 docs 或 specs 引用不存在的 current config、console script、module path 或 capability lifecycle
- **THEN** 架构边界测试 MUST 继续失败
- **AND** 失败信息 MUST 指向修正文档、恢复文件或更新 lifecycle 分类，而不是放宽测试

### Requirement: 文档 wording 检查必须避免误伤合法当前语境
retired-route wording guard MUST 只拒绝未加退役/历史/拒绝限定的当前推荐入口表达。对于 current JEPA、diagnostics、runtime cleanup、legacy output classification 或 migration guard 的合法语境，测试 MUST 允许出现 retired token。

#### Scenario: 合法历史说明不失败
- **WHEN** README、docs 或 current spec 在退役、历史、migration guard、防回流或 archive 语境中提到 HiST、KD、BGAM、viewer manifest、Raymobtime、CRAF/MARF/G2D 或 Multimodal-NF
- **THEN** 健康检查 MUST 不失败
- **AND** 只有把这些路线写成当前推荐入口、默认 workflow 或 active mainline 时才 MUST 失败

#### Scenario: 结果 caveat 检查保留
- **WHEN** 文档出现 mock、smoke、upper-bound、local substitute 或 historical ablation 数值
- **THEN** 健康检查 MUST 继续要求附近文本包含 caveat
- **AND** 测试 MUST 不依赖固定长句，而应检查结构性限定词或结果状态字段

### Requirement: 健康护栏变更必须有 focused 自检
重写健康护栏时 MUST 留下最小自检，证明它仍能拒绝三类关键回归：旧入口回流、tracked 本地产物进入源码、current path/config 引用失效。该自检 MUST 不读取真实数据、不启动训练、不写入运行产物。

#### Scenario: 架构边界 focused test
- **WHEN** 健康护栏重写完成
- **THEN** `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` MUST 通过
- **AND** 测试内容 MUST 覆盖 pyproject scripts、retired route guard、tracked artifact boundary 和 current config/path reference

#### Scenario: 本地 ignored cache 不影响测试
- **WHEN** 工作树存在 ignored `__pycache__`、`.pytest_cache`、`outputs/` 或 `logs/`
- **THEN** 常规架构边界测试 MUST 不因 ignored 文件存在而失败
- **AND** 若这些路径被 git 跟踪，测试 MUST 失败
