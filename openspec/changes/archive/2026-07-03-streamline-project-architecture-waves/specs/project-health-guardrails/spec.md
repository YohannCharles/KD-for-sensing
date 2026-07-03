## ADDED Requirements

### Requirement: Architecture boundary tests remain right-sized during streamlining
架构边界测试 MUST 在本 change 中继续验证结构事实和高风险回归，但不得复制完整源码目录清单、完整 OpenSpec prose、完整 scripts allowlist、完整 config 数据库或完整 hotspot budget 表。大型事实以 inventory、current specs、pyproject、真实 tracked paths 和 focused tests 为权威。

#### Scenario: 保留结构性失败
- **WHEN** current docs/specs 引用不存在的 config、console script、module path、public owner 或 lifecycle 分类
- **THEN** architecture boundary tests MUST 失败并指向修正文档、恢复文件或更新 lifecycle 分类
- **AND** 测试 MUST 不通过扩大阈值掩盖真实漂移

#### Scenario: 删除重复治理镜像
- **WHEN** 架构边界测试维护与 inventory、pyproject 或 OpenSpec 重复的大型 allowlist
- **THEN** 本 change MUST 删除该镜像或改为从权威来源直接推导
- **AND** 测试 MUST 仍覆盖旧入口回流、tracked runtime artifact、重依赖 barrel、facade 回流和 current path 引用失效

### Requirement: Streamlining waves have layered validation
每个 streamlining wave MUST 有分层验证：OpenSpec strict、architecture boundaries、目标领域 focused tests、公开 CLI/help smoke 或 import smoke。所有项目相关 Python 验证 MUST 使用 `conda run -n kd_mm_beam ...`。

#### Scenario: Wave focused validation
- **WHEN** wave 触碰 dataset、trainer/evaluation、model forward、diagnostics、config/scripts/import surface 或 docs/specs guardrail
- **THEN** tasks MUST 列出对应 focused test 命令
- **AND** wave 完成说明 MUST 记录实际运行结果、未运行原因和剩余风险

#### Scenario: Final regression
- **WHEN** 所有 waves 完成
- **THEN** 开发者 MUST 运行 `openspec validate streamline-project-architecture-waves --strict`、`openspec validate --all --strict` 和 `conda run -n kd_mm_beam pytest -q`
- **AND** 若全量 pytest 因环境或本地数据缺失无法完成，最终说明 MUST 列出替代 focused 验证

### Requirement: Guardrails reject mixed runtime artifacts
本 change 的健康护栏 MUST 继续拒绝 tracked runtime artifacts，并 MUST 允许 ignored cache 噪声不影响常规测试。实施不得把 `dataset/` 真实数据、`outputs/`、`logs/`、cache、checkpoint、TensorBoard event、`.pytest_cache` 或 `__pycache__` 纳入源码变更。

#### Scenario: Tracked artifact failure
- **WHEN** git tracked files 包含 `__pycache__`、`.pyc`、`.pytest_cache`、`outputs/`、`logs/`、cache、checkpoint、TensorBoard event 或非允许权重文件
- **THEN** architecture boundary 或 surface guard MUST 失败
- **AND** 未跟踪/ignored 的同类本地产物 MUST 不驱动常规架构边界测试失败

