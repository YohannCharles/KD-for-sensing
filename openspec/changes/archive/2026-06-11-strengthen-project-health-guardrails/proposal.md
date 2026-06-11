## Why

当前项目已经进入多数据集、多 CLI、多实验矩阵并行维护阶段：`src/kd_sensing` 约 6.2 万行、测试约 1.56 万行、配置 YAML 超过 100 个，且仍有活跃 JEPA/BGAM/CSI/Raymobtime 等研究线持续演进。现有架构边界测试已经有效防止退役入口回流，但对“下一批热点模块继续膨胀”“测试启动逻辑重复”“实验配置/文档支持面漂移”这些日常维护成本还缺少统一、可重复的健康护栏。

本 change 目标是在不改变训练数值语义、不移动本地数据/输出、不打断当前活跃 `add-gps-query-jepa-pooling` change 的前提下，补上一层项目健康 guardrail：让未来优化不依赖人工记忆，能通过 focused 检查快速暴露结构回流、配置漂移和测试基础设施退化。

## What Changes

- 新增项目健康护栏契约，覆盖维护性热点、测试基础设施、配置/文档支持面和快速健康检查命令。
- 将本次审视发现的第二批热点纳入可执行优化队列：`DeepSense6GDataset`、`MMWDataset`、BeamBench Image AE+GPS workflow、`engine.trainer._train_inner`、MMW/GPS/BGAM orchestration、manifest builder、`diagnostics.run_index`、`engine.evaluation_pass`、`engine.batch` 等。
- 为热点治理定义轻量静态指标和人工例外机制，例如超长函数/类清单、facade 行数预算、禁止内部代码从兼容 facade 回流导入 helper、以及新增热点必须更新 inventory。
- 收敛测试基础设施：将重复的 `sys.path.insert` 测试启动逻辑集中到 shared pytest bootstrap，并在 `pyproject.toml` 或等价 pytest 配置中记录默认 testpath、markers 和 warning 约束。
- 扩展架构边界测试，使其能检查实验配置 inventory、公开入口 allowlist、root 文档/复现文档边界、以及当前推荐健康检查命令是否与 README/OpenSpec 保持一致。
- 记录分层验证命令：OpenSpec strict validate、架构边界、CLI help、配置加载 characterization、以及触碰训练/数据/诊断时的 focused tests。
- 不引入新的训练入口、不恢复 KD/HiST/Top8/residual 退役路线、不删除或迁移 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 或历史权重。

## Capabilities

### New Capabilities

- `project-health-guardrails`: 定义项目级健康护栏，包括维护性热点预算、测试基础设施收敛、配置/文档漂移检查和分层健康检查 workflow。

### Modified Capabilities

- `project-architecture`: 扩展架构契约，要求热点拆分 inventory、架构边界测试和共享测试 bootstrap 能共同约束当前支持面。
- `project-surface-cleanup`: 扩展支持面清理契约，使配置/脚本/文档 inventory 不只防止退役入口回流，也能约束实验子目录和 root 文档漂移。

## Impact

- 影响代码/测试：预计新增或修改 `tests/conftest.py`、`tests/test_architecture_boundaries.py`、必要的 focused static-hygiene helper，以及少量测试文件移除重复 bootstrap。
- 影响配置/工具：预计在 `pyproject.toml` 或等价 pytest 配置中补充 pytest 基础设置；不新增 mandatory runtime dependency。
- 影响文档/OpenSpec：更新 `docs/project_surface_inventory.md`、README 健康检查段落和本 change 的 spec delta，记录本次审视出的优先优化队列。
- 影响验证：需要运行 `openspec validate strengthen-project-health-guardrails --strict`、`conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`、`conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_config_load_characterization.py -q`，并视实际触碰范围追加 focused tests。
