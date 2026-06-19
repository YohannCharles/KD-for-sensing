## Why

CodeGraph 当前索引显示项目已有 359 个 Python 文件、3,420 个 function 节点、2,503 个 import 节点；本地 AST 复核也显示 `src/kd_sensing` alone 占 278 个 Python 文件、约 3,249 个函数和 2,195 条 import 语句。数量本身不是错误，但当前复杂度已经集中到少数长 orchestration、dataset/trainer/diagnostics owner、BeamBench Image AE+GPS 复现族和若干单 owner helper 边界上，需要用可验证的“右尺寸化”方案同时减少低价值碎片和防止热点继续膨胀。

本 change 的目标不是机械减少文件数，而是在保持公开 CLI/import、数据 split、指标、manifest schema、run metadata 和本地产物边界兼容的前提下，明确哪些模块该拆、哪些 helper 该合并、哪些大 owner 当前可接受，并把判断写进 OpenSpec、维护索引和架构边界测试。

## What Changes

- 建立项目级 source architecture right-sizing 基线：记录 CodeGraph/AST 统计口径、目录级复杂度、热点函数、公开 facade、right-size-accepted owner 和 merge-candidate 的判定标准。
- 将重构分成可回滚 wave：先补治理表和测试，再按 BeamBench Image AE+GPS、DeepSense6G/MMW dataset、trainer/evaluation、diagnostics benchmark、低价值 helper 合并逐步实施。
- 对应“该拆的拆”：继续拆 `DeepSense6GDataset.__init__` 的 resource/scaler/target setup、`trainer._train_inner` 的 epoch/checkpoint/finalization 边界、`image_ae_gps_paper_split.py` 的 scene/checkpoint/report payload 边界、`evaluation_pass.run_evaluation_pass` 的 metric/objective/prediction metadata 边界。
- 对应“该合并的合并”：允许同一 owner 下单调用点、只服务 re-export、无复用价值或仅为降低行数而产生的 helper 合并回 owner；禁止新增跨领域 `helpers.py`、兼容聚合层、旧根脚本或退役研究线入口。
- 保留合理大 owner：`jepa_benchmark_runner.py`、`jepa_benchmark_common.py`、`jepa_benchmark_scenario_d.py` 等已登记 `right-size-accepted` 的 owner 不因文件行数大而强制拆分，但必须保留 accepted rationale、focused tests 和防回流约束。
- 收紧 import 面：新增或移动 helper 后，内部代码必须直接依赖窄模块；公开 facade 只保留 thin owner/re-export/CLI 语义，不允许 suite-specific implementation 回流。
- 更新文档和测试护栏：同步 `docs/maintainer_context_index.yaml`、`docs/project_surface_inventory.md`、`docs/agent_navigation.md`、架构边界测试和必要 focused tests。
- 不包含 **BREAKING** 的公开行为变更：CLI 名称、console scripts、public import owner、配置路径、数据/输出边界和训练/评估数值语义必须保持兼容。

## Capabilities

### New Capabilities

- 无。该 change 不引入新的运行能力，而是收紧和扩展现有架构治理能力。

### Modified Capabilities

- `project-architecture`: 增加项目级右尺寸化要求，明确拆分、合并、保留大 owner、公开 facade、防回流、轻量导入和行为兼容的验收规则。
- `maintainer-context-index`: 增加/更新机器可读 architecture sizing baseline、remediation wave、merge-candidate、right-size-accepted、public surface policy、验证命令和 rollback note 的治理要求。

## Impact

- 主要影响 `src/kd_sensing/data/`、`src/kd_sensing/engine/`、`src/kd_sensing/diagnostics/`、`src/kd_sensing/baselines/beambench/`、`src/kd_sensing/cli/` 和对应 tests。
- 文档影响 `docs/maintainer_context_index.yaml`、`docs/project_surface_inventory.md`、`docs/agent_navigation.md` 和本 change 的 OpenSpec artifact。
- 验证影响 `tests/test_architecture_boundaries.py`、BeamBench focused tests、dataset modality tests、training IO tests、evaluation pass tests、JEPA benchmark tests 和 CLI help smoke。
- 不修改 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、历史本地产物或系统启动/认证配置。
