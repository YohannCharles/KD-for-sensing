## Why

当前仓库已经通过 `.gitignore` 和架构边界测试避免把 `dataset/`、`outputs/`、`logs/`、cache 和 checkpoint 纳入源码，但本地运行产物仍缺少统一的可审计清理流程。近期盘点发现 `outputs/` 约 5.0G、其中 checkpoint 约 4.7G，`outputs/other/` 单独约 1.2G 且语义不清，说明需要在继续实验前收口输出表面积和 checkpoint 生命周期。

## What Changes

- 新增只读清理候选扫描能力：先生成 machine-readable manifest，记录候选路径、大小、类型、匹配原因、最近更新时间和保护状态，不直接删除。
- 新增受保护路径和产物类型规则：默认保护 `dataset/`、`All_models/`、源码、OpenSpec artifacts、配置、文档、已跟踪文件和未明确匹配的活跃实验。
- 收敛短生命周期产物：将 `_debug`、`_plan_check`、Python bytecode、pytest cache、空退役 spec 目录、个人备份压缩包等归入低风险清理候选类别。
- 收敛 MMW shell orchestration 默认输出路径：不再默认写入 `outputs/other/`，改为实验族专属目录，并更新帮助文本、inventory 和测试 allowlist。
- 增加 checkpoint 保留策略：区分复现必需 checkpoint、可恢复 checkpoint、临时 last checkpoint 和 sidecar metadata；支持按策略生成可删除候选。
- 扩展运行索引：在只读 run index 中暴露产物大小、checkpoint 摘要、stale/complete/failed/partial 状态和清理候选理由，供清理 manifest 复用。
- 更新文档和架构测试：记录产物边界、清理流程、默认输出命名约定和禁止回流的临时入口/目录。

## Capabilities

### New Capabilities

- `runtime-artifact-cleanup`: 定义本地运行产物清理 manifest、保护规则、候选分类和删除前审计契约。

### Modified Capabilities

- `project-architecture`: 强化源码/本地产物边界，要求清理流程必须先生成 manifest，并收口语义不清的默认输出目录。
- `experiment-run-index`: 扩展只读运行索引，为清理候选扫描提供大小、状态、checkpoint 和日志关联摘要。
- `experiment-artifact-registry`: 明确 checkpoint 保留和删除候选策略，避免 `last.pth`、重复 probe checkpoint 和临时训练产物无限累积。

## Impact

- 影响代码：`src/kd_sensing/diagnostics/run_index.py`、新增或扩展清理 manifest helper、相关 CLI、`scripts/run_mmw_sunny_modal15_l5p*.sh`、架构边界测试。
- 影响文档：`docs/project_surface_inventory.md`、README 或扩展指南中的本地产物边界说明、相关 OpenSpec specs。
- 影响本地运行产物：只生成候选清单，不自动删除 `dataset/`、`All_models/`、源码、OpenSpec、配置、文档或未匹配的实验目录。
- 影响工作流：用户执行清理必须先查看 manifest；真正删除操作需要显式确认或显式命令参数。
