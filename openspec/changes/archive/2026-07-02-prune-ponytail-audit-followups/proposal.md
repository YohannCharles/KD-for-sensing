## Why

Ponytail 全仓审计发现了一批已经有明确替代路径的低价值表面：跟踪的历史清理清单、包级重导出 facade、低价值 removed registry guard、未登记的一次性脚本、根目录历史笔记，以及评估指标中的重复计算。现在清理它们可以减少维护面积，并把“哪些东西不该再进入源码树”的规则固化到 OpenSpec 和护栏里。

## What Changes

- 删除已跟踪的根目录历史清理清单 `legacy_knowledge_decoupling_cleanup_manifest.json`；如后续仍需生成清理清单，统一放入本地运行产物位置而不是源码根目录。
- 清理忽略的本地生成元数据 `src/kd_sensing.egg-info/`，并明确这类本地 artifact 不属于源码 diff。
- **BREAKING**：收缩低价值包级 `__init__.py` 重导出/lazy facade，将内部调用点改为直接导入 owner 模块；仅保留明确公共入口和轻量 marker。
- 精简 removed component registry guard：保留仍有迁移价值的名称，删除只服务历史测试夹具或旧实现变体的 removed guard 与对应 fixture 断言。
- 不接纳未登记的一次性脚本 `scripts/run_priority_v3_budget.sh`；确有可复用价值时，合并到现有实验调度入口或登记到 inventory 后再提交。
- 删除或归档根目录历史笔记 `知乎问答下载.md`；如仍有当前维护价值，只保留压缩后的说明到合适文档位置。
- 收缩 U-Mask Beam JEPA 缺失矩阵评估中的 top-k/DBA/MAE 重复计算，保持输出语义不变。
- 更新 inventory、测试或检查脚本，使后续新增低价值 facade、跟踪运行产物、未登记脚本时能被发现。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `project-surface-cleanup`：补充 ponytail 审计候选的删除/保留判据，覆盖历史根目录文档、清理清单、未登记脚本、本地生成元数据和包级 facade 的处理边界。
- `project-health-guardrails`：强化护栏对跟踪运行产物、未登记脚本、低价值 package barrel/facade、过大的治理镜像 fixture 的检测要求。
- `component-registry`：明确 removed guard 只保留有当前迁移价值的名称，删除低价值历史实现别名时测试不得依赖特殊 removed 文案。

## Impact

- 影响源码表面：`src/kd_sensing/**/__init__.py`、组件注册表、少量内部导入点、U-Mask Beam JEPA 评估指标路径。
- 影响仓库表面：根目录历史 JSON/Markdown、`scripts/` 中未登记脚本的保留策略、`docs/project_surface_inventory.md`。
- 影响测试：架构边界、组件注册表、相关 CLI/import smoke、缺失模态评估指标测试。
- 不新增运行依赖；不改动 `dataset/`、`outputs/`、`logs/`、checkpoint 或历史实验权重。
