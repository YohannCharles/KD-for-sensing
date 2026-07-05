## Why

当前项目的 OpenSpec 规格本身是健康的，但维护导航和 project surface inventory 已经出现轻微漂移：架构边界测试发现 4 个已跟踪脚本未登记，inventory 中的规模基线也落后于真实仓库。若不收敛这些治理事实，后续 AI 对话容易把本地/未分类脚本误判为当前入口，或基于过期规模数字判断架构风险。

## What Changes

- 在 AI 维护导航顶部增加一屏摘要，直接说明当前主线、推荐入口、退役边界、必读文件和快速验证命令。
- 刷新 project surface inventory 的规模基线和配置数量，使其与当前 tracked 文件系统口径一致。
- 将未登记的 Scene31 / Scene31-34 论文表格、per-scene summary 和 final conclusion 脚本纳入脚本 lifecycle 分类，并明确其输出边界。
- 保持现有训练、评估、预处理、配置解析、模型 forward 和本地产物清理语义不变。
- 不迁移大段脚本逻辑到包内 owner；这类结构性迁移保留为后续独立 change。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `ai-maintainer-navigation`: 导航文档需要提供更短的当前项目摘要，减少 AI/维护者进入项目时的读取成本。
- `project-health-guardrails`: 架构边界检查必须继续发现未分类 tracked 脚本，并在 inventory 更新后恢复通过。
- `project-entrypoint-lifecycle`: Scene31 / Scene31-34 本地报告脚本必须有 lifecycle、owner 和输出边界，不作为 package CLI 或长期 public API。
- `openspec-document-health`: inventory 统计基线和 current surface 文档必须与真实仓库路径/数量一致，不能通过放宽测试掩盖漂移。

## Impact

- 影响文档和 OpenSpec artifact：`docs/agent_navigation.md`、`docs/project_surface_inventory.md`、本 change 下的 proposal/design/specs/tasks。
- 影响测试结果：`tests/test_architecture_boundaries.py` 当前的未分类脚本失败应在 inventory 更新后通过。
- 不新增依赖，不新增 console script，不修改训练/评估/预处理 runtime。
- 验证以 `openspec validate --all --strict`、`conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` 和 CLI/config smoke 为主。
