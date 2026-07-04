## Why

当前项目表面积 inventory、Scene31 local/manual 实验面和真实源码状态已经出现漂移：`configs/scene31/` 中仍跟踪大量 generated YAML，`scripts/run_scene31_*.sh` 存在重复 runner 逻辑，而 inventory 中的数量、生命周期和删除边界已经落后。服务器升级期间适合先做这类不依赖真实训练的治理与表面积重构。

## What Changes

- 更新项目健康基线、inventory 和导航文档，使当前 Python/YAML/script 数量、Scene31 local/manual surface、active/archived change 状态保持一致。
- 将 Scene31 next-round、BC、funnel、magic overnight、beamsoft weak 等本地实验入口统一到 manifest-backed 生命周期：源码只保留 generator、manifest、base/template 和必要 local/manual runner。
- 收敛或复用 Scene31 shell runner 中重复的 GPU worker、skip/overwrite、fresh eval、failed list 和 summary 调用逻辑。
- 明确 generated YAML 的源码边界：可本地生成和运行，但不应长期作为源码表面积静默增长；需要保留的实体 YAML 必须有 current/local/manual 理由。
- 不改变 Scene31 已有 run name、seed、epoch、sampler/loss 字段、fresh eval 指标口径、checkpoint policy 或默认输出 root。

## Capabilities

### New Capabilities
- 无。

### Modified Capabilities
- `project-hotspot-governance`: 更新治理基线、remediation wave 和热点 inventory，覆盖当前 Scene31 表面积漂移。
- `project-entrypoint-lifecycle`: 明确 Scene31 local/manual runners、generator、summary 脚本和 shell launcher 的生命周期与收敛条件。
- `project-surface-cleanup`: 将 generated YAML、重复 shell runner 和本地实验 runbook 纳入可审计清理/保留分类。
- `scene31-next-round-experiment-workflow`: 统一 Scene31 next-round、BC、funnel、magic overnight 的 manifest-backed local/manual 边界。
- `ai-maintainer-navigation`: 更新非平凡改动前的导航规则，使维护者先检查 Scene31 lifecycle、inventory 和最小验证命令。

## Impact

- 影响文档与 OpenSpec：`docs/project_surface_inventory.md`、`docs/agent_navigation.md`、`openspec/specs/*` 中相关 capability。
- 影响配置与脚本：`configs/scene31/`、`scripts/generate_scene31_*.py`、`scripts/run_scene31_*.sh`、`scripts/summarize_scene31_*.py`、`scripts/select_missing_aware_checkpoint.py`。
- 影响测试：`tests/test_scene31_next_round.py`、`tests/test_architecture_boundaries.py`。
- 不读取真实 `dataset/`，不删除、移动或提交 `outputs/`、`logs/`、checkpoint 或 cache。
