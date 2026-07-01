## Why

ponytail 审计显示当前仓库的主要复杂度不在核心训练逻辑，而在一次性研究脚本、临时实验配置、未归档 OpenSpec change、重复测试 bootstrap 和退役/历史说明占据 current surface。现在需要把这些低价值表面收口，避免后续维护者把临时运行材料误读成长期入口。

## What Changes

- 收口本轮审计确认的临时表面：不将未跟踪的 Scene31/RBMA queue 脚本、fullrun/seed/strong-encoder YAML、固定 V0-V5 汇总脚本和根目录临时 runbook 纳入当前支持面。
- 将已完成的 `add-rbma-prototype-kd-missing-workflow` 作为本次清理的前置归档项，归档后再解释当前 specs、inventory 和工作树状态。
- 修复已确认的治理漂移：`configs/fusion/` 根目录真实 YAML 与 inventory 分类不一致、current specs 中仍有归档 `TBD` Purpose、`.codegraph/daemon.pid` 仍被 git 跟踪，以及内部诊断代码仍通过 `jepa_gps_shortcut_benchmark` facade 回流导入窄 helper。
- 对一次性 research diagnostic 脚本做分类处理：删除、移动为历史说明、或明确保留为 local/manual artifact；保留项必须有 owner、输入输出边界和删除触发条件。
- 清理普通 pytest 文件中重复的 `ROOT/SRC/sys.path.insert` 启动片段，依赖 `tests/conftest.py` 或隔离子进程 probe。
- 小步合并 U-Mask eval 专用指标/导出重复面，优先复用 `kd_sensing.evaluation.metrics` 和现有导出 helper，不引入新的通用抽象。
- 折叠只剩历史说明且无 current guard 价值的退役 tombstone spec 到集中 retired summary 或 archive；保留 guard 价值的墓碑继续留在 current specs。
- 更新 inventory、导航文档和架构边界测试，使新增脚本、配置和 root 文档必须被分类；未分类 current surface 继续失败。
- 不删除 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、`All_models/` 或真实运行产物；这类清理仍走 runtime cleanup/organize manifest。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `project-surface-cleanup`: 增加 ponytail 审计 findings 的清理波次、一次性脚本/临时配置/root runbook 收口规则、U-Mask eval 重复面合并边界和本地产物保护要求。
- `project-health-guardrails`: 增加未分类脚本/配置/root 文档检查、普通测试重复 bootstrap 检查、临时 queue/runbook 防回流检查、OpenSpec `TBD` Purpose 检查、facade 回流检查、CodeGraph 运行产物跟踪检查和 focused 验证要求。
- `spec-lifecycle-boundaries`: 增加“已完成 active change 必须先归档或明确阻塞”的收口规则，并定义退役 tombstone spec 折叠到集中 retired summary 的判定边界。

## Impact

- 代码：可能删除或合并少量 `scripts/`、`src/kd_sensing/eval/*` helper、测试 bootstrap 片段和对应 tests；不触碰核心训练 loop、dataset 读取、模型 forward 或 checkpoint schema。
- 配置：清理或拒绝纳入未分类临时 YAML；保留 current/canonical 和已登记 experiment configs。
- 文档/OpenSpec：更新 change artifact、inventory、导航文档、retired summary 和相关 spec delta。
- 测试：运行 OpenSpec strict validate、架构边界、配置加载、CLI help、U-Mask eval matrix 和受影响脚本/导出 focused tests。
- 依赖：默认不删第三方依赖；只有确认无 current 源码/测试/docs/OpenSpec 引用时才单独删除依赖声明。
