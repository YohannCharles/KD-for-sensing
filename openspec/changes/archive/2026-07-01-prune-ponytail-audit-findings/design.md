## Context

当前项目已经完成多轮架构右尺寸化：旧 KD/HiST/BGAM/viewer 等路线被退役，普通 baseline 默认走 `modular_sequence`，公开 facade 和本地产物边界也已有 OpenSpec 与 inventory 约束。现在剩余成本主要来自支持面漂移，而不是核心训练逻辑本身。

本轮审计看到的具体漂移包括：

- `add-rbma-prototype-kd-missing-workflow` 已完成但仍 active，影响维护者判断当前 specs 和工作树状态。
- `configs/fusion/` 根目录真实 YAML 数量与 inventory 中的保留分类不一致，且新增 RBMA/Scene31/strong-encoder 实验配置需要明确 current/local/temporary 边界。
- `scripts/` 中出现未分类 queue/runbook/summary/diagnostic 脚本，容易被误读为长期入口。
- `src/kd_sensing/diagnostics/jepa_visual_analysis.py` 仍通过 `kd_sensing.diagnostics.jepa_gps_shortcut_benchmark` facade 导入 benchmark helper，违反“内部直连 owner 模块”的边界。
- current specs 中仍存在归档工具留下的 `TBD - created by archiving...` Purpose，现有 `openspec validate --all --strict` 不会发现。
- `.codegraph/daemon.pid` 是本机运行状态，却仍被 git 跟踪并随 daemon 启动变化。

## Goals / Non-Goals

**Goals:**

- 将本轮 ponytail 审计 findings 分类为删除、迁移、保留或后续 change，并把分类写入 inventory/guardrail。
- 收敛 `configs/fusion/` 根目录，只保留 canonical/current thin entry；实验配置放入明确子目录并更新引用。
- 收敛 `scripts/` 和临时 runbook，只保留已登记 owner、输入输出边界和 caveat 的脚本。
- 修复内部 facade 回流，新增或扩展架构边界测试防止再次通过公开 facade 导入窄 helper。
- 修复 current spec 的 `TBD` Purpose，并让健康检查能捕获归档脚手架。
- 将 `.codegraph/daemon.pid` 视为本地运行产物，移出源码跟踪并补充 ignore/guard。
- 不触碰核心训练 loop、dataset 读取、模型 forward、checkpoint schema 或真实运行产物。

**Non-Goals:**

- 不重新设计 U-MaskBeamJEPA、RBMA、trainer、dataset 或 JEPA benchmark 算法。
- 不删除 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、TensorBoard event 或 `All_models/` 权重。
- 不为了减少行数机械拆分 `models/modular.py`、`jepa_visual_analysis.py` 或 dataset 大 owner。
- 不新增新的通用脚本框架、配置生成器或跨领域 registry 抽象。

## Decisions

1. 沿用已有 change `prune-ponytail-audit-findings`，不新建重复 change。
   - 理由：该 change 的 proposal 已覆盖 ponytail 审计收口方向，补齐 design/spec/tasks 比新建并行治理 change 更少维护面。
   - 备选：新建 `tighten-project-surface-governance`。不采用，避免两个 active change 处理同一类表面漂移。

2. 先归档或明确处理已完成的 RBMA change，再做表面清理。
   - 理由：已完成 active change 会让 OpenSpec lifecycle、inventory 和工作树解释混在一起。
   - 备选：直接忽略已完成 change。不可取，因为 agent navigation 明确要求 active change 会影响当前状态判断。

3. 配置清理采用“分类优先，删除保守”的方式。
   - 根 `configs/fusion/*.yaml` 只保留 canonical/current thin entry。
   - RBMA/Scene31/strong-encoder/fullrun 等实验配置如果仍有当前本地实验价值，进入 `configs/fusion/experiments/<family>/` 并由 inventory 分类。
   - 无 current 文档、测试、OpenSpec 或脚本引用的临时配置才删除。
   - 备选：一次性删除所有未跟踪配置。不可取，可能误删用户正在跑的本地实验入口。

4. 脚本清理只保留有 owner 和边界的脚本。
   - `scripts/` 下新增 Python/shell 文件必须归为 research diagnostic、dataset preparation、figure helper、shell orchestration 或 local/manual artifact。
   - 当前 package CLI 仍是长期运行入口；训练/评估 runner 不通过 `scripts/` 兼容 alias 扩散。
   - 备选：把所有 queue shell 脚本迁入 package CLI。不可取，queue 编排通常是本地资源策略，不该成为长期 API。

5. facade 回流修复只改内部导入路径，不删除公开 facade。
   - `jepa_gps_shortcut_benchmark.py` 可继续服务 CLI 和外部兼容 import。
   - 内部模块如 `jepa_visual_analysis.py` 必须直连 `jepa_benchmark_common.py`、`jepa_benchmark_runner.py` 等 owner。
   - 备选：删除 facade。当前仍有 CLI/外部兼容价值，删除风险大于收益。

6. OpenSpec hygiene 通过 guardrail 补上，而不是依赖 `openspec validate`。
   - `openspec validate --all --strict` 验 schema，不验证项目语义质量。
   - 架构边界测试应扫描 current `openspec/specs/*/spec.md` 的 Purpose，拒绝 `TBD`、归档脚手架和空泛占位。

7. CodeGraph daemon 状态按本地产物处理。
   - `.codegraph/.gitignore` 继续保留索引数据库忽略规则，并新增 pid/socket 等运行状态忽略。
   - 已跟踪的 `.codegraph/daemon.pid` 应从 git index 移除。
   - 备选：提交最新 pid。不可取，pid/socket/startedAt 是本机状态。

## Risks / Trade-offs

- [Risk] 误删用户仍在运行的本地实验脚本或配置 -> 先分类、查引用、看 `git status --short`，只删除无 current 入口和无保留理由的项。
- [Risk] 移动配置导致已有文档命令失效 -> 同步更新 README/docs/OpenSpec/tests 中的配置路径，并运行配置加载 focused tests。
- [Risk] facade 回流测试过宽，误伤 CLI 兼容入口 -> 测试只限制内部模块，允许 `kd_sensing.cli.*` 和 facade 文件本身使用公开 facade。
- [Risk] 清理 `.codegraph/daemon.pid` 影响 CodeGraph 使用 -> 只移出 git 跟踪，不删除数据库、不停止 daemon；CodeGraph 可继续生成本地 pid/socket。
- [Risk] OpenSpec tombstone 折叠过度，丢失 migration guard 背景 -> 只有无 current guard 价值的 tombstone 才归档或集中说明，仍被 registry/config guard 消费的墓碑继续保留。

## Migration Plan

1. 确认 active changes：归档已完成的 `add-rbma-prototype-kd-missing-workflow`，或在本 change 中明确延后原因。
2. 固化清理清单：列出新增/未分类 scripts、root fusion YAML、experiment YAML、OpenSpec TBD specs、facade 回流和 tracked local tool state。
3. 先做无行为风险修复：OpenSpec Purpose、`.codegraph/daemon.pid` 跟踪边界、facade import 路径、guardrail tests。
4. 再做配置/脚本分类：移动、删除或保留并更新 inventory/引用。
5. 最后做低风险重复面合并：仅在引用清晰时处理 U-Mask eval/export 小重复；不触碰训练数值语义。
6. 每个 wave 后运行 focused tests，并检查 `git status --short` 不包含本地数据、输出、日志、cache 或 checkpoint。

## Open Questions

- `scripts/run_rbma_missing_workflow.py`、queue shell 和 summary 脚本是要保留为 local/manual artifact，还是迁入文档中的手工命令片段后删除？
- 新增 Scene31/M2Beam 单模态配置是否属于当前实验 family，还是只作为本地临时排障配置？
- 是否在本 change 中立即折叠无 guard 价值的 retired tombstone specs，还是先只补 Purpose/guardrail，把 tombstone 折叠留到单独 archive wave？
