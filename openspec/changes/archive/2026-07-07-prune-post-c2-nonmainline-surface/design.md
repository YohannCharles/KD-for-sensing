## Context

当前仓库已经完成 final C2 消融入口规划，研究重心转为 U-MaskBeamJEPA / final C2 的缺失模态波束预测。历史上为了探索主线，仓库积累了 Image+GPS JEPA、BeamBench、BEV-Fusion 2604、Vision-Position、Scene31 多轮 sweep、RBMA/KD/BTAPA/weakKD overlay、一次性统计脚本和多个诊断 package CLI。它们现在会稀释维护注意力，也让 agent 和维护者更难判断哪些入口还能安全运行。

本 change 的关键约束来自用户补充：

- MMW 支线保留，后续仍会用这个数据集；MMW dataset、preparation、MMW GPS v2、physics-informed MMW、CSI hardening、相关 configs/tests/CLI 不进入删除范围。
- 主线会用到的 YAML/manifest 不动；尤其是 final C2、当前 Scene31/Scene31-34 evidence、claim registry、experiment matrix、OpenSpec current spec、focused tests 或用户标记仍需复跑的配置输入。
- U-MaskBeamJEPA 中已经存在、但当前未胜出的 fusion 分支先不动；本 change 不删 `pcpg`、`bprr`、`raw_conf_gate`、`weighted_sum`、`concat_mlp`、`supervised_router` 等实现分支，也不删相关 forward/loss 开关。

## Goals / Non-Goals

**Goals:**

- 给出可执行的 post-C2 表面积清理波次，先保护，再删除；每个删除候选必须有引用证据、替代入口和回滚方式。
- 从 current public surface 中移除非主线 package CLI、一次性研究脚本、历史 runbook、过期文档推荐命令和无 claim 价值的历史配置族。
- 保留 MMW 支线和主线证据链，同时把非主线保留项降级为 historical、supporting 或后续单独 change。
- 更新 OpenSpec、inventory、README/docs 和架构边界测试，使清理后不会通过兼容 wrapper 或 stale docs 回流。

**Non-Goals:**

- 不删除、移动或改写 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、TensorBoard event 或任何本地训练产物。
- 不删除 MMW/CSI/physics-informed MMW 相关源码、配置、CLI 或测试。
- 不修改 U-MaskBeamJEPA 的 fusion 分支实现、router 分支实现、loss 开关或默认训练语义。
- 不删除被 current claim/evidence、final C2、当前 experiment matrix、focused tests 或用户标记保护的 YAML/manifest。
- 不在删除后新增同名 alias、stub CLI、thin wrapper、virtual config 或兼容 facade。

## Decisions

### Decision 1: 先建立 protected inventory，再执行删除

实施前先生成并提交一份 scoped protected inventory，至少覆盖：

- `protected_mainline`: final C2 launcher/summary/tests、U-MaskBeamJEPA、U-Mask loss、PCPG radar balance helper、缺失模态 eval matrix、missing-modality statistics/stress、claim/evidence docs。
- `protected_mmw`: `src/kd_sensing/data/mmw/`、`src/kd_sensing/data/datasets/mmw*.py`、`src/kd_sensing/engine/mmw_town_gps_v2*.py`、`src/kd_sensing/models/*csi*`、`src/kd_sensing/models/physics/`、`configs/csi/`、`configs/fusion/csi_hardening_matrix/`、`configs/fusion/physics_informed_mmw*.yaml`、MMW/CSI tests 和 CLI。
- `protected_yaml_manifest`: 仍被 current docs/specs/tests/claim/final C2 使用的 YAML/CSV/JSON manifest。
- `protected_umask_branches`: U-MaskBeamJEPA fusion/forward/loss 分支实现。

替代方案是直接按目录删除。这个方案更快，但容易误删主线 manifest 或 MMW 支线，不符合用户约束。

### Decision 2: 删除按波次推进，每波都可回滚

建议波次如下：

1. Wave 0: 只读保护与引用扫描。输出 deletion candidate ledger，不删文件。
2. Wave 1: 文档和 public CLI 收口。先从 README/docs/pyproject/current CLI lifecycle 中移除非主线推荐入口。
3. Wave 2: 删除一次性 `scripts/` runbook 和 summary/analyze/diagnose 脚本，保留 final C2 与仍被 protected inventory 标记的脚本。
4. Wave 3: 删除非主线复现源码和 tests，例如 BeamBench、BEV-Fusion 2604、Vision-Position、Image+GPS JEPA 诊断等；MMW 与 U-Mask protected files 跳过。
5. Wave 4: 收缩 historical configs。仅删除不被 protected YAML/manifest 引用、可由 generator 重建、或只服务历史 sweep 的 YAML/manifest。
6. Wave 5: 更新 guardrails、inventory、claim/docs 和验证命令，确保 stale reference 和入口回流会被检测。

替代方案是一次性大删。它能最大化短期减行数，但失败定位很差，且与当前 dirty worktree 和 active/complete changes 更容易冲突。

### Decision 3: package CLI 只保留主线、MMW 和核心治理入口

默认保留：

- 核心：`kd-sensing-train`、`kd-sensing-evaluate`、`kd-sensing-preprocess`
- final C2 / 缺失模态主线直接需要的 evaluation/summary 入口
- MMW/CSI 相关入口：`kd-sensing-mmw-town-gps-v2`、`kd-sensing-inspect-mmw-physics` 等
- 治理：`kd-sensing-project-surface-doctor`、必要 paper/export 或 claim gate 入口

删除候选包括非主线 dashboard/preview/architecture summary/training throughput/dataset audit/source-audit 等 package CLI。删除前必须确认 current docs/specs/tests 不再引用，或先更新为 historical/supporting。

### Decision 4: configs 以“证据链保护”优先于“目录名删除”

删除配置时不按目录机械删除，而按引用和主线状态判断：

- 保留：final C2、current Scene31/Scene31-34 evidence、current claim/protocol、MMW/CSI、OpenSpec current specs、focused tests 或用户标记仍需的 YAML/manifest。
- 可删：只服务历史 sweep、旧 RBMA/KD/BTAPA/weakKD/tau/seed、可由 generator/template 无损重建且不作为 claim provenance 的实体 YAML。
- 暂缓：无法确认是否主线会用到的 YAML/manifest，登记为 `pending-user-confirmation` 或 `protected-until-next-audit`。

### Decision 5: U-MaskBeamJEPA fusion 分支只登记，不修改

本 change 只更新 inventory 和后续删除触发条件，不改 `u_mask_beam_jepa.py`、`u_mask_beam_jepa.py` loss helpers 中已存在 fusion 分支实现。后续如果要删输掉分支，应另开 change，并基于 final C2 完整结果、配置引用和测试覆盖单独判断。

## Risks / Trade-offs

- [Risk] 误删主线 YAML/manifest。→ Mitigation: Wave 0 必须生成 protected YAML/manifest ledger；删除 wave 前跑 stale config reference 和 focused tests。
- [Risk] MMW 未来数据集工作被破坏。→ Mitigation: MMW/CSI 路径进 protected inventory，并在 architecture guardrail 中加保留检查。
- [Risk] 删除 package CLI 后文档仍引用旧命令。→ Mitigation: 先做 docs/pyproject/CLI lifecycle 同步，再跑 `kd-sensing-project-surface-doctor --scope cli-surface`。
- [Risk] 一次性删除太多导致验证红点难定位。→ Mitigation: 每个 wave 单独提交或至少单独验证，失败只回滚当前 wave。
- [Risk] 历史报告中仍有有价值结论。→ Mitigation: 删除报告前把仍需保留的 conclusion/caveat 摘到 `docs/mainline_experiment_history.md`、`docs/result_claims_registry.md` 或 inventory historical note。

## Migration Plan

1. 运行 closeout/status 检查，确认 `final-c2-ablation-v1` 和 `overnight-branch-router-v2` 的状态，不在本 change 中改写它们的成果。
2. 生成 protected inventory 和 deletion candidate ledger，列出每个候选的当前引用、替代入口、验证命令和回滚方式。
3. 从文档和 CLI lifecycle 先移除非主线推荐入口，保留 historical note。
4. 按 Wave 2-4 删除 scripts、源码、tests 和 configs；每波结束运行 scoped validation。
5. 更新 README、docs、OpenSpec current specs、inventory、architecture boundary tests 和 project surface doctor expected surface。
6. 最终运行 `openspec validate --all --strict`、`make verify-quick`、`make verify-cli-config` 和 `make verify-compile`；若修改 MMW 周边，还追加 MMW focused tests。

Rollback 策略：每个 wave 的删除必须能通过 git 恢复对应文件；若 protected check 发现误删，立即恢复文件并把候选改为 `protected` 或 `pending-confirmation`。

## Open Questions

- 哪些具体 final C2 YAML/manifest 会被用户标记为长期主线输入？实现前需要从现有 final C2 launcher/manifest 和用户下一步运行计划中确认。
- 是否保留 `kd-sensing-paper-export` 和 research dashboard/preview 的治理用途？建议 paper export 暂保留，dashboard/preview 作为删除候选进入 Wave 1 证据审计。
- BeamBench/Image+GPS JEPA 是否完全退役为 historical？本方案按非主线删除候选处理；若后续论文仍要引用某条结果，应把对应 config/provenance 加入 protected inventory。
