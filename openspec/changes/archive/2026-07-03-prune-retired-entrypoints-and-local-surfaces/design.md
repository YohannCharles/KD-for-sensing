## Context

当前仓库没有 active change，但工作树已有未提交的 `streamline-project-architecture-waves` 归档、OpenSpec/spec 文档更新，以及 dataset/runtime/model 的重构改动。本 change 不回退这些改动，也不继续扩大它们的源码重构范围；它只收缩仍然暴露在项目表面的旧入口、隐藏 CLI、本地脚本、可生成配置和重复 tombstone。

现状判断：

- `pyproject.toml` 已删除 HiST/BGAM/viewer/AMR-Net_gps_image/JEPA-MSAC 等旧 console scripts。
- `configs/` 中旧实体配置基本已经删除，旧路径通过 config migration guard fail-fast。
- 旧路线仍大量出现在 current specs、docs、测试和 inventory 中，且部分 module-only CLI / 本地脚本仍像隐藏入口。
- 架构边界测试当前健康，但部分检查通过维护较大的删除清单和 tombstone 语境来实现，长期会继续膨胀。

## Goals / Non-Goals

**Goals:**

- 让 public entry surface 只剩清晰入口：`pyproject.toml` console scripts、README/current docs 推荐入口、current spec 明确保护 owner 和 registry/config 构建入口。
- 删除或转正 module-only CLI，避免 `python -m kd_sensing.cli.*` 成为隐藏 public API。
- 把退役路线从“每条路线一份 tombstone spec + 专用测试”收缩为集中 retired-route summary、迁移 guard 和参数化防回流测试。
- 删除低价值本地脚本、固定 GPU queue 和已有 package CLI 覆盖的 helper；保留必须的 dataset preparation、config generator 和少数 current research diagnostic。
- 将可生成实验 YAML 收敛为 manifest/generator/base config；删除可无损生成且无 current/reproduction/diagnostic 价值的实体 YAML。
- 保持当前用户可见 CLI、current config、metric/checkpoint/schema 和本地产物边界兼容。

**Non-Goals:**

- 不新增算法、模型、训练能力或结果 claim。
- 不改变当前 training/evaluation 数学语义、dataset split、beam label/label-space、metric schema 或 checkpoint schema。
- 不删除真实 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、TensorBoard event 或历史本地产物。
- 不把当前已有 dataset/runtime/model 重构重做一遍；本 change 只在必要时更新入口引用和测试。
- 不用兼容 wrapper 维持被删除的内部路径；未登记 public surface 的路径允许 breaking。

## Decisions

### Decision 1: Public surface 采用白名单，internal surface 默认可删

Public surface 只包括：

```text
pyproject console script
README/current docs 明确推荐入口
current OpenSpec 明确保护的 owner
registry/config 构建入口
focused tests 明确保护的 public import
```

其它 module-only CLI、package facade、thin wrapper、单调用点 helper 和 local/manual script 都默认 internal。Internal path 删除后不新增 compatibility wrapper，调用方改用真实 owner。

替代方案是继续给 module-only CLI 和 local/manual script 做 lifecycle 说明。该方案文档负担低于实质删除，不能减少入口数量，因此拒绝。

### Decision 2: Retired routes 集中 guard，不保留专用墓碑森林

将 JEPA-MSAC、AMR-Net_gps_image、HiST/Hist、BGAM、viewer manifest、Raymobtime、legacy KD、GPS residual、Top8 selector 等 retired route 收敛为：

- 一份集中 retired-route summary/spec 或现有 lifecycle spec 中的集中 requirement。
- 一个 machine-readable retired token/source 清单。
- 一个参数化 `test_retired_routes.py` 或架构边界测试段。
- `config/migration_guards.py` 中少量 fail-fast guard。

删除条件：若某 tombstone spec 只重复“已退役/不得回流”，且没有独立 current guard 价值，则折叠。若它仍保护复杂迁移语义，可先改成集中 summary 的一行并保留测试覆盖。

替代方案是逐个保留 tombstone spec。该方案最安全但持续膨胀；本轮用户已允许重大重构，因此选择集中 guard。

### Decision 3: Hidden CLI 必须转正或删除

`src/kd_sensing/cli/*.py` 中不在 `pyproject.toml` 的 module-only CLI 不再默认保留。处理规则：

- 若 README/current spec 推荐它，给它增加 console script 并补 `--help` smoke。
- 若只是历史 mock、local fallback 或 owner module 的薄转发，删除 CLI 文件，调用方改用 package console script 或 owner module。
- `cli/common.py` 这类 shared parser/config helper 不是入口，可保留。

### Decision 4: Scripts 只保留三类

保留：

- `dataset_preparation`: 仍需直接操作本地数据准备且没有 package CLI 覆盖的脚本。
- `config_generator`: 生成 manifest-backed config family，并有 sanity test。
- `research_diagnostic`: 当前 claim/protocol 仍引用、只读本地 outputs、输出 ignored 路径的少量诊断脚本。

删除：

- 固定 GPU queue / local runbook shell。
- 已有 package CLI 覆盖的 thin alias。
- 只服务历史 one-off 分析、无 current result registry 引用、无测试的脚本。

### Decision 5: Config 实体 YAML 只保留不可生成或 current 可读性有价值的

实体 YAML 保留条件：

- canonical/current quickstart。
- paper/workflow reproduction。
- diagnostics manifest。
- local/manual 但需要人工审查、generator 无法无损表达。

其它规则化 sweep、night-grid、seed matrix、next-round queue 优先保留 base + manifest + generator。删除实体 YAML 前必须有 sanity test 覆盖 run name、seed、epoch、sampler、loss weights、missing pattern、output boundary 和 retired path 不回流。

### Decision 6: Guardrail 测试验证事实，不复制目录

架构边界测试继续验证：

- retired console scripts 不存在。
- current paths 真实存在。
- tracked runtime artifacts 不进入源码。
- package marker 不恢复 eager barrel。
- scripts/configs 与 inventory 分类一致，或由 generator manifest 推导。
- retired route token 只出现在 retired/guard/history 语境。

测试不维护完整 OpenSpec prose、完整 scripts allowlist、完整 hotspot budget 或每条 tombstone 的专用文件清单。

## Risks / Trade-offs

- 删除 module-only CLI 破坏少数人工调用习惯 -> 只删除未登记 public surface；README/current docs 推荐的入口必须转正为 console script。
- Tombstone 折叠过度导致旧路线回流 -> 保留集中 retired-route guard、migration guard 和参数化测试。
- 删除本地脚本影响未提交实验复盘 -> 先检查 docs/result registry 和 inventory 引用；只删无 current 引用或已有替代入口的脚本。
- 删除生成型 YAML 影响复现实验队列 -> 先补 generator sanity test 和 manifest；不能无损生成的 YAML 暂保留。
- 当前脏工作树干扰归因 -> 实施前后记录 `git status --short`，不回退已有 dataset/runtime/model 改动。

## Migration Plan

1. 捕获当前状态：`git status --short`、OpenSpec status、已有验证结果。
2. 精简 hidden CLI：列出 module-only CLI，删除低价值 wrapper 或转正 current 入口。
3. 收缩 retired-route specs/tests：新增集中 retired-route guard，合并 JEPA-MSAC/AMR-gps-image 等专用测试，折叠低价值 tombstone specs。
4. 精简 scripts：删除固定 GPU queue、历史 one-off 和已有 package CLI 覆盖脚本；更新 inventory 分类。
5. 精简 configs：将可生成配置收敛到 base/manifest/generator，删除无 current 价值实体 YAML，更新 tests。
6. 更新 README、docs、maintainer index 和 architecture boundary tests。
7. 验证：`openspec validate prune-retired-entrypoints-and-local-surfaces --strict`、`openspec validate --all --strict`、`conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py tests/test_config_load_characterization.py tests/test_cli_help.py -q`，必要时运行相关 generator/script focused tests。

Rollback 以删除批次为单位：若某类删除破坏 current workflow，恢复该批文件并把它们转正为 public surface 或登记明确 deferral；不得恢复 retired route 作为 current 入口。
