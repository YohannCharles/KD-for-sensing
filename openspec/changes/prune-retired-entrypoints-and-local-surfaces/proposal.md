## Why

项目已经删除了多数旧 CLI、旧实体配置和旧实现模块，但退役路线仍以大量 tombstone spec、专用测试、隐藏 `python -m` 入口、本地脚本和实体化实验 YAML 的形式占据维护表面。现在继续新增模型或诊断前，应先把这些“不是 current workflow 的东西”压缩成少量护栏，降低阅读、验证和后续重构成本。

## What Changes

- 删除或转正未登记为 public surface 的 module-only CLI；默认只保留 `pyproject.toml` console scripts、README/current docs 推荐入口、current OpenSpec 明确保护的 owner module 和 registry/config 构建入口。
- 将 JEPA-MSAC、AMR-Net_gps_image、HiST/Hist、BGAM、viewer、Raymobtime、legacy KD、GPS residual、Top8 selector 等退役路线的重复 tombstone specs 和专用测试折叠为集中 retired-route guard；保留 fail-fast migration guard 和防回流测试。
- 收缩 `scripts/` 本地研究脚本和 shell queue 表面：删除低价值 one-shot、固定 GPU 队列、已有 package CLI 覆盖的 helper；保留 dataset preparation、config generator 和少量仍有 current claim/protocol 价值的 research diagnostic。
- 将规则化 Scene31/night-grid/next-round/seed sweep 配置族优先表达为 base config + manifest/generator sanity test；删除可由 generator 无损重建且不属于 canonical/current/reproduction/diagnostic 的实体 YAML。
- 更新 README、`docs/project_surface_inventory.md`、`docs/agent_navigation.md`、`docs/maintainer_context_index.yaml` 和架构边界测试，使它们验证结构事实而不是维护第二份完整旧入口目录。
- **BREAKING**：未登记为 public surface 的内部 import path、module-only CLI、local/manual script、retired tombstone spec 文件和可生成实验 YAML 可以被删除；内部调用方必须改用真实 owner module 或 package console script。
- **不改变**：当前 `kd-sensing-*` console script 名称、current canonical config 语义、dataset split/label-space/metric/checkpoint schema、默认输出分区、runtime artifact 保护和已登记 current workflow 的用户可见行为。

## Capabilities

### New Capabilities

无。本 change 不引入新的算法、训练能力或用户级 workflow，只收缩现有项目表面。

### Modified Capabilities

- `project-entrypoint-lifecycle`: 明确 module-only CLI、local/manual script 和 hidden public API 的删除/转正规则。
- `project-import-surface-consolidation`: 扩大内部 import path、low-value facade、package marker 和 hidden module entry 的收缩边界。
- `canonical-config-resolution`: 增加可生成实验 YAML 的删除条件、generator/manifest 保留语义和 retired path guard 要求。
- `spec-lifecycle-boundaries`: 将 retired tombstone 保留改为 guard-value 驱动，允许折叠到集中 retired-route summary。
- `project-health-guardrails`: 调整架构边界测试职责，保留旧入口回流、tracked artifact、facade 回流和 current path 失效检查，删除重复 tombstone/spec/script 镜像。
- `project-architecture`: 明确本轮大规模 surface pruning 的 public behavior compatibility 和 internal breaking import 边界。

## Impact

- 主要影响 `pyproject.toml`、`src/kd_sensing/cli/`、`src/kd_sensing/config/migration_guards.py`、`tests/test_architecture_boundaries.py`、退役路线 focused tests、`scripts/`、`configs/scene31/`、`configs/fusion/experiments/`、`docs/project_surface_inventory.md`、README、`openspec/specs/` 和本 change artifacts。
- 需要先记录当前脏工作树：已有 `streamline-project-architecture-waves` archive、dataset/runtime/model 重构和未跟踪新模块不得被本 change 覆盖或回退。
- 验证以 OpenSpec strict、architecture boundary、config/CLI smoke、retired-route guard focused tests 和必要 generator sanity tests 为主；最终视改动范围运行 `conda run -n kd_mm_beam pytest -q`。
