## Why

最近一次表面积审计发现，仓库已经删掉大量旧入口和大 facade，但仍残留几类低价值复杂度：包级 re-export 聚合、单用途包装文件、重复私有 CSV/JSON/float helper，以及 registry 中长期维护的 removed-name table。它们没有新增实验能力，却扩大导入面、测试面和未来改动成本；现在适合用一次小步 OpenSpec change 把这些尾巴收掉。

## What Changes

- 移除或收缩没有当前公开契约价值的 package-level barrel exports，内部代码改为直接导入 owner module。
- 合并单用途包装文件和只被一个调用点使用的薄 helper，避免为了“分层”继续维护一跳转发。
- 收敛重复的私有 CSV/JSON/float/slug helper：优先复用已有 owner helper；确需共享时只放入一个窄模块，不新增跨领域 `utils` 大杂烩。
- 简化 registry removed-name 处理：仍有当前迁移价值的旧名保留清晰拒绝；完全退役且已有 migration guard 或 tombstone 覆盖的历史名可回落为普通 unknown-name 错误。
- 补充架构边界检查，防止后续重新引入重依赖 barrel、旧兼容 facade、tracked bytecode/运行产物或重复 helper 聚合。
- 不改变训练、评估、预处理、模型 forward、指标、checkpoint schema、本地数据目录或任何正式实验数值。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `project-architecture`: 明确低价值 barrel、单用途包装层和重复 helper 的收敛边界，要求内部实现直接依赖 owner module。
- `component-registry`: 明确 removed-name guard 的保留条件，允许完全退役历史名使用普通 unknown-name 错误，避免长期维护低价值 tombstone table。
- `project-health-guardrails`: 增加针对 package barrel 回流、兼容 facade 回流、重复 helper 聚合和 tracked runtime artifact 污染的静态护栏。

## Impact

- 主要影响 `src/kd_sensing/` 下的轻量包导出、registry、config source 包装、transform normalization re-export、诊断 helper 和相关测试。
- 可能删除或缩短的候选包括 `src/kd_sensing/config/source.py`、`src/kd_sensing/data/transform_ops/normalization.py`、若干 `__init__.py` eager re-export、以及 registry 中不再需要的 removed-name table。
- 不新增依赖，不新增 CLI，不恢复旧脚本入口，不读取或修改 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 或 `All_models/`。
- 验证以 OpenSpec strict validate、架构边界、配置加载、component registry、训练 extension/objective focused tests 为主。
