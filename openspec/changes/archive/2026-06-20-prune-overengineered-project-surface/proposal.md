## Why

现有仓库已经完成多轮结构收敛，但仍保留了若干“为过去复杂度服务”的表面：机器可读治理索引、退役 tombstone 规格、兼容导入门面、单实现注册表、强依赖和大段守卫测试。它们增加维护成本，却很少改善当前训练、评估、预处理和可视化工作流，因此需要一次明确受控的二阶段瘦身。

本变更把 ponytail 审计结果转成 OpenSpec 契约：先修改仍在保护这些复杂面的规格，再按风险分层删除或收缩代码、文档、测试和依赖，避免“看似清理、实则违约”。

## What Changes

- **BREAKING** 收缩兼容导入面：允许删除不再作为当前公共入口维护的 re-export facade、legacy wrapper 和已迁移别名，内部代码和文档改用真实 owner 模块路径。
- **BREAKING** 降级已退役组件名的专用迁移守卫：对不再承诺兼容的历史 registry 名称，允许使用普通 unknown-name 错误、文档迁移说明或集中退役清单替代运行时 guard table。
- **BREAKING** 放宽 current spec 中的 retired-tombstone 保留要求：当退役研究线不再需要当前运行时守卫时，允许把 tombstone 规格归档或折叠到一个集中历史清单。
- 收缩机器可读治理层：用更小的结构化清单和 focused tests 替代 `docs/maintainer_context_index.yaml` 驱动的大型镜像测试，保留关键入口、边界和漂移检查。
- 删除单实现扩展点：`JEPA_DOWNSTREAM_ADAPTERS` 这类只有 identity 实现、没有实际选择面的注册表可折叠为直接 no-op 路径；等出现第二个真实实现时再恢复注册表。
- 精简依赖和样板：以 Pillow 或既有工具替换 `scikit-image` 图像读取；把 `h5py` 从默认运行时路径移出或改为可选依赖；在 Python 版本契约允许后删除无收益的 `from __future__ import annotations`。
- 简化一次性分析脚本和测试辅助：删除 pandas 分支、巨型 helper、重复 allowlist 和只验证文档复写的断言，保留真正防止架构漂移的最小检查。
- 保持当前核心行为：不改变数据契约、训练语义、评估指标、CLI 主入口、输出分区、checkpoint 策略和本地产物边界。

## Capabilities

### New Capabilities

- 无。本变更是项目表面收缩和契约修订，不引入新的用户功能。

### Modified Capabilities

- `project-surface-cleanup`: 明确第二阶段瘦身范围，覆盖兼容门面、样板代码、依赖和治理表面的删除策略。
- `project-architecture`: 调整公共导入面和 package-level facade 契约，允许把当前代码迁回 owner 模块路径并删除非必要延迟 re-export。
- `project-health-guardrails`: 把必须依赖 `maintainer_context_index.yaml` 的大型治理测试改为小型、可维护、面向关键边界的健康检查。
- `spec-lifecycle-boundaries`: 允许退役 tombstone 规格在满足归档和集中历史说明后退出 current spec 集合。
- `component-registry`: 放宽已移除组件的专用 guard table 和单实现 adapter registry 要求，保留当前组件发现和错误诊断的必要边界。
- `jepa-downstream-extensibility`: 把 identity adapter 视为可内联的默认 no-op，而不是必须通过注册表暴露的扩展点。

## Impact

- 代码范围：`src/kd_sensing/models/` 兼容门面、BeamBench legacy wrapper、JEPA downstream adapter 路径、registry removed-name guard、图像读取与缓存路径、MMW HDF5 path semantics、一次性分析脚本和相关测试 helper。
- 文档范围：`docs/agent_navigation.md`、`docs/project_surface_inventory.md`、`docs/maintainer_context_index.yaml`、README/OpenSpec 中关于公共入口、退役线、健康检查和导入边界的说明。
- 测试范围：`tests/test_architecture_boundaries.py`、`tests/helpers/maintainer_context.py`、registry/JEPA/image preprocessing/path semantics 相关 focused tests。
- 依赖范围：候选删除默认 `scikit-image`；候选把 `h5py` 改为可选依赖；如果当前 Python 版本契约升至或确认不低于 3.10，则批量删除 annotation future import。
- API 影响：历史兼容 import、旧组件名称和 legacy facade 可能停止工作；当前 CLI、配置 schema、训练/评估/预处理主工作流应保持可用。
- 验证范围：OpenSpec strict validate；架构边界 focused tests；图像读取、registry、JEPA downstream、path semantics 的窄测试；必要时运行 `conda run -n kd_mm_beam pytest -q` 做最终回归。
