## Why

本轮 ponytail-audit 显示，当前 surface doctor 已经能让仓库保持登记一致，但仍有少量低价值包装层和重复入口占据 current surface：difficulty 子包级 re-export facade、Scene31 薄 wrapper、eval matrix export 小聚合模块、测试路径样板和一个 JSON 往返 deep copy。它们没有增加研究能力，却会继续扩大维护面和误导后续协作者。

本 change 目标是一次性删掉或合并这些已确认低价值表面，保留真实 owner、当前 package CLI、canonical recipe 和必要诊断能力。不新增兼容 alias，不恢复旧入口，不触碰本地数据或运行产物。

## What Changes

- **BREAKING**：内部源码和测试从 `kd_sensing.data.difficulty` package facade 迁到真实 owner module；该子包 `__init__.py` 收缩为轻量 marker 或删除 re-export 表。保留 `kd_sensing.data.difficulty.operators` 的注册 side effect 边界。
- 删除 Scene31 重复 wrapper：`scripts/summarize_scene31_beamsoft_weak.py` 以及只转发 `eval_modular_lite_maskfix` 的 shell wrapper；文档、inventory 和测试指向 canonical 命令。
- 将 `kd_sensing.eval.export` 中的 CSV/JSON/Markdown writer 合并回 U-mask Beam JEPA eval matrix owner 或更窄的本地 helper；删除跨领域小聚合模块。
- 用 `copy.deepcopy` 替换 `gps_query_evidence` 中的 JSON 往返复制；只在能减少代码且不削弱错误信息时，才简化 `dataset_descriptors` dataclass 层。
- 删除 tests 中由 `tests/conftest.py` 已覆盖的重复 `ROOT/SRC` path 样板和对应 `E402` 噪声。
- 更新 `docs/project_surface_inventory.md`、architecture boundary、focused tests 和相关 OpenSpec current specs，使删除项不会回流。

## Capabilities

### New Capabilities

- 无。本 change 只收缩现有源码、脚本和测试表面。

### Modified Capabilities

- `project-import-surface-consolidation`：增加本次审计确认的 package facade、单用途 writer 聚合、样板测试导入和 gated descriptor simplification 的收敛规则。
- `project-entrypoint-lifecycle`：增加 Scene31 重复 wrapper 删除和 canonical command 收口规则。
- `modality-difficulty-pipeline`：明确 difficulty pipeline 内部导入必须走 owner module，不能通过子包 barrel 回流；注册型 `operators` package 例外保留。
- `u-mask-beam-jepa-eval-matrix`：明确 eval matrix export writer 归属 eval matrix owner，不保留 `kd_sensing.eval.export` 小型聚合面。

## Impact

- 影响范围：`src/kd_sensing/data/difficulty/__init__.py`、difficulty import 调用点、Scene31 scripts/docs/tests/inventory、`src/kd_sensing/eval/export.py` 调用点、`gps_query_evidence` 配置复制、tests 路径样板、architecture boundary 和 project surface inventory。
- 不影响范围：训练数学语义、模型 forward、dataset split、canonical configs、package CLI 主入口、JEPA/GPS benchmark owner、`canonical_virtual.py`、runtime artifact cleanup、本地 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 和 `All_models/`。
- 兼容性：旧 package-level difficulty import 和被删 Scene31 wrapper path 不再承诺可用；替代路径是 owner module import 或 current canonical command。
- 验证：OpenSpec strict validate、architecture boundary、surface doctor、difficulty pipeline tests、U-mask Beam JEPA eval matrix tests、Scene31 focused tests，以及 descriptor/gps evidence touched 时的相应窄测试。
