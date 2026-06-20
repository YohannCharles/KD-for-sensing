## 1. 契约和基线确认

- [x] 1.1 运行 `openspec validate prune-overengineered-project-surface --strict`，确认本 change 的 proposal、design、spec delta 和 tasks 可被 OpenSpec 接受
- [x] 1.2 记录实现前的 `git status --short`，确认后续修改不包含 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 或 `All_models/` 新产物
- [x] 1.3 用 `rg` 和 CodeGraph 确认本 change 涉及的 facade、removed guard、adapter registry、维护索引和依赖调用点，形成实现时的最小触达文件清单
- [x] 1.4 运行当前架构基线 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`，记录既有红点和本 change 需要更新的断言边界

## 2. 依赖与样板瘦身

- [x] 2.1 将 `skimage.io.imread` 调用替换为 Pillow 或已保留依赖，并保持当前图像 profile 的 dtype、shape、通道和缓存行为
- [x] 2.2 从 `pyproject.toml` 默认依赖中删除 `scikit-image`，并更新任何仍提到该默认依赖的文档或测试
- [x] 2.3 审核 `h5py` 当前调用路径，把非默认 HDF5 path semantics 或真实 HDF5 读取改为局部导入、optional extra 或保留默认依赖并记录理由
- [x] 2.4 在确认 Python 版本契约不低于 3.10 后，批量删除无收益的 `from __future__ import annotations`
- [x] 2.5 运行图像和导入相关 focused tests，例如 `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py tests/test_modality_visual_diagnostics.py -q`

## 3. 单实现扩展点和一次性脚本

- [x] 3.1 删除或折叠 `JEPA_DOWNSTREAM_ADAPTERS` 单实现 registry，将 `identity` adapter 保持为默认 no-op 路径
- [x] 3.2 保留 JEPA downstream pooler 的实际选择面，确保 `mean` 和 `gps_query_attention` 仍可通过配置构建
- [x] 3.3 更新 JEPA downstream 相关测试和文档，使它们不再要求 identity adapter 注册表存在
- [x] 3.4 简化 `scripts/analyze_csi_hardening_sweep.py` 的重复分支，只保留当前需要的 CSV/rows 聚合路径，删除无当前价值的 pandas 特化路径
- [x] 3.5 运行 JEPA 和 CSI focused tests 或 import smoke，例如 `conda run -n kd_mm_beam pytest tests -q -k "jepa_downstream or csi_hardening"`

## 4. Facade 和 registry guard 收缩

- [x] 4.1 将内部源码、tests、README/docs 和 OpenSpec 当前引用迁出 `kd_sensing.models` package-level 便利导出，改用真实 owner 模块、registry 名称或配置路径
- [x] 4.2 收缩或删除 `src/kd_sensing/models/__init__.py`、`src/kd_sensing/models/csi.py` 等仅提供历史 re-export 的 facade，并保留轻量导入边界
- [x] 4.3 审核 BeamBench legacy wrapper 和 JEPA benchmark facade，删除无 current public surface 价值的 wrapper，或把仍需保留的入口压成薄委托层
- [x] 4.4 精简 registry removed-name guard table：只保留仍有当前迁移价值的旧名称，其余回落为普通 unknown-name 错误或集中退役说明
- [x] 4.5 更新 registry、models package 和 facade 相关测试，验证 current registry discovery 只列当前入口且旧入口不会被兼容重定向

## 5. 健康护栏和维护文档瘦身

- [x] 5.1 删除、收缩或替换 `docs/maintainer_context_index.yaml`，只保留无法从 pyproject、OpenSpec、真实路径或 inventory 推导的最小结构化事实
- [x] 5.2 删除或收缩 `tests/helpers/maintainer_context.py`，把仍必要的解析逻辑迁到 focused tests 或更小的测试私有 helper
- [x] 5.3 重写 `tests/test_architecture_boundaries.py` 中依赖大型维护索引的断言，使其直接验证 pyproject scripts、current docs 路径、退役 token、轻量导入、本地产物边界和关键 facade 回流
- [x] 5.4 更新 `docs/agent_navigation.md`、`docs/project_surface_inventory.md` 和 README 中关于维护索引、健康检查、current entrypoint 和 breaking import 的说明
- [x] 5.5 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`，确认健康护栏仍能拒绝退役路线回流和重依赖 eager import

## 6. Retired tombstone 和生命周期收口

- [x] 6.1 枚举 `openspec/specs/` 中 lifecycle 为 `retired-tombstone` 的 specs，区分仍需 current guard 的墓碑和只剩历史说明的墓碑
- [x] 6.2 将无 current guard 价值的 tombstone spec 归档或折叠到集中历史清单，并更新 lifecycle inventory
- [x] 6.3 确认 current specs、README/docs 和健康护栏不再把已归档退役能力解释为 current 支持面
- [x] 6.4 运行 `openspec validate prune-overengineered-project-surface --strict` 和必要的 lifecycle 文档检查

## 7. 回归验证和收口

- [x] 7.1 运行 `openspec status --change prune-overengineered-project-surface`，确认 implementation 任务状态和 remaining work 清晰
- [x] 7.2 运行本 change 的 focused tests：`conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`
- [x] 7.3 运行受影响模块的窄测试：`conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py tests/test_modality_visual_diagnostics.py -q`
- [x] 7.4 若 facade、registry、依赖和健康护栏 wave 都已修改，运行最终回归 `conda run -n kd_mm_beam pytest -q`
- [x] 7.5 检查 `git status --short`，确认没有数据、输出、日志、cache、checkpoint 或临时验证产物进入源码变更
- [x] 7.6 在最终说明中列出 breaking changes、删除的依赖/文件、未运行验证及剩余风险
