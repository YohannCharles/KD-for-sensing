## 1. 基线证据

- [x] 1.1 记录实现前 `git status --short`，确认不触碰无关的归档 change、用户改动和未纳入本方案的实验脚本变更。
- [x] 1.2 确认 `legacy_knowledge_decoupling_cleanup_manifest.json`、`知乎问答下载.md`、`scripts/run_priority_v3_budget.sh`、`src/kd_sensing.egg-info/` 的 tracked/ignored/untracked 状态。
- [x] 1.3 记录待收缩 package facade 的内部调用方证据，至少覆盖 `kd_sensing.losses`、`kd_sensing.baselines.rmbp_mm`、`kd_sensing.data.mmw`、`kd_sensing.engine`、`kd_sensing.utils`。
- [x] 1.4 盘点 component registry 的 `register_removed` 项，给每个保留项写明当前迁移价值，给每个删除项写明普通 unknown-name 诊断是否足够。

## 2. 仓库表面清理

- [x] 2.1 删除已跟踪根目录历史清理清单 `legacy_knowledge_decoupling_cleanup_manifest.json`，确认没有 current spec、README、inventory、CLI 或测试仍引用它。
- [x] 2.2 本地删除 ignored 生成物 `src/kd_sensing.egg-info/`，并确认它不出现在 git diff 中。
- [x] 2.3 删除或归档根目录历史笔记 `知乎问答下载.md`；若发现仍有当前维护价值，只保留压缩后的内容到合适文档位置。
- [x] 2.4 不提交 `scripts/run_priority_v3_budget.sh`；若其包含唯一可复用调度逻辑，则合并到 `scripts/run_next_v3_experiments.sh` 或现有 Python 工作流入口，并同步登记。
- [x] 2.5 更新 `docs/project_surface_inventory.md`，反映删除、合并、保留和 local-clean 决策。

## 3. Package Facade 收缩

- [x] 3.1 将 `kd_sensing.losses` 的内部/测试调用改为直接导入 owner 模块。
- [x] 3.2 将 `kd_sensing.baselines.rmbp_mm` 的 CLI/测试调用改为直接导入 owner 模块或已登记 workflow 入口。
- [x] 3.3 收缩 `src/kd_sensing/data/mmw/__init__.py`、`engine/__init__.py`、`utils/__init__.py`、`preprocessing/__init__.py`、`evaluation/__init__.py`、`losses/__init__.py`、`models/physics/__init__.py`、`baselines/rmbp_mm/__init__.py` 为轻量 marker 或明确公共入口。
- [x] 3.4 确认受规格保护的公开 CLI/API facade 未被误删，特别是 JEPA benchmark 兼容入口。

## 4. Component Registry 精简

- [x] 4.1 删除只服务历史强弱模型、teacher/student 临时类、旧 feature extractor 或历史实现变体的低价值 `register_removed` 项。
- [x] 4.2 保留仍有当前迁移价值的 removed guard，并确保错误信息仍给出替代方向。
- [x] 4.3 收缩 `tests/fixtures/legacy_model_registry_retirement.yaml` 和相关测试，使其只断言当前迁移 guard、canonical 注册和普通 unknown-name 行为。

## 5. U-Mask Beam JEPA 指标去重

- [x] 5.1 调整缺失矩阵评估，使 top1/top3/top5、ADBA、MAE 从同一次 `beam_classification_circular_summary` 派生。
- [x] 5.2 检查 `topk_accuracy` 的剩余调用方；若无调用方则删除，否则保留为其他 workflow 的独立 helper，但不在缺失矩阵中重复计算。
- [x] 5.3 保持 U-Mask Beam JEPA 缺失矩阵输出字段和数值语义不变。

## 6. 健康护栏与测试更新

- [x] 6.1 增补或调整架构/健康检查，拒绝跟踪清理 manifest、跟踪 package metadata、未登记脚本和内部 facade 回流导入。
- [x] 6.2 增补或调整 registry 测试，确保低价值 removed guard 删除后不再被大 fixture 保活。
- [x] 6.3 确保护栏不会误伤公开 CLI/API 兼容 facade 和 ignored 本地 artifact。

## 7. 验证

- [x] 7.1 运行 `openspec validate prune-ponytail-audit-followups --strict`。
- [x] 7.2 运行 `openspec status --change prune-ponytail-audit-followups`，确认 artifacts 状态符合实施要求。
- [x] 7.3 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
- [x] 7.4 运行 `conda run -n kd_mm_beam pytest tests/test_component_registry.py tests/test_training_io_workflow.py tests/test_wcl2025_missing_modality.py tests/test_u_mask_beam_jepa_eval_matrix.py -q`。
- [x] 7.5 运行 `conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_config_load_characterization.py -q`。
- [x] 7.6 若改动触及公共工作流或检查脚本，运行 `conda run -n kd_mm_beam pytest -q` 作为最终回归。
