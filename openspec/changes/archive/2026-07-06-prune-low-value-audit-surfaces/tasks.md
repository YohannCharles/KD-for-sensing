## 1. Baseline 与边界确认

- [x] 1.1 运行 `git status --short`，确认本 change 只处理源码、测试、docs、OpenSpec 和 inventory，不包含本地数据、outputs、logs、cache、checkpoint 或历史权重。
- [x] 1.2 记录本轮候选和保留项：删除/合并 difficulty facade、Scene31 wrapper、eval export writer、JSON deep copy 和 test path 样板；明确不动 `canonical_virtual.py`、runtime artifact cleanup、JEPA GPS shortcut benchmark owner、TinyViT 4 registry names 和 canonical YAML 配置族。
- [x] 1.3 检查 active complete changes：`right-size-public-entrypoint-surface` 和 `shrink-experiment-config-families` 只作为 closeout 提醒，不纳入本 change 实现范围。

## 2. Difficulty package facade 收缩

- [x] 2.1 枚举所有 `from kd_sensing.data.difficulty import ...` 和等价 package-root import 调用点，按符号归属迁到 `schema`、`pipeline` 或真实 owner module。
- [x] 2.2 将 `src/kd_sensing/data/difficulty/__init__.py` 收缩为轻量 marker 或明确 public shim，删除内部 re-export barrel 和 lazy forwarding 维护表。
- [x] 2.3 保留 `kd_sensing.data.difficulty.operators` 注册 side effect，不把 operators eager import 回 package root。
- [x] 2.4 更新 architecture boundary 或 surface guardrail，拒绝内部代码重新从 `kd_sensing.data.difficulty` package root 导入 schema/pipeline 符号。
- [x] 2.5 运行 `conda run -n kd_mm_beam pytest tests/test_modality_difficulty.py tests/test_architecture_boundaries.py -q`。

## 3. Scene31 wrapper 删除

- [x] 3.1 删除 `scripts/summarize_scene31_beamsoft_weak.py`，将 docs/tests/inventory 中当前推荐命令迁到 `scripts/summarize_scene31_bc_next.py --root ...`。
- [x] 3.2 删除 `scripts/run_scene31_modular_maskfix_eval.sh` 和 `scripts/run_scene31_baseline_pack_maskfix_eval.sh`，将当前推荐命令迁到 `scripts/run_scene31_subset_reliability.sh --group eval_modular_lite_maskfix`。
- [x] 3.3 更新 `docs/project_surface_inventory.md`、相关 README/docs 和脚本 allowlist，使已删除 wrapper 不再被 current surface 要求存在。
- [x] 3.4 运行 scripts surface doctor 或等价检查：`PYTHONPATH=src conda run -n kd_mm_beam python -m kd_sensing.cli.project_surface_doctor --scope scripts --format markdown`。
- [x] 3.5 运行仍存在的 Scene31 focused tests；若某个测试因旧 wrapper 删除而应改写，改为验证 canonical command 或 inventory 约束。

## 4. Eval matrix export writer 合并

- [x] 4.1 将 `src/kd_sensing/eval/export.py` 的 CSV/JSON/Markdown writer 合并到 `src/kd_sensing/eval/u_mask_beam_jepa_eval_matrix.py` 或该 owner 附近的窄 helper。
- [x] 4.2 更新 `src/kd_sensing/cli/eval_u_mask_beam_jepa_matrix.py`、`src/kd_sensing/engine/trainer_runtime_helpers.py` 和相关 tests 的 import。
- [x] 4.3 删除 `src/kd_sensing/eval/export.py`，不新增 `eval.utils`、`export_utils` 或旧路径 wrapper。
- [x] 4.4 确认 CSV、JSON、Markdown 输出字段、排序和 formatter 行为保持一致。
- [x] 4.5 运行 `conda run -n kd_mm_beam pytest tests/test_u_mask_beam_jepa_eval_matrix.py tests/test_architecture_boundaries.py -q`。

## 5. 微缩项

- [x] 5.1 在 `gps_query_evidence` 中用 `copy.deepcopy` 替换 JSON 往返 deep copy；只在不扩大依赖和不改变 merge 语义时顺手简化局部 merge helper。
- [x] 5.2 删除普通测试文件中由 `tests/conftest.py` 已覆盖的重复 `ROOT/SRC` path 样板和对应 `E402` noqa；不做无关测试风格重排。
- [x] 5.3 对 `dataset_descriptors` 做 gated simplification：若静态 mapping + query functions 明显更短且保持 API/错误语义，则实施；否则保留实现并补 inventory retained-with-reason。
- [x] 5.4 若触碰 descriptor，运行 `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py tests/test_training_io_dataset_workflow.py -q`。

## 6. 文档、inventory 与防回流

- [x] 6.1 更新 `docs/project_surface_inventory.md`，记录删除项、canonical replacement、保留项理由和 focused validation。
- [x] 6.2 更新 README/docs/OpenSpec current references，移除旧 wrapper、facade 或 `kd_sensing.eval.export` 的 current 推荐路径。
- [x] 6.3 更新 `tests/test_architecture_boundaries.py` 或 project surface doctor，覆盖 difficulty package-root import、Scene31 wrapper 回流和 `kd_sensing.eval.export` 回流。
- [x] 6.4 确认实现没有新增 compatibility shim、alias、二级聚合层或跨领域 helper 杂物间。

## 7. 验证与收尾

- [x] 7.1 运行 `openspec validate prune-low-value-audit-surfaces --strict`。
- [x] 7.2 运行 `openspec validate --all --strict`。
- [x] 7.3 运行 focused pytest：architecture boundary、modality difficulty、U-mask Beam JEPA eval matrix、Scene31 相关测试，以及 descriptor touched 时的 config/data tests。
- [x] 7.4 运行 project surface doctor：scripts、hotspots、closeout scope；记录 complete unarchived change warning 是否仍只指向既有 closeout 候选。
- [x] 7.5 最终说明列出删除、合并、保留未改的项，未运行验证原因，以及后续是否需要单独 archive 已完成 changes。
