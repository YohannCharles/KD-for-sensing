## 1. 状态与入口基线

- [x] 1.1 记录当前 `git status --short --untracked-files=all`，区分已有 `streamline-project-architecture-waves` 归档/源码重构、本 change artifacts 和本 change 新增删除。
- [x] 1.2 枚举 `pyproject.toml` console scripts、`src/kd_sensing/cli/*.py` module-only CLI、README/docs 推荐入口和 current inventory 入口分类。
- [x] 1.3 确认旧 HiST/BGAM/viewer/Raymobtime/AMR-Net_gps_image/JEPA-MSAC/KD 实体 CLI 与实体 YAML 不存在，保留需要的 migration guard。

## 2. Hidden CLI 与 public surface 收口

- [x] 2.1 删除或转正 `src/kd_sensing/cli/` 下不在 `pyproject.toml` 且不是 shared helper 的 module-only CLI。
- [x] 2.2 更新 README、docs/project_surface_inventory.md 和 tests，使 current CLI 推荐只指向 package console scripts 或明确 owner module。
- [x] 2.3 更新 `tests/test_architecture_boundaries.py`，拒绝新增隐藏 runnable CLI，同时允许 `cli/common.py` 等 shared helper。

## 3. Retired route tombstone 合并

- [x] 3.1 建立集中 retired-route guard 数据或 fixture，覆盖旧 CLI、config path、module path、registry token 和 docs token。
- [x] 3.2 将 `tests/test_jepa_msac.py`、`tests/test_amr_net_gps_image.py` 等只验证退役路线的专用测试合并为参数化 retired-route guard。
- [x] 3.3 折叠无独立 guard 价值的 retired tombstone specs，保留或更新集中 retired-route summary。
- [x] 3.4 更新 docs/maintainer_context_index.yaml 和 docs/project_surface_inventory.md，避免重复维护完整墓碑目录。

## 4. Scripts 本地表面精简

- [x] 4.1 枚举 `scripts/` 当前 tracked Python/shell，按 dataset_preparation、config_generator、research_diagnostic、local/manual shell、delete 分类。
- [x] 4.2 删除固定 GPU queue、one-shot、本地 runbook 和已有 package CLI 覆盖的脚本。
- [x] 4.3 将仍保留脚本的 owner、lifecycle、输出边界和删除条件压缩到 inventory，不重复长篇逐脚本说明。
- [x] 4.4 更新架构边界测试，让脚本分类从精简 inventory 或 manifest 推导，不维护完整 allowlist 镜像。

## 5. Config 实体化收缩

- [x] 5.1 枚举 `configs/scene31`、`configs/fusion/experiments` 和 diagnostics/pretraining YAML，区分 current canonical、reproduction、diagnostic manifest、generated/local manual。
- [x] 5.2 对可由 generator/manifest/base config 无损重建的实体 YAML，删除实体文件或记录暂缓理由。
- [x] 5.3 补充或更新 generator sanity tests，覆盖 run name、seed、epoch、sampler、loss weights、missing pattern 和 output boundary。
- [x] 5.4 确认 generator/virtual config 不接管 retired KD/BGAM/viewer/Hist/Raymobtime/AMR/JEPA-MSAC path。

## 6. 文档与测试护栏收口

- [x] 6.1 更新 README、docs/agent_navigation.md、docs/project_surface_inventory.md、docs/maintainer_context_index.yaml 和相关 current specs，反映删减后的 public surface。
- [x] 6.2 更新 `tests/test_architecture_boundaries.py`、config/CLI/retired-route focused tests，确保 guardrail 验证结构事实而不是重复大型目录。
- [x] 6.3 运行 `openspec validate prune-retired-entrypoints-and-local-surfaces --strict`。
- [x] 6.4 运行 `openspec validate --all --strict`。
- [x] 6.5 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py tests/test_config_load_characterization.py tests/test_cli_help.py -q`。
- [x] 6.6 按实际触碰范围追加 generator/script/retired-route focused tests；若不运行全量 `conda run -n kd_mm_beam pytest -q`，最终说明记录原因和剩余风险。

## 7. 收尾

- [x] 7.1 运行 `git status --short --untracked-files=all`，确认没有新增 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、TensorBoard event、`.pytest_cache` 或 `__pycache__` 源码变更。
- [x] 7.2 更新本 tasks 的完成状态，列出删除项、保留项、验证结果和 internal breaking import surface。

## 收尾记录

- 删除/折叠：BeamBench mock-only hidden CLI、JEPA-MSAC/AMR 专用退役测试、18 个单路线 retired tombstone specs、7 个固定 GPU/local shell runner、79 个 Scene31 generated YAML。
- 保留/转正：`kd-sensing-model-architecture-summary` 作为 package console script；Scene31 manifest/base/generator；`scripts/run_rbma_missing_workflow.py`、BTAPA smoke、fresh eval/analysis 和 MMW dataset preparation 脚本作为 local/manual 或 research diagnostic。
- 新增 guard：`openspec/specs/retired-route-summary/spec.md`、`tests/test_retired_routes.py`、Scene31 generator sanity / generated-YAML-not-sourced guard、hidden runnable CLI guard。
- Internal breaking surface：未登记 public surface 的 module-only CLI、deleted shell runner、generated Scene31 YAML 路径和折叠 tombstone spec 路径不再兼容；调用方应使用 package console script、真实 owner module、manifest/generator 或 `kd-sensing-train --config <generated-yaml>`。
- 验证：`openspec validate prune-retired-entrypoints-and-local-surfaces --strict`、`openspec validate --all --strict`、focused pytest 组合、retired-route/Scene31/model-summary focused tests 和全量 `conda run -n kd_mm_beam pytest -q` 均通过。
