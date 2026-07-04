## 1. 文档和维护索引收口

- [x] 1.1 更新 `docs/maintainer_context_index.yaml` 的 `validation.focused`，删除已不存在的 `openspec validate prune-retired-entrypoints-and-local-surfaces --strict`，改为当前可运行的 `openspec validate --all --strict` 和 focused pytest 命令。
- [x] 1.2 更新 `docs/project_surface_inventory.md` 的架构尺寸基线段，明确当前工作树 on-disk 或 tracked-only 统计口径、扫描范围、排除项和“趋势信号而非硬 KPI”用途。
- [x] 1.3 如需要，更新 `docs/agent_navigation.md` 或 README 的相关段落，说明归档后 archive 目录、generated YAML 删除和当前 validation 命令的读取边界。
- [x] 1.4 检查文档中所有当前可复制的 `openspec validate <name> --strict` 命令，确认 `<name>` 是 active change、current spec 或通用 strict 校验；历史 archive artifact 中的旧命令保持历史语境即可。

## 2. 架构边界测试增强

- [x] 2.1 扩展 `tests/test_architecture_boundaries.py`，检查维护索引或当前 focused validation 中的 `openspec validate <name> --strict` 不引用 inactive/missing change。
- [x] 2.2 扩展 `tests/test_architecture_boundaries.py`，检查普通 `tests/test_*.py` 文件不得在文件级插入 `tests/` 目录到 `sys.path`，保留 subprocess/import-boundary probe 例外。
- [x] 2.3 扩展 `tests/test_architecture_boundaries.py`，检查 `docs/project_surface_inventory.md` 的统计基线段包含统计口径、扫描范围、排除项和非硬 KPI 说明。
- [x] 2.4 扩展 `tests/test_architecture_boundaries.py` 或实现等价静态检查，审计 OpenSpec active change 删除与同名 dated archive 新增是否成对出现在 git 状态中；若暂不强制失败，必须输出可定位的 deferral 信息。

## 3. 测试 helper 导入收口

- [x] 3.1 将 `tests/test_jepa_gps_shortcut_benchmark.py`、`tests/test_jepa_gps_shortcut_manifest.py` 和 `tests/test_jepa_gps_shortcut_perturbations.py` 的 `jepa_gps_shortcut_helpers` 导入改为 shared bootstrap 可解析的 package-style import。
- [x] 3.2 将 `tests/test_training_io_cache_workflow.py`、`tests/test_training_io_dataset_workflow.py`、`tests/test_training_io_label_workflow.py` 和 `tests/test_training_io_run_metadata.py` 的 `training_io_helpers` 导入改为 shared bootstrap 可解析的 package-style import。
- [x] 3.3 删除上述普通测试文件中的 `TESTS = ...`、`sys.path.insert(0, str(TESTS))` 和由此产生的 `# noqa: E402` 例外；保留真实需要的源码导入排序。
- [x] 3.4 运行 `conda run -n kd_mm_beam pytest tests/test_training_io_cache_workflow.py tests/test_training_io_dataset_workflow.py tests/test_training_io_label_workflow.py tests/test_training_io_run_metadata.py -q`。
- [x] 3.5 运行 `conda run -n kd_mm_beam pytest tests/test_jepa_gps_shortcut_benchmark.py tests/test_jepa_gps_shortcut_manifest.py tests/test_jepa_gps_shortcut_perturbations.py -q`。

## 4. 可选 MMW warning 清理

- [x] 4.1 检查 `src/kd_sensing/data/datasets/mmw_columns.py` 中逐列插入 DataFrame 的位置，确认 warning 来源和当前 sample 字段语义。
- [x] 4.2 如确认无行为风险，将多列构造改为先构建新列 mapping/DataFrame，再按原 index 一次性 `pd.concat(axis=1)`；不得改变字段名、index、label 或 metadata。
- [x] 4.3 运行 `conda run -n kd_mm_beam pytest tests/test_mmw_town10_preparation.py -q`，确认 MMW sample/preparation contract 不变。
- [x] 4.4 若 4.2 导致 focused test 失败或需要扩大 dataset 语义改动，则回滚该可选优化，仅保留治理文档和测试收口。

## 5. 归档和生成型配置收口检查

- [x] 5.1 运行 `openspec list --json`，确认当前 active change 状态，并记录新 change 与既有未提交 archive 目录的关系。
- [x] 5.2 运行 `git status --short`、`git ls-files --deleted` 和 `git ls-files --others --exclude-standard openspec/changes/archive`，确认 deleted active change 与 archive 新增成对存在。
- [x] 5.3 确认 Scene31 generated YAML 仍不作为 tracked source surface 回流，必要时运行 `conda run -n kd_mm_beam pytest tests/test_scene31_next_round.py -q`。
- [x] 5.4 最终说明中记录 archive 成对提交状态；若因用户提交节奏暂不纳入某些 archive 目录，记录 deferral 原因和后续处理触发条件。

## 6. 验证

- [x] 6.1 运行 `openspec validate stabilize-post-archive-governance --strict`。
- [x] 6.2 运行 `openspec validate --all --strict`。
- [x] 6.3 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py tests/test_scene31_next_round.py -q`。
- [x] 6.4 运行 `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py tests/test_cli_help.py -q`。
- [x] 6.5 按实际触碰范围运行 training/data/evaluation 或 JEPA diagnostics focused tests；至少覆盖本 tasks 中第 3 组和第 4 组已修改文件。
- [x] 6.6 最终运行 `conda run -n kd_mm_beam pytest -q`；若因环境或时间未运行，最终说明必须列出已运行替代验证和剩余风险。
