## 1. Inventory 和安全网

- [x] 1.1 建立项目表面积 inventory，统计实体 YAML、脚本入口、README/OpenSpec TBD purpose、本地产物和重复 wrapper 候选。
- [x] 1.2 扩展 `tests/test_architecture_boundaries.py` 或新增 focused test，拒绝已跟踪 `__pycache__`、`.pyc`、`.pytest_cache`、训练输出、日志、cache 和新 checkpoint。
- [x] 1.3 为重复入口建立 allowlist，明确保留的研究脚本、shell orchestration 和允许的薄 alias。
- [x] 1.4 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`，确认新增检查不读取真实数据、不加载 checkpoint、不启动训练。

## 2. 高级配置矩阵 recipe 化

- [x] 2.1 对 `configs/fusion/` 中 G2D、CRAF、MARF 和 ablation YAML 做字段 inventory，标记可由现有 recipe 生成、需要新增 recipe、必须保留三类。
- [x] 2.2 扩展 `src/kd_sensing/config/canonical_recipes/` 和 `src/kd_sensing/config/canonical.py`，补齐第一批删除候选所需 overlay recipe。
- [x] 2.3 添加配置等价测试，比较实体 YAML 与 recipe 生成配置的 experiment、task、dataset、modalities、model、loss/distillation、training、run_name 和 checkpoint 来源。
- [x] 2.4 删除已通过等价测试覆盖的冗余 `configs/fusion/*.yaml`，保留 base/example 或无法无损生成的实体配置。
- [x] 2.5 运行 `conda run -n kd_mm_beam pytest tests/test_student_configs.py tests/test_config_load_characterization.py -q`，确认实体和 virtual 配置加载语义兼容。

## 3. 重复入口和脚本边界

- [x] 3.1 删除已由 `kd-sensing-export-viewer-manifest` 和 `python -m kd_sensing.cli.export_viewer_manifest` 覆盖的 manifest fallback wrapper。
- [x] 3.2 检查 `scripts/`、`tools/analysis/` 和 `tools/visualization/` 中其它 Python 入口，保留项必须归类为研究脚本、shell orchestration 支持脚本或明确允许的薄 alias。
- [x] 3.3 更新 README 和工具文档，所有 manifest 导出推荐命令改为 console script 或包内 CLI。
- [x] 3.4 运行 `conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_architecture_boundaries.py -q`，确认 console script help 和重复入口检查通过。

## 4. README、docs 和 OpenSpec 沉积整理

- [x] 4.1 将 README 收缩为安装、环境、快速健康检查、主要入口、数据/产物边界和链接索引。
- [x] 4.2 新增或更新 `docs/experiment_matrix.md`，承载 README 中的 G2D、CRAF、MARF、CSI hardening 和推荐实验矩阵细节。
- [x] 4.3 保留 `tools/visualization/README.md` 作为 Gradio viewer 详细文档，并移除 README 中重复的大段 viewer 操作说明。
- [x] 4.4 补齐本变更涉及的 OpenSpec specs 中 `TBD - created by archiving` purpose，删除只描述历史迁移过程且不定义当前行为的正文。
- [x] 4.5 运行 `openspec validate reduce-redundant-project-surface --strict` 和 `openspec status --change reduce-redundant-project-surface`，确认 OpenSpec artifact 有效。

## 5. 回归验证

- [x] 5.1 运行快速结构回归：`conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py tests/test_cli_help.py -q`。
- [x] 5.2 运行配置加载回归：`conda run -n kd_mm_beam pytest tests/test_student_configs.py tests/test_student_configs.py::test_load_config_applies_overrides_after_canonical_config_resolution -q`。
- [x] 5.3 运行 manifest 入口回归：`conda run -n kd_mm_beam kd-sensing-export-viewer-manifest --help` 和 `conda run -n kd_mm_beam kd-sensing-visualize-modalities --help`。
- [x] 5.4 若上述 focused checks 通过，运行最终回归：`conda run -n kd_mm_beam pytest -q`。
