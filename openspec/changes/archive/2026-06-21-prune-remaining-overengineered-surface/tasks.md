## 1. 契约与基线

- [x] 1.1 运行 `openspec validate prune-remaining-overengineered-surface --strict`，修正 proposal/design/spec delta 的格式或 requirement 问题。
- [x] 1.2 记录实现前 `git status --short`，确认本 change 不包含 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 或 `All_models/` 新产物。
- [x] 1.3 读取 `docs/project_surface_inventory.md` 中 config lifecycle、源码热点、entrypoint surface 和文档生命周期段落，列出本 change 的最小触达文件清单。
- [x] 1.4 用 CodeGraph 或 AST/rg 确认候选项当前调用方：薄 facade、`registry_self_check`、`_typing.AnyConfig`、`SampleRow`、第二份 `deep_merge`、CSI sweep analyzer、JEPA benchmark star imports 和 BeamBench 聚合 owner。
- [x] 1.5 运行当前基线 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`，记录既有红点和本 change 需要更新的断言边界。

## 2. 配置矩阵分类与等价检查

- [x] 2.1 统计 `configs/` 当前 YAML 数量和各目录数量，更新 inventory 中已漂移的 91/12/12 等旧数字。
- [x] 2.2 为 `configs/fusion/experiments/jepa_image_gps/*.yaml` 建立候选分类：canonical 保留、recipe 可无损生成、recipe 有差异、人工样例、debug/smoke、diagnostics manifest、删除/归档。
- [x] 2.3 为 CSI hardening、fusion CSI hardening、BEV-Fusion、pretraining smoke、diagnostics manifest 和 difficulty configs 建立同样候选分类。
- [x] 2.4 为可 recipe/overlay 化的配置补最小 config load 等价测试，覆盖 experiment name、task/objective、dataset type、enabled modalities、model type、loss type、training defaults、output run name 和 checkpoint/artifact policy。
- [x] 2.5 删除已通过等价检查的重复实体 YAML，并同步 README、docs、OpenSpec specs、scripts 和 tests 中的旧路径引用。
- [x] 2.6 运行 `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py -q` 和涉及被删配置的 focused tests。

## 3. Facade 与内部 import 收缩

- [x] 3.1 将内部源码和测试中对 `kd_sensing.engine.objective_metadata` 的引用迁到 `kd_sensing.engine.objectives.metadata` 或对应 objectives owner。
- [x] 3.2 将内部源码和测试中对 `kd_sensing.data`、`kd_sensing.data.datasets` lazy export 的依赖迁到真实 dataset/sample/target-shot owner 模块。
- [x] 3.3 将 fusion 模型测试和文档从 `kd_sensing.models.fusion` 便利导入迁到 `fusion.networks`、`fusion.cls_token_transformer`、`fusion.token_transformer` 或具体 owner。
- [x] 3.4 删除或最小化 `src/kd_sensing/models/fusion/__init__.py` 中旧类名 `_REMOVED_ALIASES` 迁移表，确认退役 CRAF/MARF/HiST 类名不再通过 facade 提供专属错误。
- [x] 3.5 将 BeamBench CLI、tests 和分析脚本从 `kd_sensing.baselines.beambench.image_ae_gps` 大聚合导入迁到 `image_ae_gps_training.py`、`image_ae_gps_paper_split.py`、`image_ae_gps_config.py`、`image_ae_gps_datasets.py`、`image_ae_gps_models.py` 等 owner。
- [x] 3.6 删除或压缩 `image_ae_gps.py` 和 `baselines/beambench/__init__.py` 的长期 re-export 表，保留 package CLI 行为。
- [x] 3.7 运行 BeamBench focused tests 和 CLI help：`conda run -n kd_mm_beam pytest tests/test_beambench_image_ae_gps_direct.py tests/test_beambench_dataset_check.py tests/test_cli_help.py -q`。

## 4. Registry、Guard 和小工具收缩

- [x] 4.1 删除 `registry_self_check` 及其 `__all__` 导出，确认 `tests/test_component_registry.py` 覆盖 build、unknown name、duplicate name 和 missing required parameter。
- [x] 4.2 审核 `register_removed` 调用，只保留高频迁移价值的 dataset alias、KD/image/profile 等 guard；完全退役路线回落为 ordinary unknown-name 或集中 tombstone 说明。
- [x] 4.3 收缩 `src/kd_sensing/config/migration_guards.py` 中重复路径解析和低价值 retired-route value scan，保留当前 config load 必需拒绝项。
- [x] 4.4 删除 `src/kd_sensing/_typing.py`，将 `AnyConfig` 调用方改为 `dict[str, Any]` 或局部 alias。
- [x] 4.5 删除 `kd_sensing.config.canonical_recipes.common.deep_merge` 副本，统一使用 `kd_sensing.config.io.deep_merge` 或迁移后的单一 owner。
- [x] 4.6 运行 `conda run -n kd_mm_beam pytest tests/test_component_registry.py tests/test_config_load_characterization.py -q`。

## 5. Dataset Runtime Row 收口

- [x] 5.1 将 `target_shot_splits.py` 改为直接消费 `Mapping[str, Any]` rows，并保留 JSON metadata/resource_refs/target_ref 字符串解析。
- [x] 5.2 若仍需 row dataclass，将其迁入 `target_shot_splits.py`；否则删除 `src/kd_sensing/data/dataset_runtime.py`。
- [x] 5.3 简化 `dataset_descriptors.py` 中仅包装静态表的 dataclass 层，或记录保留理由；查询函数和 config validation 行为必须保持兼容。
- [x] 5.4 更新 dataset runtime specs、inventory 和 tests 中对 `SampleRow` 独立文件的引用。
- [x] 5.5 运行 target-shot/dataset/config focused tests：`conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py tests/test_training_io_workflow.py -q -k "target_shot or dataset_descriptor or input_profiles"`。

## 6. 一次性脚本和 JEPA Benchmark API 风格

- [x] 6.1 删除或归档 `scripts/analyze_csi_hardening_sweep.py`，将仍有价值的 CSI hardening 调试结论写入 `docs/research_notes.md` 或对应 current 报告。
- [x] 6.2 删除只服务该脚本的测试或改成文档/metadata focused check，避免继续要求脚本存在。
- [x] 6.3 将 `src/kd_sensing/diagnostics/jepa_benchmark_*.py` 中的 `from kd_sensing.diagnostics.jepa_benchmark_common import *` 改为显式导入实际使用符号。
- [x] 6.4 清理不必要 `__all__` 镜像，只保留 public facade 需要导出的稳定 API。
- [x] 6.5 运行 `conda run -n kd_mm_beam pytest tests/test_jepa_gps_shortcut_benchmark.py -q`。

## 7. 健康护栏重写

- [x] 7.1 收缩 `tests/test_architecture_boundaries.py` 顶部大型常量和 allowlist，只保留 retired token、关键 current path、pyproject script、tracked artifact boundary 和轻量 import probe 所需数据。
- [x] 7.2 删除逐字文档短语镜像断言，改成检查 current path/config/module/lifecycle 是否真实存在，以及 retired route 是否带历史/退役/拒绝语境。
- [x] 7.3 保留或补充三个最小结构回归：旧入口回流失败、tracked 本地产物失败、current config/path 引用失效失败。
- [x] 7.4 确认 ignored `__pycache__`、`.pytest_cache`、`outputs/` 和 `logs/` 不会让常规架构边界测试失败。
- [x] 7.5 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。

## 8. 文档与 OpenSpec 同步

- [x] 8.1 更新 `docs/project_surface_inventory.md`：配置数量、候选分类、删除/保留理由、facade 收缩、registry guard 收缩和健康护栏新边界。
- [x] 8.2 更新 `docs/agent_navigation.md`：强调架构边界测试验证结构事实、维护索引只保留最小事实、内部 import 使用 owner 模块。
- [x] 8.3 更新 `docs/maintainer_context_index.yaml`，只保留无法从 pyproject、真实路径、OpenSpec 或 inventory 推导的 retired tokens 和 validation 命令。
- [x] 8.4 更新 README/相关 docs 中 BeamBench、配置矩阵、retired routes 和 current CLI 的入口说明，删除已删 facade 或实体 YAML 的 current 引用。
- [x] 8.5 确认 `add-scene-conditioned-meta-offset-calibration` active change 没有被本 change 的文档更新误覆盖；如有冲突，记录合并顺序。

## 9. 最终验证与收口

- [x] 9.1 运行 `openspec validate prune-remaining-overengineered-surface --strict`。
- [x] 9.2 运行架构和配置 smoke：`conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py tests/test_config_load_characterization.py tests/test_cli_help.py -q`。
- [x] 9.3 运行受影响 focused suite：component registry、BeamBench、JEPA benchmark、dataset/target-shot split。
- [x] 9.4 如多个 wave 均触碰核心模块，运行最终回归 `conda run -n kd_mm_beam pytest -q`。
- [x] 9.5 再次检查 `git status --short`，确认没有 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、`All_models/` 或临时验证产物进入源码变更。
- [x] 9.6 在最终说明中列出 breaking import changes、删除/保留的配置类别、未运行验证、剩余风险和后续可继续删除的候选项。
