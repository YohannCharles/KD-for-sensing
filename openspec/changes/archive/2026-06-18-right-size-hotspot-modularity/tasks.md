## 1. Baseline 与治理 schema

- [x] 1.1 记录当前热点 baseline：`wc -l`、主要 public functions/classes、当前 `git status --short`、已知测试红点和本 change 不负责的本地产物噪声。
- [x] 1.2 运行并记录初始 `openspec validate right-size-hotspot-modularity --strict`。
- [x] 1.3 运行并记录初始 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`，区分既有红点和后续 wave 引入的红点。
- [x] 1.4 更新 `docs/maintainer_context_index.yaml` 的 hotspot metadata schema，增加 `right-size-accepted`、`merge-candidate`、`enforcement`、`headroom_lines`、`consolidation_targets`、`remediation_waves`、`planned_action`、`public_surface_policy` 和 rollback/validation 字段。
- [x] 1.5 更新 `tests/test_architecture_boundaries.py` 或其 helper，使测试能验证新 metadata 的合法值、必填字段、路径存在性、owner 合法性和 `conda run -n kd_mm_beam` 验证命令。
- [x] 1.6 更新 `docs/project_surface_inventory.md` 和 `docs/agent_navigation.md`，说明完整修复 campaign、wave 顺序、拆/合/保留决策矩阵和 keep-and-test 语义。

## 2. Wave 1 - engine data factory 拆分

- [x] 2.1 为 `src/kd_sensing/engine/data_factory.py` 捕获 public surface：列出必须保留的公开函数和内部 helper，不改变 CLI、trainer 或 evaluator 调用语义。
- [x] 2.2 新建或更新 loader 职责模块，迁移 `build_dataloader_kwargs`、`resolve_dataloader_split_config`、`_split_loader_value` 和 `shutdown_dataloader_workers` 等 loader-only 逻辑。
- [x] 2.3 新建或更新 protocol/scene split 职责模块，迁移 protocol role、scene union dataset、stratified 2604 split 和 scene retarget helper。
- [x] 2.4 新建或更新 group split 职责模块，迁移 label stratification、sequence group keys、holdout count 和 proportional group count helper。
- [x] 2.5 新建或更新 internal validation 职责模块，迁移 validation-from-train config、subset split、subset annotation 和 related helper。
- [x] 2.6 新建或更新 scaler/normalizer 职责模块，迁移 GPS scaler fit/apply、multi-scene scaler harmonization、normalization kwargs 和 LiDAR normalizer coordination。
- [x] 2.7 保持 `data_factory.py` 作为构建 owner，只保留公开构建入口和少量 orchestration；确保 helper 模块不反向导入 public owner。
- [x] 2.8 运行 `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py tests/test_epoch_subsampling.py -q` 和 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。

## 3. Wave 1b - preprocessing sequence 拆分

- [x] 3.1 捕获 `src/kd_sensing/preprocessing/sequences.py` 的 public surface：`generate_sequence_data`、`SequencePreprocessor`、sequence dataclasses 和被测试直接导入的 helper。
- [x] 3.2 拆出 sequence column plan/source column/required column validation 到专用模块。
- [x] 3.3 拆出 sequence window materialization 和 window column naming 到专用模块。
- [x] 3.4 拆出 balanced split selection、split scoring、protocol plan 和 distribution distance 到专用模块。
- [x] 3.5 拆出 metadata writer、label distribution summary 和 JSON-ready helper 到专用模块。
- [x] 3.6 保留 `sequences.py` 为预处理 owner/orchestration，不创建无领域语义的 `utils` 聚合。
- [x] 3.7 运行相关 preprocessing/config focused tests；若没有专用测试，至少运行 `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py tests/test_architecture_boundaries.py -q`。

## 4. Wave 2 - BeamBench Image AE+GPS 完整拆分

- [x] 4.1 捕获 `src/kd_sensing/baselines/beambench/image_ae_gps.py` 的 public surface、CLI 调用路径和当前 `run_image_ae_gps_paper_split_training` 预算红点。
- [x] 4.2 拆出 config/dataclass/normalization/device/performance metadata 到 BeamBench config 模块。
- [x] 4.3 拆出 `BeamBenchImageAEGPSDataset`、`BeamBenchImageOnlyDataset`、`BeamBenchImageAEGPSFeatureDataset` 和 loader/image helpers 到 dataset 模块。
- [x] 4.4 拆出 `BeamBenchDenseModel`、`BeamBenchImageAEGPSDirectModel` 和 model-specific helpers 到 model 模块。
- [x] 4.5 拆出 camera AE train/load、latent encoding、feature cache path/signature 和 cache IO 到 AE/cache 模块。
- [x] 4.6 拆出 direct training、classifier epoch、loss、optimizer/runtime helpers 到 training 模块。
- [x] 4.7 拆出 evaluation pass、prediction rows、CSV/JSON-ready helpers 到 evaluation/report IO 模块。
- [x] 4.8 拆出 paper split orchestration、scene-specific config、checkpoint reuse、paper split train/eval summary 到 paper split 模块，修复当前预算红点。
- [x] 4.9 保留 `image_ae_gps.py` 为 public owner/re-export 或明确 thin orchestration；更新维护索引和架构测试，防止 helper 回流。
- [x] 4.10 运行 `conda run -n kd_mm_beam pytest tests/test_beambench_image_ae_gps_direct.py -q` 和 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。

## 5. Wave 3 - DeepSense6G/MMW dataset 与 trainer runtime

- [x] 5.1 对 `DeepSense6GDataset` 做 second pass：拆出 sample assembly、resource reader glue、scaler/normalizer setup、target provider adapter，避免新增多层 mixin 继承。
- [x] 5.2 对 `MMWDataset` 做 second pass：拆出 manifest/derived-column ensure helpers、geometry payload/tensor helpers、radio/path semantic helpers 和 availability helpers。
- [x] 5.3 更新 dataset hotspot metadata，记录哪些 helper 是 split target，哪些保持在 class 内作为 cohesion boundary。
- [x] 5.4 拆分 `trainer._train_inner`：runtime plan、dataloader setup、epoch loop、validation/checkpoint coordination、final evaluation 和 artifact finalization。
- [x] 5.5 运行 `conda run -n kd_mm_beam pytest tests/test_deepsense6g_contract_helpers.py -q`。
- [x] 5.6 运行 `conda run -n kd_mm_beam pytest tests/test_gps_modality.py tests/test_lidar_modality.py tests/test_mmwave_modality.py tests/test_csi_modality.py -q`。
- [x] 5.7 运行 `conda run -n kd_mm_beam pytest tests/test_mmw_town10_preparation.py tests/test_beam_label_calibration.py -q`。
- [x] 5.8 运行 `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py tests/test_epoch_subsampling.py -q` 和 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。

## 6. Wave 4 - JEPA benchmark 第二层拆分

- [x] 6.1 捕获 `jepa_benchmark_common.py`、`jepa_benchmark_scenario_d.py`、`jepa_benchmark_runner.py` 的 public surface 和 re-export 需求。
- [x] 6.2 从 `jepa_benchmark_common.py` 拆出 scalar/numeric helpers、metadata/sample-id helpers、JSON/CSV/path/hash IO helpers。
- [x] 6.3 从 `jepa_benchmark_scenario_d.py` 拆出 scenario D/CxD suite normalization、CxD phase grid/artifacts、dominance/crossing analysis、failure mode decomposition 和 metric row generation。
- [x] 6.4 从 `jepa_benchmark_runner.py` 拆出 robustness summary、shortcut reliance、case studies、analysis bundle reader 和 runner manifest builder。
- [x] 6.5 确认 `jepa_gps_shortcut_benchmark.py` 继续作为 public facade，且 suite-specific helper 不回流。
- [x] 6.6 运行 `conda run -n kd_mm_beam pytest tests/test_jepa_gps_shortcut_benchmark.py -q` 和 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。

## 7. Wave 5 - 合并、保留与最终收口

- [x] 7.1 扫描本 change 产生或触碰的单调用点包装类、无领域语义 helper、重复 `utils` 聚合和无公开价值 facade。
- [x] 7.2 将低价值边界合并回清晰 owner 或改为私有局部 helper；若暂缓合并，登记为 `merge-candidate` 并写明 owner、consolidation target 和验证命令。
- [x] 7.3 将 `losses/jepa.py`、`losses/gps_lidar_bgam_losses.py` 和 `models/csi_encoder.py` 明确标记为 keep-and-test 或 monitor，除非实现阶段发现重复抽象或测试缺口。
- [x] 7.4 更新 README/docs 中与 public import、CLI、workflow owner 或热点治理相关的说明；不得新增旧入口或把输出产物纳入源码。
- [x] 7.5 运行 `openspec validate right-size-hotspot-modularity --strict`。
- [x] 7.6 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
- [x] 7.7 运行所有被触碰领域的 focused tests；若环境允许，最终运行 `conda run -n kd_mm_beam pytest -q`。
- [x] 7.8 最终说明中列出每个 wave 的完成状态、验证结果、未运行命令、剩余风险和未清理的后续候选。
