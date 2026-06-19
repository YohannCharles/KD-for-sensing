## 1. 治理基线与护栏

- [x] 1.1 在 `docs/maintainer_context_index.yaml` 增加 architecture sizing baseline，记录 CodeGraph/AST 统计口径、源码/测试/scripts 分布、主要子包复杂度、排除的本地产物路径和统计更新时间。
- [x] 1.2 扩展 `governance.hotspots` 元数据，补齐 `planned_action`、`public_surface_policy`、`rollback_note`、`accepted_size_rationale`、`consolidation_targets`、`split_targets` 和二级热点状态字段。
- [x] 1.3 将 `jepa_visual_analysis.py`、`runtime_artifact_cleanup.py`、`models/modular.py`、`config/canonical.py`、`data/difficulty/operators/image.py`、`data/transform_ops/csi.py` 等二级热点纳入 index 或 inventory，并标注 monitor、split-next、defer-with-rationale 或 keep-and-test。
- [x] 1.4 更新 `docs/project_surface_inventory.md`，补充“文件数/函数数/import 数不是单独 KPI”的解释、该拆/该合并/该保留的判定矩阵和新增二级热点说明。
- [x] 1.5 更新 `docs/agent_navigation.md`，说明架构整理时先读取 architecture sizing baseline、remediation wave 和 merge-candidate/accepted owner。
- [x] 1.6 更新 `tests/helpers/maintainer_context.py` 和 `tests/test_architecture_boundaries.py`，验证新增索引字段、状态枚举、public surface policy、rollback note、accepted rationale 和本地产物排除规则。
- [x] 1.7 运行 `openspec validate right-size-project-architecture --strict` 和 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。

## 2. BeamBench Image AE+GPS Wave

- [x] 2.1 用 CodeGraph callers/callees 确认 `run_image_ae_gps_training`、`run_image_ae_gps_paper_split_training`、`image_ae_gps.py` public owner 和 CLI alias 的 public surface。
- [x] 2.2 将 paper split summary payload、summary artifact writer 和 GPS calibration metadata 写出职责迁移到 `image_ae_gps_reports.py` 或等价 reports owner。
- [x] 2.3 将 paper split AE checkpoint resolution、AE auto-train/load 和 frozen feature cache setup 收敛到 `image_ae_gps_ae.py` 或 `image_ae_gps_training.py` 中已有 owner，避免 `image_ae_gps_paper_split.py` 继续吸收 cache/checkpoint 细节。
- [x] 2.4 将 scene-specific train/eval dataset build、GPS scaler fit/apply 和 DataLoader build 复用 `image_ae_gps_datasets.py` 的窄 helper，保留 paper split 函数作为 orchestration。
- [x] 2.5 保持 `kd_sensing.baselines.beambench.image_ae_gps`、`kd-sensing-train-beambench-image-ae-gps`、`kd-sensing-run-beambench-image-ae-gps-tableiii` 和 root script thin alias 的公开行为不变。
- [x] 2.6 补充或调整 `tests/test_beambench_image_ae_gps_direct.py`，覆盖 dry-run、paper split summary fields、checkpoint reuse metadata、feature cache report 和 public import owner。
- [x] 2.7 运行 `conda run -n kd_mm_beam pytest tests/test_beambench_image_ae_gps_direct.py -q` 和 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。

## 3. Dataset 与 Trainer Wave

- [x] 3.1 将 `DeepSense6GDataset.__init__` 中 resource reader/path setup 迁入 `deepsense6g_loaders.py` 或等价 loader owner，保持 sample contract 不变。
- [x] 3.2 将 GPS/LiDAR/mmWave/CSI scaler 和 normalizer setup 迁入 `deepsense6g_scalers.py` 或等价 scaler owner，保持 normalization artifact schema 不变。
- [x] 3.3 将 target provider、beam target source、auxiliary target setup 迁入 `deepsense6g_targets.py` 或等价 target owner，保持 `beam_target_source=current` 等契约不变。
- [x] 3.4 对 `MMWDataset` 只补充 monitor/暂缓 rationale 和测试护栏；除非有对应 focused coverage，不在本 wave 深拆 evolving split/label calibration 逻辑。
- [x] 3.5 将 `trainer._train_inner` 的 startup/build context、epoch loop、checkpoint coordination、final evaluation 和 artifact finalization 拆到现有 `training_state`、`training_metrics`、`checkpointing`、`trainer_runtime_helpers` 或少量新增 owner。
- [x] 3.6 保持 `kd_sensing.engine.trainer.train` 公开入口、checkpoint schema、history、final_config、resolved_config、run status 和 TensorBoard metadata 兼容。
- [x] 3.7 运行 `conda run -n kd_mm_beam pytest tests/test_deepsense6g_contract_helpers.py -q`。
- [x] 3.8 运行 `conda run -n kd_mm_beam pytest tests/test_gps_modality.py tests/test_lidar_modality.py tests/test_mmwave_modality.py tests/test_csi_modality.py -q`。
- [x] 3.9 运行 `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py tests/test_epoch_subsampling.py -q` 和 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。

## 4. Evaluation 与 Diagnostics Wave

- [x] 4.1 将 `run_evaluation_pass` 的 metric accumulation、objective auxiliary outputs、prediction metadata rows 和 optional modality quality report 抽成 schema-safe helper，保持 `EvaluationPassResult` 字段兼容。
- [x] 4.2 将 `run_deepsense6g_gps_lidar_bgam`、`run_mmw_town_gps_v2` 中 manifest loading、protocol dispatch、summary writer 和 plot handoff 登记为 monitor/split target；只在 focused tests 覆盖时实施拆分。
- [x] 4.3 将 `jepa_visual_analysis.py` 的 report/table/figure/cache/manifest 子职责登记并优先拆出不影响公开 CLI 的内部 owner。
- [x] 4.4 将 `run_index.py` 的 process/resource collection、artifact summary、CSV/render writer 和 `runtime_artifact_cleanup.py` 的 manifest/apply/render/organize 边界登记为二级热点，并为后续拆分补验证入口。
- [x] 4.5 保持 `jepa_benchmark_runner.py`、`jepa_benchmark_common.py`、`jepa_benchmark_scenario_d.py` 等 accepted owner 的 rationale；active predictive 语义稳定前不强拆 predictive 子域。
- [x] 4.6 运行 `conda run -n kd_mm_beam pytest tests/test_evaluation_pass.py tests/test_modality_difficulty.py -q`。
- [x] 4.7 运行 `conda run -n kd_mm_beam pytest tests/test_jepa_gps_shortcut_benchmark.py -q` 和 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。

## 5. 合并与 Import 面收口

- [x] 5.1 用 CodeGraph callers/callees 和 `rg` 字面路径扫描识别同 owner、单调用点、只服务 re-export 或无独立 public contract 的 helper 文件。
- [x] 5.2 合并确认的 `merge-candidate` helper，删除旧 helper 作为长期 owner 的暗示，不新增兼容 wrapper 或跨领域 `helpers.py`。
- [x] 5.3 对 `losses/jepa.py`、`losses/gps_lidar_bgam_losses.py`、`models/csi_encoder.py` 等小而内聚模块保持 keep-and-test，并在 index/inventory 记录理由。
- [x] 5.4 检查 `__init__.py`、config、registry、public facade 和 thin CLI 的 eager import，确保不新增 dataset/model/trainer/matplotlib/pandas/scipy/skimage/checkpoint 依赖泄漏。
- [x] 5.5 运行 `conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_config_load_characterization.py -q` 和 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。

## 6. 最终验收

- [x] 6.1 运行 `openspec validate right-size-project-architecture --strict`。
- [x] 6.2 运行 `openspec status --change right-size-project-architecture`，确认 artifacts 和任务状态可追踪。
- [x] 6.3 运行 `conda run -n kd_mm_beam pytest -q` 作为最终回归；若环境或本地数据限制导致无法完成，在最终说明中列出未运行原因和替代 focused 验证。
- [x] 6.4 检查 `git status --short`，确认没有将 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、`.pytest_cache` 或系统配置文件纳入源码变更。
