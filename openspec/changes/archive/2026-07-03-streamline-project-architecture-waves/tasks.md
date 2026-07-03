## 1. Wave 0 状态收口与基线捕获

- [x] 1.1 运行 `openspec list --json`，确认 `align-amber-amr-paper-architectures`、`add-scene31-next-round-experiments` 和其它 active change 的状态。
- [x] 1.2 对已完成 active change 执行归档，或在本 change 实施说明中记录明确 deferral、重叠范围和后续归档触发条件。
- [x] 1.3 运行 `git status --short` 和 `git diff --stat`，将现有用户/前序实验改动、未跟踪配置/脚本、本地 cache 噪声和本 change artifacts 分组记录。
- [x] 1.4 检查 `dataset/.gitkeep`、`.gitignore`、`.codegraph/`、`.codex/skills/`、未跟踪 `__pycache__` / `.pytest_cache` 状态，确认不会被本 change 误纳入架构 diff。
- [x] 1.5 运行 `openspec validate streamline-project-architecture-waves --strict`，修复 proposal/design/spec/tasks 的 schema 问题。
- [x] 1.6 运行 `openspec validate --all --strict`，记录全仓 OpenSpec baseline。
- [x] 1.7 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`，记录架构边界 baseline。
- [x] 1.8 运行 `conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_config_load_characterization.py -q`，记录 CLI/config smoke baseline。
- [x] 1.9 在 `docs/project_surface_inventory.md` 或本 change 实施说明中登记本 campaign 的 baseline、wave 顺序、验证矩阵和 rollback 规则。

## 2. Wave 1 Dataset Contract Adapter 化

- [x] 2.1 梳理 `DeepSense6GDataset` 和 `MMWDataset` 当前职责，列出 sample contract、modality readers、target providers、scaler/cache、metadata assembly、family-specific adapter 的目标 owner。
- [x] 2.2 提取或收敛 sequence sample core，覆盖 scene/root CSV 解析、portion sampling、sample cache key、return metadata 和 shared sample list 访问。
- [x] 2.3 提取 modality resource reader coordination，使 image/radar/GPS/LiDAR/mmWave/CSI 读取和 cache 策略不继续扩大 dataset 主体。
- [x] 2.4 提取 target provider coordination，使 beam target、soft label、occlusion、position、physics target 和 auxiliary target assembly 位于 target/sample helper。
- [x] 2.5 将 DeepSense6G scaler/normalizer setup 收敛到明确 owner，覆盖 GPS/mmWave/LiDAR/CSI/position target artifact save/load 兼容。
- [x] 2.6 为 MMW 建立 dataset-family adapter，迁出 condition layout、CSV 补列、beam label calibration、geometry、availability、radio semantic、path semantic、physical label 和 physics supervision setup。
- [x] 2.7 保留 `DATASETS.build({"type": "deepsense6g"})` 和 `DATASETS.build({"type": "mmw"})` public registry 行为；若保留 `MMWDataset` subclass，内部必须委托 adapter。
- [x] 2.8 更新 dataset/import 调用方，确保内部源码直接导入真实 owner module，不通过 package facade 或旧 helper 回流。
- [x] 2.9 增加或更新 synthetic/fixture focused tests，覆盖 DeepSense6G sample keys、MMW metadata/domain metadata、cache/scaler artifact 兼容和 adapter 输出 schema。
- [x] 2.10 运行 `conda run -n kd_mm_beam pytest tests/test_deepsense6g_contract_helpers.py tests/test_gps_modality.py tests/test_lidar_modality.py tests/test_mmwave_modality.py tests/test_csi_modality.py tests/test_mmw_town10_preparation.py -q`。
- [x] 2.11 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`，确认 dataset wave 未引入 facade 回流或产物边界回归。

## 3. Wave 2 Training / Evaluation Runtime 分层

- [x] 3.1 设计并实现 `TrainingRunContext` 或等价结构，承载 cfg、objective、run_dir、artifact writer、dataloaders、normalization artifacts、device/model/optimizer/scheduler/scaler、checkpoint manager、TensorBoard、extension 和 state。
- [x] 3.2 将 `_train_inner` 拆成 objective/run setup、resource build、checkpoint restore、epoch loop、validation/final evaluation、artifact finalization 和 resource shutdown phases。
- [x] 3.3 保持 `kd_sensing.engine.trainer.train(cfg)` public 行为、run directory、status file、checkpoint layout、`train_log.json` 和 `final_config.yaml` 兼容。
- [x] 3.4 将 extension、CSI RMS handoff、early stopping、TensorBoard startup scalars 和 final test evaluation 接入 phase helper，避免 `_train_inner` 吸收新增 suite-specific 分支。
- [x] 3.5 拆分 `run_evaluation_pass` 的 batch iteration、difficulty application、model step、objective labels、output recording、metadata recording、metric aggregation 和 prediction artifact schema。
- [x] 3.6 确认 `validator.validate`、`evaluator.evaluate`、diagnostics real-forward 和 training final-test evaluation 继续复用 shared evaluation pass。
- [x] 3.7 更新 failure-safe finalization，确保异常路径继续写 failed status、关闭 dataloaders/TensorBoard，并传播原始异常。
- [x] 3.8 更新 runtime focused tests，覆盖 training IO、prediction objectives、evaluation pass、modality difficulty 和 failure/finalization 关键路径。
- [x] 3.9 运行 `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py tests/test_prediction_objectives.py tests/test_evaluation_pass.py tests/test_modality_difficulty.py -q`。
- [x] 3.10 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。

## 4. Wave 3 ModularSequenceModel Forward 阶段化

- [x] 4.1 将 `ModularSequenceModel.forward` 拆为 raw/reliability input collection、encoder dependency resolution、encoder/projector execution、core input assembly、head execution、post-processing、diagnostics assembly 和 auxiliary output stages。
- [x] 4.2 保持 forward public signature、`adapt_model_output` 消费语义和既有 output keys 兼容。
- [x] 4.3 将 encoder dependency、context kwargs、reliability kwargs、runtime diagnostics、visual token diagnostics 和 temporal auxiliary metadata 逻辑收敛到 stage helper。
- [x] 4.4 将 spatial modality tokens、missing modality metadata、availability mask 和 token/core assembly 逻辑收敛到 stage helper。
- [x] 4.5 将 geometry prior、geometry fusion、safe residual rerank 和 branch diagnostics 逻辑收敛为 post-processing stage。
- [x] 4.6 将 feature consistency、token readout、AMBER auxiliary、auxiliary heads 和 metadata assembly 收敛到 diagnostics/auxiliary stage。
- [x] 4.7 确保新增组件通过 capability flags、metadata 和 existing hooks 接入，不向 main forward 添加 baseline-specific 硬编码分支。
- [x] 4.8 更新 model architecture summary 和 training strategy metadata tests，确认组件移动后 registry id、class path、role、参数量和 metadata 仍可审计。
- [x] 4.9 运行 `conda run -n kd_mm_beam pytest tests/test_modular_sequence_next_query_transformer.py tests/test_cls_token_transformer_fusion.py tests/test_geometry_prior_beam_fusion.py tests/test_amber_full_architecture.py tests/test_u_mask_beam_jepa.py -q`。
- [x] 4.10 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。

## 5. Wave 4 Diagnostics Runner 与 Runtime Artifact 模块收敛

- [x] 5.1 梳理 `jepa_benchmark_runner.py`、`jepa_visual_analysis.py`、`mmw_town_gps_v2.py`、`run_index.py`、runtime cleanup/organize owner 的 public API、CLI 和 manifest schema。
- [x] 5.2 将 JEPA benchmark runner 拆成 manifest/schema、metric row construction、suite aggregation、artifact planning、writers/plots 和 real-forward execution responsibilities。
- [x] 5.3 保持 `kd-sensing-jepa-gps-shortcut-benchmark` CLI、public runner return dict、`benchmark_manifest.json` 和核心 CSV/JSON/NPY output registration 兼容。
- [x] 5.4 将 predictive、Scenario C、Scenario D/CxD、geometry prior、fusion diagnostic 和 legacy compatibility rows 的 helper 保持在对应窄 owner，不回流 public facade。
- [x] 5.5 收敛 JEPA visual analysis 对 benchmark artifact 的读取路径，内部导入直接指向 benchmark owner modules。
- [x] 5.6 拆分或登记 `mmw_town_gps_v2` protocol dispatch、label-space resolution、summary writing 和 plot handoff。
- [x] 5.7 拆分或登记 `run_index` 的 process/resource collection、artifact summary、state classification、CSV/JSON/render writers。
- [x] 5.8 收敛 runtime cleanup/organize manifest 的 schema、scan、classification、protection、apply report 和 dry-run/execute 边界。
- [x] 5.9 运行 `conda run -n kd_mm_beam pytest tests/test_jepa_gps_shortcut_benchmark.py tests/test_jepa_visual_analysis.py tests/test_run_index.py tests/test_runtime_artifact_cleanup.py tests/test_runtime_output_layout.py -q`。
- [x] 5.10 运行 `conda run -n kd_mm_beam kd-sensing-jepa-gps-shortcut-benchmark --help`、`conda run -n kd_mm_beam kd-sensing-jepa-visual-analysis --help`、`conda run -n kd_mm_beam kd-sensing-runs --help`。
- [x] 5.11 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。

## 6. Wave 5 Config / Script / Entry Surface 收敛

- [x] 6.1 枚举 tracked 与未跟踪 `configs/scene31`、`configs/fusion`、diagnostics manifests 和 local/manual overlays，按 canonical/current、experiment reproduction、generated/local queue、diagnostics manifest、retired 分类。
- [x] 6.2 为规则化 Scene31/night-grid/next-round/seed sweep 配置族建立 recipe、manifest 或 generator owner，避免继续无限提交实体 YAML。
- [x] 6.3 为 generator 增加 sanity tests，校验 run name、seed、epoch、sampler、loss weights、missing pattern、difficulty profile、output boundary 和 manifest 行一致。
- [x] 6.4 删除或迁移可由 recipe 无损生成且不属于 current/canonical/reproduction/diagnostics 的重复实体 YAML。
- [x] 6.5 确认 virtual config 不接管 retired KD、BGAM、viewer、Hist、Raymobtime、AMR mock 或 JEPA-MSAC 路径。
- [x] 6.6 枚举 `scripts/` 与 `scripts/analysis/`，按 package CLI、数据准备、研究诊断、shell orchestration、local/manual helper 分类。
- [x] 6.7 删除重复 package CLI 的 Python thin alias；保留 local/manual runner 时登记 owner、输出边界、dry-run/sanity path 和删除条件。
- [x] 6.8 更新 README、docs/experiment_matrix、docs/project_surface_inventory 和 docs/agent_navigation 中的入口说明，确保不推荐 local/manual helper 作为长期 current CLI。
- [x] 6.9 运行 `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py tests/test_cli_help.py tests/test_scene31_next_round.py -q`。
- [x] 6.10 运行相关 console script help smoke：`conda run -n kd_mm_beam kd-sensing-train --help`、`conda run -n kd_mm_beam kd-sensing-evaluate --help`、`conda run -n kd_mm_beam kd-sensing-preprocess --help`。
- [x] 6.11 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。

## 7. Wave 6 Import Surface / Low-Value Helper Consolidation

- [x] 7.1 枚举 package `__init__.py`、public facade、thin wrapper、low-value `__all__`、single-call helper、重复 helper 聚合和内部 import 回流。
- [x] 7.2 删除未登记 public surface 的 re-export facade、thin wrapper、lazy export 和大型 internal `__all__` 镜像。
- [x] 7.3 将内部源码和测试导入迁移到真实 owner module，禁止通过 public facade 获取 private helper。
- [x] 7.4 合并单调用点、无 public contract、只服务 re-export 或只为降行数存在的 helper；不得新建跨领域 `helpers.py` 杂物间。
- [x] 7.5 收敛 registry helper surface，删除无调用方且已由 focused tests 覆盖的 runtime self-check 或旧 alias helper。
- [x] 7.6 删除退役整模型 direct import、alias 或 removed wrapper；保留仍被 current modular component 使用的 feature extractor。
- [x] 7.7 更新 architecture boundary tests，使其拒绝 facade 回流、重依赖 barrel、内部 package facade import 和未登记 whole-model exception。
- [x] 7.8 运行 `conda run -n kd_mm_beam pytest tests/test_component_registry.py tests/test_architecture_boundaries.py -q`。
- [x] 7.9 运行 `conda run -n kd_mm_beam pytest tests/test_model_architecture_summary.py tests/test_config_load_characterization.py -q`，若本地测试文件名不同则使用对应 architecture summary focused tests。

## 8. Wave 7 OpenSpec / Docs / Guardrails 收口

- [x] 8.1 审计 `docs/project_surface_inventory.md` 中 current/supporting/retired-tombstone lifecycle，确认每个保留 tombstone 的 guard 价值。
- [x] 8.2 对无 registry/config/CLI/docs/tests guard 且无迁移说明价值的 retired tombstone 进行归档或折叠到集中 retired summary。
- [x] 8.3 对仍保留的 tombstone，在 spec 开头或 inventory 中记录 guard 价值，避免文件名被误读为 current workflow。
- [x] 8.4 更新 `docs/agent_navigation.md`、`docs/project_surface_inventory.md`、`docs/maintainer_context_index.yaml` 的职责边界，使最小结构化事实不重新膨胀为完整目录镜像。
- [x] 8.5 更新 `openspec/specs/project-architecture`、`project-hotspot-governance`、`project-import-surface-consolidation`、`project-health-guardrails` 等 current specs 的 archive-ready wording。
- [x] 8.6 调整 `tests/test_architecture_boundaries.py`，保留结构性检查，删除重复 OpenSpec prose、完整 scripts allowlist、完整 config 数据库或完整 hotspot budget 镜像。
- [x] 8.7 运行 `openspec validate streamline-project-architecture-waves --strict`。
- [x] 8.8 运行 `openspec validate --all --strict`。
- [x] 8.9 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。

## 9. Cross-Wave Compatibility 与回归

- [x] 9.1 运行 `conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_config_load_characterization.py tests/test_component_registry.py tests/test_architecture_boundaries.py -q`。
- [x] 9.2 运行 dataset/runtime/model/diagnostics/config/script 所有已列 focused tests，确认没有 wave 间回归。
- [x] 9.3 运行 package CLI help smoke：`conda run -n kd_mm_beam kd-sensing-train --help`、`conda run -n kd_mm_beam kd-sensing-evaluate --help`、`conda run -n kd_mm_beam kd-sensing-preprocess --help`、`conda run -n kd_mm_beam kd-sensing-runs --help`。
- [x] 9.4 运行 diagnostics CLI help smoke：`conda run -n kd_mm_beam kd-sensing-jepa-visual-analysis --help`、`conda run -n kd_mm_beam kd-sensing-jepa-gps-shortcut-benchmark --help`、`conda run -n kd_mm_beam kd-sensing-clean-runtime-artifacts --help`、`conda run -n kd_mm_beam kd-sensing-organize-runtime-outputs --help`。
- [x] 9.5 运行 `conda run -n kd_mm_beam pytest -q` 作为最终回归。
- [x] 9.6 若全量 pytest 因环境、本地数据或耗时限制无法完成，记录未运行原因、已完成替代 focused 验证和剩余风险。
- [x] 9.7 运行 `git status --short`，确认未提交 `dataset/` 真实数据、`outputs/`、`logs/`、cache、checkpoint、TensorBoard event、`.pytest_cache`、`__pycache__` 或其它本地产物。
- [x] 9.8 更新最终实施说明，列出每个 wave 的完成状态、验证结果、internal breaking import surface、public behavior compatibility 和 rollback note。

## 10. Archive 准备

- [x] 10.1 确认所有 tasks 勾选完成，或对任何未完成项记录明确 deferral、原因、风险和后续 change 名称。
- [x] 10.2 运行 `openspec status --change streamline-project-architecture-waves`，确认 artifacts 和 tasks 状态。
- [x] 10.3 运行 `openspec validate streamline-project-architecture-waves --strict` 作为 archive 前校验。
- [x] 10.4 确认 current specs、inventory、README/docs 和 architecture boundary tests 已反映最终实现状态。
- [x] 10.5 准备归档说明，标明本 change 的 public behavior 保持项、internal breaking import 删除项、验证矩阵和未解决风险。
