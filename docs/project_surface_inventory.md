# 项目表面积 Inventory

本 inventory 记录 `refine-source-architecture-and-entry-surface` 的可审计基线。统计口径只覆盖源码、配置、文档和 OpenSpec artifact；`dataset/`、`outputs/`、`logs/`、cache、checkpoint、下载压缩包和其它本地运行产物不属于本 change 的处理范围。

## 项目健康护栏基线

`strengthen-project-health-guardrails` 于 2026-06-11 生成维护性基线；`prune-to-jepa-query-pool-surface` 收口后，当前 `src/kd_sensing` 约有 211 个 Python 文件，`tests/` 约有 49 个 Python 文件，`configs/` 约有 91 个 YAML；仓库根目录有 11 个 Markdown，`docs/` 有 10 个 Markdown。本基线只读扫描源码、测试、配置和文档，不读取真实 `dataset/` 数据，不写入 `outputs/`、`logs/`、cache、checkpoint 或本地训练产物。

分层健康检查命令如下：

- OpenSpec：`openspec validate strengthen-project-health-guardrails --strict`
- 架构边界与健康护栏：`conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`
- CLI/config smoke：`conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_config_load_characterization.py -q`
- 触碰训练、数据集、诊断、CLI、配置解析或模型 forward 时，追加对应 focused tests，并在 tasks 或最终说明中记录未运行项及原因。

新增热点维护规则：

- 新增长函数、长类、manifest builder、orchestration workflow 或兼容 facade 时，先拆到窄模块；若暂缓拆分，必须在本 inventory 记录文件、符号、规模指标、拆分方向和暂缓原因。
- 兼容 facade 只保留公开 import/CLI 语义；内部实现不得从 facade 回流导入 helper，应直接依赖职责明确的窄模块。
- 普通 pytest 文件依赖 `tests/conftest.py` 或 editable install 导入 `src/`；只有架构边界 import probe、subprocess smoke 或显式隔离环境测试可以在子进程代码中局部设置 `sys.path`。

当前 AST 热点清单如下；预算用于阻止热点静默扩大，不代表本 change 立即重构这些 runtime 逻辑：

| 文件 | 符号 | 类型 | 当前规模 | 推荐拆分方向 | 暂缓原因 / 优先级 |
| --- | --- | --- | --- | --- | --- |
| `src/kd_sensing/data/datasets/deepsense6g.py` | `DeepSense6GDataset` | 超长 dataset 类 | 1209 行 | scene/CSV audit、modality sample assembly、label/history adapters、cache/transform glue、BeamBench GPS Direct calibration glue、2604 GPS BEV XY coordinate glue | 数据契约和真实路径耦合强；近期为支持 `paper_distance_angle` scene calibration、BeamBench `beam_target_source=current` 和 BEV-Fusion 2604 `gps_bev_xy` raw coordinate 契约小幅扩容，后续仍优先拆分；优先级 P1 |
| `src/kd_sensing/data/datasets/deepsense6g.py` | `DeepSense6GDataset.__init__` | 超长初始化函数 | 261 行 | scene/dataframe audit、feature mode setup、cache policy、target source 和 modality flags 拆到窄 helper | 初始化契约与多条 current baseline/reproduction 配置耦合；本次只登记预算，不改数据语义；优先级 P1 |
| `src/kd_sensing/data/datasets/mmw.py` | `MMWDataset` | 超长 dataset 类 | 592 行 | manifest parsing、sequence window、label-space metadata、sensor feature loading | MMW group-safe split 与 label calibration 仍在演进；优先级 P1 |
| `src/kd_sensing/engine/trainer.py` | `_train_inner` | 超长训练 orchestration 函数 | 316 行 | dataloader setup、epoch loop、validation/checkpoint coordination、artifact finalization | 训练数值语义敏感；优先级 P1 |
| `src/kd_sensing/engine/mmw_town_gps_v2.py` | `run_mmw_town_gps_v2` | 超长诊断 workflow 函数 | 277 行 | protocol dispatch、label-space resolution、summary writing、plot handoff | MMW GPS v2 仍承担对照解释；优先级 P2 |
| `src/kd_sensing/baselines/beambench/image_ae_gps.py` | `run_image_ae_gps_training` | BeamBench Image AE+GPS workflow | 244 行 | AE train/load、fusion dataset build、metric/report writers | 复现实验入口需要保持 Table III 语义；本次只同步预算，不改训练语义；优先级 P1 |
| `src/kd_sensing/baselines/beambench/image_ae_gps.py` | `run_image_ae_gps_paper_split_training` | BeamBench Image AE+GPS workflow | 259 行 | scene split orchestration、checkpoint reuse、per-scene summary | 与本地 scene31-34 复现产物耦合；优先级 P1 |
| `src/kd_sensing/engine/deepsense6g_gps_lidar_bgam.py` | `run_deepsense6g_gps_lidar_bgam` | BGAM orchestration | 234 行 | manifest loading、ablation dispatch、summary writer | 需与 MMW BGAM contract 对齐后再拆；优先级 P2 |
| `src/kd_sensing/engine/evaluation_pass.py` | `run_evaluation_pass` | evaluation pass | 216 行 | metric aggregation、objective outputs、prediction metadata | evaluation schema 为多个 CLI 共享；优先级 P2 |
| `src/kd_sensing/engine/batch.py` | module helpers | batch preparation 热点 | 文件级 batch contract | modality target preparation、label adapters、history anchor inputs | 触碰会影响训练/验证公共 batch contract；优先级 P2 |
| `src/kd_sensing/diagnostics/run_index.py` | module helpers | 诊断 run index 热点 | 文件级诊断 contract | process/resource collection、artifact summary、CSV/render writers | 输出 schema 已在 README 暴露；优先级 P3 |
| `src/kd_sensing/diagnostics/viewer_manifest.py` | `export_viewer_manifest` facade/workflow | manifest builder 热点 | facade 预算 220 行 | schema、cache、path、merge、writer 窄模块继续保持职责分离 | 已拆第一批，后续只防回流；优先级 P3 |

## 配置生命周期分类

`configs/` 当前 89 个 YAML 按生命周期维护：

- canonical/current root configs：`configs/<image|radar|gps|lidar|mmwave|csi>/{strong,lightweight,supervised}.yaml`、`configs/deepsense6g_gps_adapter_v2.yaml`、`configs/deepsense6g_gps_lidar_bgam.yaml`、`configs/mmw_town_gps_adapter_v2.yaml`、`configs/mmw_town_gps_lidar_bgam.yaml`。
- canonical fusion root：`configs/fusion/*.yaml` 只保留长期 supervised、token-transformer/objective-aware 和 BeamBench Image AE+GPS thin/reproducibility 入口；具体 allowlist 见下文 `configs/fusion/` 根目录分类。
- experiment reproduction：`configs/fusion/experiments/jepa_image_gps/*.yaml` 记录 JEPA image+GPS、GPS-biased checkpoint reuse、GPS query pooling、2604/BeamBench fair 和必要 supervised/random-best 对照；这些不是 root canonical 入口。
- CSI experiment matrix：`configs/csi/hardening_matrix/*.yaml` 是主矩阵，`configs/csi/hardening_matrix/debug/*.yaml` 是 debug/smoke，`configs/fusion/csi_hardening_matrix/*.yaml` 是 GPS+CSI 验证矩阵。
- dataset preparation：`configs/preprocess/*.yaml` 服务当前数据准备、索引和 cache；DeepVerse/DT31 与 Raymobtime s008 预处理配置已删除。
- diagnostics：`configs/diagnostics/modality_visualization.yaml` 服务 viewer manifest/diagnostic；`configs/diagnostics/jepa_visual_analysis_2604.yaml` 服务 JEPA 离线分析；`configs/diagnostics/jepa_gps_shortcut_benchmark_*.yaml` 服务 JEPA vs GPS shortcut benchmark 和 Scenario D image observability smoke/canonical manifest。它们不是训练入口。
- baseline/reproduction：`configs/baselines/*.yaml` 和 `configs/pretraining/*.yaml` 服务 BeamBench reproduction 和 GPS-conditioned JEPA pretraining 复现；GPS window baseline 配置已删除。
- retired history：已删除的 KD、HiST/Hist、Top8 selector、GPS residual、camera residual、CRAF/MARF/G2D/Multimodal-NF 实体配置只允许作为历史或 migration guard 说明出现，不得作为当前推荐入口。

## OpenSpec capability lifecycle 分类

`openspec/specs/` 同时保存当前需求契约、支撑能力和退役墓碑。维护者和 AI agent 读取某个 capability 前，先按下表判断 lifecycle，再决定它是否代表当前入口：

- `current`：当前需求契约、可推荐入口或当前运行能力。
- `supporting`：被当前 workflow 消费的 helper、metric、manifest、cleanup、migration guard 或数据契约，不作为 standalone 推荐入口。
- `retired-tombstone`：只保留退役、拒绝、迁移边界和防回流说明，不属于当前支持能力。

| Capability | Lifecycle | 说明 |
| --- | --- | --- |
| `ai-maintainer-navigation` | `current` | 当前 agent/maintainer 导航契约。 |
| `automated-cache-policy` | `current` | 当前 cache policy 契约。 |
| `beam-distribution-shift-diagnostics` | `supporting` | source/target label 分布诊断支撑，不是独立训练入口。 |
| `beambench-baseline-reproduction` | `current` | BeamBench/Arnold22 baseline 复现与报告能力。 |
| `beamspace-physical-labels` | `supporting` | 物理标签、beam power/path label 和泄漏边界支撑。 |
| `bev-fusion-2604-reproduction` | `current` | 2604 paper-aligned BEV-Fusion 复现实验能力。 |
| `canonical-config-resolution` | `current` | 当前实体/virtual/overlay config 解析契约。 |
| `cls-token-transformer-fusion` | `current` | 当前 CLS-token Transformer fusion 模型能力。 |
| `component-registry` | `current` | 当前 registry 与退役组件拒绝边界。 |
| `configurable-multimodal-fusion` | `current` | 当前 fusion 配置和 virtual overlay 运行语义。 |
| `cross-scene-loso-workflow` | `supporting` | 通用 LOSO fold/split/few-shot 支撑；Hist 默认 runner 已退役。 |
| `csi-channel-data` | `current` | 当前 CSI 数据列和样本字段契约。 |
| `csi-channel-degradation` | `current` | 当前 CSI degradation profile 能力。 |
| `csi-hardening-debug-validation` | `current` | 当前 CSI hardening debug 矩阵验证。 |
| `csi-hardening-experiment-matrix` | `current` | 当前 CSI hardening 控制变量矩阵。 |
| `csi-modality-model` | `current` | 当前 CSI 模态模型契约。 |
| `dataset-directory-layout` | `current` | 当前数据集目录和本地产物边界。 |
| `dataset-runtime-contracts` | `current` | 当前 dataset descriptor/sample/adapter/target provider 契约。 |
| `deepsense6g-camera-ae-residual-correction` | `retired-tombstone` | DeepSense6G camera residual 路线已退役。 |
| `deepsense6g-gps-lidar-bgam-reranker` | `current` | 当前 DeepSense6G GPS+LiDAR BGAM reranker workflow。 |
| `deepsense6g-gps-residual-fusion` | `retired-tombstone` | DeepSense6G GPS residual fusion 路线已退役。 |
| `deepsense6g-gps-top8-candidate-selector` | `supporting` | 旧 standalone selector 已退役；BGAM 可复用 TopK manifest/dataset/loss 支撑语义。 |
| `deepsense6g-scene-selection` | `current` | 当前 DeepSense6G scene 选择与输出隔离。 |
| `distillation-free-project-surface` | `current` | 当前去蒸馏化项目表面和旧 KD 拒绝契约。 |
| `experiment-artifact-registry` | `current` | 当前 checkpoint/artifact registry 与历史产物隔离。 |
| `experiment-run-index` | `current` | 当前只读 run index 能力。 |
| `experiment-workflow` | `current` | 当前配置驱动训练/评估/预处理/诊断 workflow。 |
| `first-class-prediction-tasks` | `current` | 当前 prediction objective 元数据、loss 和指标契约。 |
| `geometry-residual-beam-labels` | `retired-tombstone` | geometry residual label 路线已退役。 |
| `ieee-11282996-gps-image-reproduction` | `current` | AMR-Net_gps_image source-audit、GPS+Image-only local substitute 和 blocked official claim 边界。 |
| `gps-coarse-anchor-prediction` | `retired-tombstone` | GPS coarse anchor prediction 已退役。 |
| `gps-conditioned-jepa-pretraining` | `current` | 当前 GPS-conditioned JEPA 预训练能力。 |
| `gps-modality-model` | `current` | 当前 GPS 模态模型契约。 |
| `gps-preprocessing` | `current` | 当前 GPS sequence/relative-polar/scaler 预处理契约。 |
| `gps-pseudo-label-bgam` | `current` | 当前 BGAM pseudo-history label 与 candidate 输入契约。 |
| `gps-query-jepa-pooling` | `current` | 当前 Image+GPS JEPA query-pool 下游能力。 |
| `hist-beam-cross-scene-adaptation` | `retired-tombstone` | HiST/Hist 跨场景适配研究线已退役。 |
| `history-anchored-residual-beam` | `retired-tombstone` | history-anchor Hist 路径已退役。 |
| `image-only-legal-crossroad-probe` | `retired-tombstone` | image-only Hist probe 已退役。 |
| `image-preprocessing-profiles` | `current` | 当前 image profile 与 RGB/ImageNet 预处理契约。 |
| `jepa-downstream-extensibility` | `current` | 当前 JEPA downstream pooler/adapter 扩展契约。 |
| `jepa-gps-shortcut-benchmark` | `current` | 当前 JEPA vs GPS shortcut benchmark 能力。 |
| `jepa-visual-analysis-suite` | `current` | 当前 JEPA visual analysis 离线诊断能力。 |
| `legacy-kd-isolation` | `retired-tombstone` | legacy KD 入口只作为拒绝、历史读取和 migration guard 墓碑。 |
| `lidar-modality-model` | `current` | 当前 LiDAR 模态模型契约。 |
| `lidar-preprocessing` | `current` | 当前 LiDAR 点云/BEV cache/scaler 预处理契约。 |
| `mainline-experiment-documentation` | `current` | 当前主线模型目录、实验协议表和结果/claim 账本治理能力；不替代 OpenSpec 行为契约。 |
| `mmw-beam-label-calibration` | `current` | 当前 MMW beam label calibration 契约。 |
| `mmw-cross-scene-adaptation-protocol` | `supporting` | MMW split/adaptation protocol 支撑；MMW HiST wording 只可作历史边界。 |
| `mmw-sensor-assisted-beam-prediction` | `current` | 当前 MMW sensor-assisted beam prediction 边界。 |
| `mmw-town-gps-adapter-v2` | `current` | 当前 MMW Town GPS-only v2 workflow。 |
| `mmw-town-gps-lidar-bgam-reranker` | `current` | 当前 MMW Town GPS+LiDAR BGAM reranker workflow。 |
| `mmw-town-gps-top8-candidate-selector` | `supporting` | standalone Top8 CLI/config 已退役；BGAM 内部候选 manifest 支撑可保留。 |
| `mmw-town10-dataset-preparation` | `current` | 当前 MMW Town10 数据准备能力。 |
| `mmwave-modality-model` | `current` | 当前 mmWave 模态模型契约。 |
| `mmwave-preprocessing` | `current` | 当前 mmWave sequence/power/scaler 预处理契约。 |
| `model-architecture-extension-contract` | `current` | 当前新增 baseline、模块化组件、整模型例外、workflow reproduction 和 metadata 护栏契约。 |
| `modality-aware-data-loading` | `current` | 当前按模态选择加载、normalization 和 cache 行为契约。 |
| `modality-contracts` | `current` | 当前中心化模态顺序、batch key 和 profile 拒绝边界。 |
| `modality-difficulty-pipeline` | `current` | 当前 difficulty profile/operator pipeline。 |
| `modality-visual-diagnostics` | `current` | 当前 viewer manifest 兼容诊断入口。 |
| `modular-sequence-model` | `current` | 当前模块化序列模型结构。 |
| `multi-task-occlusion-position-learning` | `current` | 当前 occlusion/position 辅助监督能力。 |
| `observability-aware-fusion` | `current` | 当前 image/GPS reliability metadata 自适应融合、uncertainty gating 和 JEPA fallback 契约。 |
| `original-code-compatibility` | `supporting` | 历史 checkpoint/config 读取兼容支撑，不是旧训练入口。 |
| `path-prototype-hist-beam-adaptation` | `retired-tombstone` | P3/HiST path prototype 已退役。 |
| `project-architecture` | `current` | 当前包结构、入口、轻量导入和退役边界。 |
| `project-health-guardrails` | `current` | 当前健康护栏、inventory 和静态检查契约。 |
| `project-surface-cleanup` | `current` | 当前源码表面清理、退役路线和本地产物边界。 |
| `spec-lifecycle-boundaries` | `current` | 当前 OpenSpec capability lifecycle 分类和读取边界。 |
| `radar-student-model` | `current` | 当前 radar student 模型契约。 |
| `radar-teacher-model` | `current` | 当前 radar teacher/复现兼容模型契约。 |
| `radio-semantic-hist-beam-adaptation` | `retired-tombstone` | radio-semantic Hist 路线已退役。 |
| `raymobtime-s008-retirement` | `retired-tombstone` | Raymobtime s008 退役墓碑。 |
| `resnet18-image-encoder` | `current` | 当前 ResNet-18 ImageNet encoder 能力。 |
| `runtime-artifact-cleanup` | `current` | 当前只读 cleanup manifest 和显式删除 workflow。 |
| `scenario-d-image-observability-benchmark` | `current` | 当前 Scenario D 图像可观测性等级、CxD benchmark 矩阵和输出产物边界。 |
| `snapshot-next-frame-baselines` | `current` | 当前 snapshot next-frame baseline 契约。 |
| `soft-beam-label-training` | `current` | 当前 circular/soft beam label supervised training。 |
| `target-shot-domain-splitting` | `current` | 当前 source-target domain 和 target-shot split 契约。 |
| `training-throughput-optimization` | `current` | 当前训练吞吐 profiling 与建议。 |
| `vision-position-baseline-suite` | `current` | 当前 Vision-Position baseline preset 矩阵。 |

退役/支撑例外摘要：

- HiST/Hist、Raymobtime s008、GPS coarse anchor、GPS residual、camera residual、geometry residual、image-only Hist probe、P3/Radio-semantic Hist、legacy KD、CRAF/MARF/G2D 和 Multimodal-NF 不属于当前推荐入口；只能在退役墓碑、migration guard、历史说明或拒绝边界中出现。
- Top8/TopK 的旧 standalone selector 训练、plot、compare CLI/config 已退役；BGAM 当前 workflow 可继续消费 GPS logits 重新计算出的 candidate manifest、TopK loss/metric 或字段映射支撑语义。
- 通用 LOSO/few-shot split、beamspace physical label、distribution diagnostics、runtime cleanup 和历史 checkpoint 读取属于 supporting 或 current guardrail；它们不得恢复 Hist、Raymobtime、旧 KD 或 residual 研究线的旧 CLI、root config、console script 或实体 YAML。

模型扩展路径分类：

- config-only baseline：只改 YAML、virtual recipe、overlay 或 hyperparameter；普通 supervised/adaptation baseline 的首选路径。
- component baseline：新增或替换 `ENCODERS`、`PROJECTORS`、`REPRESENTATION_CORES` 或 `HEADS` 子组件，并通过 `model.primary.type: modular_sequence` 选择。
- whole-model exception：新增完整 `@MODELS.register(...)` 时必须有 current spec、active change artifact、inventory 或测试 allowlist 中的明确例外说明，并覆盖 registry build、forward/output adaptation 和 `training_strategy_metadata()`。
- workflow/paper reproduction：BeamBench AE+GPS、BGAM、官方协议包装、多阶段训练或特殊报告产物应位于 `src/kd_sensing/baselines/<family>/`、包内 CLI 或已登记薄 alias；它们不是普通模块化 baseline，不得复制通用训练循环或恢复旧入口。

## 文档生命周期分类

根目录 Markdown：

- current quickstart and short index：`README.md`；只保留安装、主要入口、quickstart、数据/产物边界和文档索引，不复制完整模型目录、协议表或结果账本。
- agent/developer operating rules：`AGENTS.md`。
- environment/data setup：`ENVIRONMENT.md`、`DATASET_STRUCTURE.md`。
- current reproducibility/reporting：`README_REPRODUCE.md` 提供 BeamBench/Arnold22 当前推荐命令并指向 current summary；`BASELINE_REPORT.md` 开头维护 current summary、claim status 和 caveat；`results/reproduce_baseline.md` 是历史流水账，不能覆盖 current summary。
- historical research notes：`TODO_FOR_ATTENTION_MODULE.md`、`deep-research-report.md`、`PATCH_NOTES.md`、`跨场景自适应方案.md`、`跨场景自适应方案_融合推理修改版.md`。这些文档只能作为历史背景，不得把退役 KD/HiST/Top8/residual/camera residual 路线重新描述为当前推荐入口。

`docs/` Markdown：

- current agent/maintainer navigation：`docs/agent_navigation.md`；它只提供修改前权威来源、当前状态、任务路由、误读边界和验证选择导航，不替代 README、AGENTS 或 OpenSpec specs，也不维护完整项目表面积审计。
- current architecture/health inventory：`docs/project_surface_inventory.md`；记录 capability lifecycle、文档生命周期和源码/配置/入口表面积，不承担实验参数横向比较。
- current mainline workflow facts：`docs/mainline_model_catalog.md` 维护当前主线模型、baseline/control、诊断和 benchmark 目录；`docs/experiment_protocols.md` 维护 formal/lowmem/smoke/debug/upper-bound/historical ablation 参数口径；`docs/result_claims_registry.md` 维护可引用结果、blocked official、本地 substitute、upper-bound、mock/smoke 和 historical ablation 的 claim provenance。
- current workflow quickstart：`docs/experiment_matrix.md`、`docs/extension_guide.md`、`docs/training_throughput.md`；`docs/experiment_matrix.md` 只保留推荐顺序、入口命令和关键 caveat，并指向三份 current mainline 文档。
- dataset/diagnostic focused notes：`docs/research_notes.md`。
- historical analysis：`docs/p3_v7_multisource_crossroad_analysis.md`，只保留研究背景，不作为当前长期入口。

## 源码热点模块

本批次拆分的热点 facade 与职责模块如下：

- 模型扩展热点优先落在 `src/kd_sensing/models/modular.py` 的模块化子组件、`src/kd_sensing/models/image_encoders.py` / `jepa.py` 等已有 encoder 或窄模型文件，以及 `src/kd_sensing/registries.py` 的轻量 registry。普通 baseline 不应新增 dataset 解析、训练循环、validation loop、专用 `prepare_*` 或 `forward_task_model` 分支。
- `ObservabilityAwareFusion` / reliability-aware fusion 长期视为显式 opt-in adaptive fusion helper 或可组合 representation core 候选；普通 early-concat、CLS-token transformer、JEPA 和 Vision-Position baseline 不应被静默替换语义。消费 reliability metadata 的模型必须在 run metadata 或 `training_strategy_metadata()` 中标记。

- Raymobtime s008 dataset、预处理器、selection 模型、配置和 focused test 已退役删除；旧 registry 名称和配置路径只保留 migration guard 错误信息。
- `src/kd_sensing/models/csi.py` 保留公开 import 路径；pilot estimation、CSI hardening、view tokenizer/fusion、debug helpers 和 encoder registry glue 分别迁移到 `csi_estimation.py`、`csi_hardening.py`、`csi_views.py`、`csi_debug.py`、`csi_encoder.py`。
- `src/kd_sensing/engine/objective_metadata.py` 保留公开兼容 facade；objective 名称、默认 metric、metric alias 和 mode 表迁移到 `src/kd_sensing/engine/objectives/registry.py`，history fields 与 TensorBoard scalar schema 迁移到 `src/kd_sensing/engine/objectives/history.py`，runtime metadata/validation helper 在 `src/kd_sensing/engine/objectives/metadata.py`。
- `src/kd_sensing/diagnostics/viewer_manifest.py` 保留 manifest 导出公开 orchestration；配置解析、dataset 构建、采样、statistics、sample id/JSON schema、cache metadata、row path resolution、prediction/quality/gate merge 和 asset writer 分别位于 `src/kd_sensing/diagnostics/viewer_manifest_config.py`、`src/kd_sensing/diagnostics/viewer_manifest_datasets.py`、`src/kd_sensing/diagnostics/viewer_manifest_sampling.py`、`src/kd_sensing/diagnostics/viewer_manifest_stats.py`、`src/kd_sensing/diagnostics/viewer_manifest_schema.py`、`src/kd_sensing/diagnostics/viewer_manifest_cache.py`、`src/kd_sensing/diagnostics/viewer_manifest_paths.py`、`src/kd_sensing/diagnostics/viewer_manifest_merge.py` 和 `src/kd_sensing/diagnostics/viewer_manifest_writer.py`。
- `src/kd_sensing/diagnostics/jepa_gps_shortcut_benchmark.py` 提供 JEPA vs GPS shortcut benchmark manifest schema、deterministic perturbation helper、metrics aggregation、runner manifest writer 和 visual-analysis ingestion bundle helper；默认输出限定在 `outputs/analysis/`，不读取真实 `dataset/`，除非用户显式用真实 config/weights 执行评估计划。
- DeepVerse/DT31 generator、label builder、split、sanity check 和 focused test 已退役删除；MMW beam power 所需的通用 ULA/DFT codebook helper 保留在 `src/kd_sensing/data/beam_codebook.py`。
- `src/kd_sensing/data/mmw/preparation.py` 保留 Town10/Skybridge MMW preparation 公开 orchestration 和兼容导出；配置 schema、默认常量和 override loading 迁移到 `src/kd_sensing/data/mmw/preparation_config.py`，zip/input audit、extract marker、availability report 迁移到 `src/kd_sensing/data/mmw/preparation_audit.py`，sensor/channel indexing 与 path parsing 迁移到 `src/kd_sensing/data/mmw/preparation_index.py`，sequence row、group-safe split、guard band 和 leakage diagnostics 迁移到 `src/kd_sensing/data/mmw/preparation_splits.py`，channel payload、DFT/codebook beam power 和 power validation 迁移到 `src/kd_sensing/data/mmw/preparation_beam_power.py`，manifest/split/report 写出迁移到 `src/kd_sensing/data/mmw/preparation_writers.py`，relative geometry、pose/proxy features 和 azimuth bin helper 迁移到 `src/kd_sensing/data/mmw/preparation_geometry.py`。

新增内部代码不得从 `kd_sensing.engine.objective_metadata` 或 `kd_sensing.data.mmw.preparation` 回流导入窄 helper；应直接使用上面的窄模块。`kd_sensing.diagnostics.viewer_manifest` 和 `kd_sensing.data.mmw.preparation` 可作为公开 orchestration/import 入口，但内部 helper 引用应分别指向 `viewer_manifest_*` 与 `preparation_*` 窄模块。

## 第二梯队热点

第二梯队热点先纳入 inventory 和架构 review 清单，不在本批次做大规模行为改写：

- HiST-Beam engine/model/evaluation 专用源码已退役并从当前支持面删除；旧 registry 名称和配置路径只保留 migration guard 错误信息。
- `src/kd_sensing/diagnostics/run_index.py`：后续优先抽出 process/resource collection、artifact summary、CSV/render writers；当前保持诊断输出 schema 兼容。
- `src/kd_sensing/data/transform_ops/csi.py`：后续优先抽出 CSI parsing、hardening feature transforms 和 temporal window helpers；当前避免同时改动数据契约。
- `src/kd_sensing/engine/batch.py`：后续优先抽出 modality target preparation、label adapters 和 history anchor input helper；当前保持训练 batch contract。
- `src/kd_sensing/engine/evaluation_pass.py`：后续优先抽出 metrics aggregation、objective-specific outputs 和 prediction metadata helper；当前保持 evaluation result schema。
- `src/kd_sensing/engine/loso_data.py`：当前保留 LOSO source/target dataloader helper。结构检查未发现当前内部调用，但该模块通过 `__all__` 暴露 `build_loso_dataloaders`、`build_loso_source_train_loader` 和 `build_loso_target_stage_loader`，且当前 OpenSpec 仍保留 cross-scene LOSO/few-shot sampling 语义；若后续完全退役，应另起 change 同步 specs、docs 和外部兼容说明。

本次已删除高置信孤立源码：`src/kd_sensing/evaluation/flops.py`、`src/kd_sensing/evaluation/latency.py` 和 `src/kd_sensing/data/transform_ops/cache.py`。删除前检查确认它们不属于 console script、package `__init__` 公开导出、注册入口、README/docs/OpenSpec 当前声明或测试依赖；image/LiDAR cache 的当前实现入口继续分别位于 `src/kd_sensing/data/transform_ops/image_cache.py` 和 `src/kd_sensing/data/transform_ops/lidar.py`。

## 配置 YAML

当前 `configs/fusion/` 根目录有 12 个实体 YAML，只保留长期 canonical 或当前明确薄入口。`configs/fusion/experiments/jepa_image_gps/` 有 11 个 JEPA image+GPS 实验特化 YAML，用于 BeamBench-fair、arXiv:2604.05668 对齐、GPS-biased/GPS-query checkpoint reuse、必要 supervised/random-best 对照和保留的 BeamBench fair 复查配置；其中 GPS-biased/GPS-query checkpoint reuse 是 JEPA 下游主线。这些路径可被文档指向，但不算根目录推荐入口。`configs/diagnostics/jepa_gps_shortcut_benchmark_smoke.yaml` 是不读真实数据的 benchmark smoke manifest，`configs/diagnostics/jepa_gps_shortcut_benchmark_beambench_fair.yaml` 是引用现有 Vision-Position baseline 与 JEPA downstream 配置的 canonical benchmark manifest，checkpoint 路径为本地占位；`configs/diagnostics/jepa_gps_shortcut_benchmark_scenario_d_smoke.yaml` 是 Scenario D / CxD image observability smoke manifest，默认输出到 ignored `outputs/analysis/scenario_d_image_observability/smoke`，不提交 CSV/NPY/PNG。`configs/csi/hardening_matrix/` 有 13 个主矩阵 YAML，`configs/csi/hardening_matrix/debug/` 有 5 个 debug YAML；`configs/fusion/csi_hardening_matrix/` 有 4 个 GPS+CSI 验证矩阵 YAML。

`configs/fusion/` 根目录保留分类如下：

- canonical strong/current supervised: `all_modalities_lidar_supervised.yaml`、`all_modalities_supervised.yaml`、`image_gps_supervised.yaml`、`image_gps_resnet18_modular_supervised.yaml`、`mmwave_csi_supervised.yaml`、`mmwave_csi_medium_degraded_supervised.yaml`、`radar_gps_supervised.yaml`、`radar_lidar_supervised.yaml`。
- current thin/reproducibility entry: `beambench_image_ae_gps_direct.yaml`。
- current token-transformer/objective-aware entries: `token_transformer_all_modalities_supervised.yaml`、`token_transformer_all_modalities_multitask_supervised.yaml`、`token_transformer_image_radar_supervised.yaml`。

已迁移到 `configs/fusion/experiments/jepa_image_gps/` 的实验特化配置如下：

- fair/2604 当前文档复核配置：`image_gps_jepa_gps_biased_best_beambench_fair_lowmem.yaml`、`image_gps_jepa_gps_biased_best_2604_s32_s34_lowmem.yaml`、`image_gps_jepa_gps_query_pool_best_beambench_fair_lowmem.yaml`、`image_gps_jepa_gps_query_pool_best_2604_s32_s34_lowmem.yaml` 和 `image_gps_jepa_gps_query_pool_best_2604_s32_s34_fasttrain.yaml` 是主线或快速复核主线；`image_gps_supervised_beambench_fair_lowmem.yaml`、`image_gps_jepa_random_best_beambench_fair_lowmem.yaml`、`image_gps_supervised_2604_s32_s34_lowmem.yaml`、`image_gps_jepa_random_best_2604_s32_s34_lowmem.yaml` 是对照。保留 `beambench_fair` 文件名的配置现在对齐 BeamBench Table III 的输入/split/target/metric 口径：`seq_len=1`、`num_pred=1`、`beam_target_source=current`、GPS `paper_distance_angle`、scene paper calibration angle、S32-S34 train、S31-S34 test 和 linear DBA；它们仍是 Image+GPS/JEPA 下游模型，不是 Table III Camera AE+GPS Direct 模型。
- BeamBench fair 保留复查配置：`image_gps_jepa_random_last_beambench_fair_lowmem.yaml` 和 `image_gps_jepa_gps_biased_pooler_param_groups_beambench_fair_lowmem.yaml`。
- 已退役删除配置：scene31-only low-memory/best-last 配置、非 BeamBench last-checkpoint 配置，以及 `jepa_gru.yaml`、`jepa_snapshot.yaml`、`jepa_plain_token_transformer.yaml`、`jepa_next_query_transformer.yaml` next-beam downstream ablation 配置。

已退役的 CRAF、MARF、G2D、Multimodal-NF 和 KD 实体 YAML、overlay recipe 与 virtual alias 不再作为支持入口存在。删除实体文件后，配置加载器只为当前 strong/lightweight canonical、snapshot、objective-aware、Vision-Position baseline preset 和保留 overlay 生成 virtual config，不接管退役路径；旧 `logits_kd` / `rkd` 路径只作为 migration guard 的拒绝命中保留。Vision-Position 当前 virtual preset 为 `configs/fusion/{camera_ae_gps,resnet_gps,transformer_image_gps,gps_only_neural}.yaml`，默认使用 BeamBench-style `seq_len=1`、`num_pred=1`、`paper_distance_angle`、`beam_target_source=current` 和 linear DBA 口径；这些 preset 只是项目对照，不得作为 Arnold22 Table III row 的数值复现入口。`gps_only_neural` 不是论文 GPS `Classical*` 或 `Dense†` 行；Table III Camera AE+GPS row 只能走 `configs/fusion/beambench_image_ae_gps_direct.yaml` 和 `scripts/run_beambench_image_ae_gps_tableiii.py`。

## 脚本入口 Allowlist

保留入口按 lifecycle 分类如下；新增 `scripts/` 或 `tools/analysis/` 下的 Python/shell 文件必须同步更新本 inventory 和 `tests/test_architecture_boundaries.py`。

- package_cli: `kd_sensing.cli.jepa_visual_analysis`、`kd_sensing.cli.jepa_gps_shortcut_benchmark`、`kd_sensing.cli.prepare_deepsense6g_gps_lidar_bgam_manifest`、`kd_sensing.cli.run_deepsense6g_gps_lidar_bgam`、`kd_sensing.cli.evaluate_deepsense6g_gps_lidar_bgam`、`kd_sensing.cli.prepare_mmw_town_gps_lidar_bgam_manifest`、`kd_sensing.cli.run_mmw_town_gps_lidar_bgam`、`kd_sensing.cli.evaluate_mmw_town_gps_lidar_bgam`、`kd_sensing.cli.run_amr_net_gps_image`。JEPA visual analysis 与 GPS shortcut benchmark 是只读模型/benchmark 产物的诊断入口，输出限定在 ignored 的 `outputs/visual_analysis/` 或 `outputs/analysis/`；GPS+LiDAR BGAM 是保留的包内入口；AMR-Net_gps_image 入口只写 source-audit/report/mock manifest 到 ignored `outputs/analysis/amr_net_gps_image/`，不启动真实长训练、不启用 LiDAR、不声明 official reproduction。GPS coarse anchor、Top8 selector、DeepSense6G residual 和 camera residual 入口已退役，不再作为当前 package CLI。
- thin_cli_alias: `scripts/train.py`、`scripts/evaluate.py`、`scripts/preprocess.py`、`scripts/train_baseline.py`、`scripts/eval_baseline.py`、`scripts/train_beambench_image_ae_gps.py`、`scripts/run_beambench_image_ae_gps_tableiii.py`。前三者只委托包内主训练/评估/预处理 CLI；README 推荐 `kd-sensing-train`、`kd-sensing-evaluate` 和 `kd-sensing-preprocess`。`scripts/train_baseline.py` 和 `scripts/eval_baseline.py` 是 BeamBench baseline 审计/mock smoke 薄入口，`scripts/train_beambench_image_ae_gps.py` 是 Arnold22 BeamBench Table III `Camera=AE, GPS=Direct, Fusion=Yes` 本地训练薄入口，`scripts/run_beambench_image_ae_gps_tableiii.py` 是四场景 Table III 本地复现实验薄入口；主要实现位于 `src/kd_sensing/baselines/beambench/` 和 `src/kd_sensing/cli/`，默认输出限定在 ignored 的 `outputs/evaluations/beambench_baseline/`、`outputs/scene<id>/beambench_image_ae_gps_direct/` 或 `outputs/scenegroup_s32_s34/beambench_image_ae_gps_direct_tableiii/`。
- research_diagnostic: `scripts/analyze_csi_hardening_sweep.py`、`scripts/analysis/beambench_ae_gps_diagnostics.py`、`scripts/analysis/deepsense_gps_v2_support_sweep_artifacts.py`、`scripts/analysis/visualize_deepsense_beambench_correspondence.py`、`scripts/debug_eval_consistency.py`、`scripts/figures/draw_jepa_architecture.py`、`scripts/profile_training_io.py`、`scripts/recommend_parallel_training.py`、`scripts/mmw/visualize_gps_angle_beam_correspondence.py`、`scripts/mmw/visualize_gps_prediction_trajectory.py`、`scripts/mmw/visualize_prediction_error_label_distribution.py`。旧模态子集/扰动研究脚本不再作为长期入口；通用 subset/mask 验证保留在 `kd-sensing-evaluate` 使用的共享 evaluation pass 与配置化 `evaluation.modality_subsets` 中。
- dataset_preparation: `scripts/inspect_dataset.py`、`scripts/check_dataset.py`、`scripts/mmw/prepare_town10_skybridge.py`、`scripts/mmw/build_sequence_splits_from_manifest.py`、`scripts/mmw/visualize_town_label_distribution.py`。
- shell_orchestration: `scripts/run_csi_hardening_matrix.sh`、`scripts/run_deepsense_gps_circular_soft_label.sh`、`scripts/run_mmw_gps_circular_soft_label_ablation.sh`、`scripts/run_mmw_sunny_modal15_l5p3_h123.sh`、`scripts/run_mmw_sunny_modal15_l5p6_h246.sh`。

MMW 入口生命周期说明：

- `scripts/check_dataset.py` 属于 dataset_preparation。职责是只读检查 BeamBench/DeepSense6G CSV 字段、传感器路径引用、beam label 范围以及 scene/sample/sequence/timestamp 标识解析；输出可写入显式 JSON 报告，不移动、不删除、不生成真实数据。
- `scripts/train_baseline.py` 和 `scripts/eval_baseline.py` 属于 thin_cli_alias。职责是委托 `kd_sensing.baselines.beambench` 中的 BeamBench 复现实现：前者打通 mock train/eval/checkpoint smoke，后者生成官方 `challenge.py` 评估计划或执行 mock checkpoint 评估。真实官方评估只有在官方数据、权重、源码和环境齐备且显式传入 `--execute` 时才运行。
- `scripts/train_beambench_image_ae_gps.py` 属于 thin_cli_alias。职责是委托 `kd_sensing.baselines.beambench.image_ae_gps` 中的论文 row 专用实现：从本地 DeepSense6G scene31-34 sequence CSV 读取 camera/GPS/current beam target，先训练或加载 Camera AE，再冻结 AE encoder，使用官方 BeamBench `dense_model` 等价 head（Camera AE latent + GPS Direct、Sigmoid+BCE）训练 fusion classifier，输出 checkpoint、history、predictions 和 BeamBench DBA/top-k metrics；输出限定在 `outputs/scene<id>/` 或显式用户路径下，不得提交新 checkpoint、日志或 predictions。
- `scripts/run_beambench_image_ae_gps_tableiii.py` 属于 thin_cli_alias。职责是委托 `kd_sensing.cli.run_beambench_image_ae_gps_tableiii`，顺序运行 scene31-34 的 Camera AE + GPS Direct 本地复现实验并输出 Table III 风格 CSV/Markdown/JSON 汇总；默认输出限定在 `outputs/scenegroup_s32_s34/`，评估-only 汇总可写入 `outputs/evaluations/`，不得提交新 checkpoint、feature cache、predictions 或 summary runtime artifact。
- `scripts/analysis/beambench_ae_gps_diagnostics.py` 属于 research_diagnostic。职责是读取本地 BeamBench AE+GPS 复现实验产物，汇总训练历史、预测和指标诊断，辅助分析 Camera AE + GPS Direct row 的本地复现差异；输出限定为 `outputs/analysis/` 等显式诊断路径，不得提交生成统计、图表或 checkpoint。
- `scripts/analysis/visualize_deepsense_beambench_correspondence.py` 属于 research_diagnostic。职责是读取本地 DeepSense6G scene31-34 原始 scenario CSV、GPS 和 beam labels，输出 BeamBench Fig.2 风格的 calibrated GPS angle 与 centered beam index 空间对应图；输出限定为 `outputs/analysis/` 等显式本地诊断路径，不得提交生成图片或统计产物。
- `scripts/analysis/deepsense_gps_v2_support_sweep_artifacts.py` 属于 research_diagnostic。职责是汇总 DeepSense6G GPS v2 support sweep 本地 artifact，输出只限 `outputs/analysis/` 等显式诊断路径，不得提交生成统计或图表产物。
- `scripts/figures/draw_jepa_architecture.py` 属于 research_diagnostic。职责是生成 JEPA pretraining/downstream reuse 架构示意图，输出限定为 `docs/figures/` 或显式本地图形路径；不得提交由临时运行产生的缓存、checkpoint 或训练产物。
- `scripts/mmw/build_sequence_splits_from_manifest.py` 属于 dataset_preparation。职责是在已有 `Prepared/<scene>/manifests/frame_manifest.csv` 基础上生成指定 `seq_len`/`pred_len` 的 sequence split CSV 和 `split_metadata.json`，服务于已完成 manifest 准备但需要补建 split 的本地数据准备流程。推荐长期入口仍是包内 MMW 数据准备能力或 `scripts/mmw/prepare_town10_skybridge.py`；该脚本是短期可审计的补充入口。输出仅允许写入 dataset 或显式本地数据根下的 `Prepared/<scene>/splits/<split_tag>/`，不得写入源码目录。删除/收敛条件是包内公开 split materialization utility 或 preprocessor CLI 覆盖同等参数、metadata 和错误提示后，将该脚本降级为 thin alias 或移除。
- `scripts/mmw/visualize_town_label_distribution.py` 属于 dataset_preparation。职责是读取本地 MMW Town split/manifest 数据并输出标签分布诊断图或摘要，辅助确认场景标签偏移；输出限定为显式本地诊断路径，不得提交生成图片或统计产物。
- `scripts/mmw/visualize_gps_angle_beam_correspondence.py` 属于 research_diagnostic。职责是读取本地 MMW Town split CSV 和 GPS anchor calibration summary，输出 BeamBench 风格的 GPS calibrated angle 与 mapping-centered beam index 空间对应图；输出限定为 `outputs/analysis/` 等显式本地诊断路径，不得提交生成图片或统计产物。
- `scripts/mmw/visualize_gps_prediction_trajectory.py` 属于 research_diagnostic。职责是读取本地 MMW Town split CSV 与 GPS anchor `predictions.csv`，输出真实 beam、GPS 预测 beam、DBA 在实际空间轨迹和样本序列上的对照图，辅助定位 DBA=0 的空间/序列偏移来源；输出限定为 `outputs/analysis/` 等显式本地诊断路径，不得提交生成图片或统计产物。
- `scripts/mmw/visualize_prediction_error_label_distribution.py` 属于 research_diagnostic。职责是读取本地预测 artifact 中的 `predictions.csv` 和 `summary.json`，输出预测错误样本的真实 beam label 分布图、源/目标场景标注和摘要；输出限定为 `outputs/analysis/` 等显式本地诊断路径，不得提交生成图片或统计产物。
- `scripts/run_mmw_sunny_modal15_l5p3_h123.sh` 和 `scripts/run_mmw_sunny_modal15_l5p6_h246.sh` 属于 shell_orchestration。职责是运行 sunny MMW 15 组 modal quick validation profile，分别固定对应 `seq_len`/`num_pred` 和 metric horizon 组合，并可选调用 split、radar map 和 cache 预热准备。默认输出根为 `outputs/mmw_sunny_modal15/<horizon_tag>/`；输出边界限定为 `outputs/`、`logs/`、dataset 准备产物和 cache/checkpoint 等本地运行产物，不得提交新生成结果。
- `scripts/run_mmw_gps_circular_soft_label_ablation.sh` 属于 shell_orchestration。职责是运行 sunny MMW GPS neural baseline 的 hard CE 与 circular Gaussian soft-label CE 对照实验，固定 MMW split、GPS-only 输入和 DBA 早停指标，用于诊断 beam codebook 边界/跳变对 GPS 监督的影响。输出边界限定为 `outputs/analysis/mmw_town_label_distribution/gps_circular_soft_label_ablation/`、`logs/mmw_gps_circular_soft_label_ablation/`、checkpoint 和本地训练缓存，不得提交新生成结果。
- `scripts/run_deepsense_gps_circular_soft_label.sh` 属于 shell_orchestration。职责是运行 DeepSense6G scene31-34 的 GPS-only circular Gaussian soft-label baseline，固定 DeepSense sequence CSV、GPS-only 输入和 DBA 早停指标，用于和 MMW Town GPS 监督诊断对照。输出边界限定为 `outputs/training/deepsense6g_gps_circular_soft_label/`、`logs/deepsense6g_gps_circular_soft_label/`、checkpoint 和本地训练缓存，不得提交新生成结果。
- `scripts/run_csi_hardening_matrix.sh` 属于 shell_orchestration。默认 CSI A0 配置为 `configs/csi/hardening_matrix/A0_clean_full_strong.yaml`，分析基线 run name 为 `csi_A0_clean_full_strong`；脚本不得重新引用已不存在的 `A0_clean_full_teacher.yaml` 作为默认入口。
已退役的 image-only legal crossroad probe、P3/V8 批处理和等待式 shell wrapper 已从 allowlist 删除；历史本地输出只通过 runtime cleanup manifest 作为候选审计，不再作为当前入口维护。

`tools/visualization/` viewer support 不得回流；`kd-sensing-export-viewer-manifest` 和 `python -m kd_sensing.cli.export_viewer_manifest` 已覆盖 manifest 导出 workflow。

## 本地产物

本地运行产物清理采用两阶段工作流：先运行 `kd-sensing-clean-runtime-artifacts` 生成 JSON manifest，再人工检查候选路径、规则、大小、mtime、风险等级和保护原因。真正删除必须复用 manifest 并显式传入 `--delete --manifest <path> --confirm-delete`；删除阶段会再次检查路径仍在扫描根内、未被 git 跟踪、未落入受保护根且状态没有相对 manifest 漂移。

当前 runtime output taxonomy 为：`outputs/cache/`、`outputs/cleanup_manifests/`、`outputs/analysis/`、`outputs/visual_analysis/`、`outputs/evaluations/`、`outputs/scene<id>/`、`outputs/scenegroup_<range-or-list>/` 和 `outputs/archive/`。新默认训练不得写入 `outputs/other/`、根级 `outputs/<run_name>/`、数字场景根 `outputs/31/` 或根级 `outputs/best_checkpoints/`；registry 默认位于当前 scene/scenegroup 下的 `best_checkpoints/`。

`outputs/mmw_sunny_modal15/<horizon_tag>/` 是 MMW modal15 shell orchestration 的历史语义化输出命名约定，保留为显式 workflow root，不作为通用训练默认根。历史 `outputs/other/`、根级 run、`outputs/eval_*`、数字场景根和根级 `outputs/best_checkpoints/` 不自动迁移、不自动删除；它们只通过 cleanup/organize manifest 作为人工确认候选出现，并保留 run index 的状态、checkpoint 数量和大小摘要。

本 change 不移动、删除、压缩或重写真实数据与本地实验产物。架构边界测试只检查已跟踪路径，继续拒绝：

- `__pycache__`、`.pyc`、`.pytest_cache`
- `outputs/`、`logs/`
- 除 `dataset/.gitkeep` 之外的 `dataset/` 内容
- 非 `All_models/` 历史资料范围内的 `.pth`、`.pt`、`.ckpt`

`dataset/.gitkeep` 是允许的源码占位文件。
