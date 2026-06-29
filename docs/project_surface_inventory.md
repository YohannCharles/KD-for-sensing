# 项目表面积 Inventory

本 inventory 记录 `refine-source-architecture-and-entry-surface` 的可审计基线。统计口径只覆盖源码、配置、文档和 OpenSpec artifact；`dataset/`、`outputs/`、`logs/`、cache、checkpoint、下载压缩包和其它本地运行产物不属于本 change 的处理范围。

`docs/maintainer_context_index.yaml` 已收缩为最小结构化事实清单，只保留难以从 pyproject、OpenSpec、真实路径或本 inventory 推导的退役 token 和验证命令。入口、root fusion config、模型注册例外、batch/runtime 分支、热点 rationale、remediation wave 和 lifecycle 解释以本 inventory、OpenSpec、pyproject 和 focused tests 为准；不再维护完整源码目录清单或大型 allowlist/budget YAML。

## 项目健康护栏基线

`prune-remaining-overengineered-surface` 收口时重新扫描：当前 `src/kd_sensing` 约有 229 个 Python 文件，`tests/` 约有 55 个 Python 文件，`scripts/` 有 11 个 Python 文件，`configs/` 有 141 个 YAML；仓库根目录有 9 个 Markdown，`docs/` 有 11 个 Markdown。本基线只读扫描源码、测试、配置和文档，不读取真实 `dataset/` 数据，不写入 `outputs/`、`logs/`、cache、checkpoint 或本地训练产物。

`right-size-project-architecture` 于 2026-06-19 记录新的 source architecture sizing baseline：CodeGraph 索引为 644 个文件、359 个 Python 文件、3,420 个 function 节点和 2,503 个 import 节点；AST 复核范围为 `src/`、`tests/` 和 `scripts/`，共 359 个 Python 文件、4,232 个 function 定义和 2,971 条 import 语句，其中 `src/` 278 个 Python 文件、3,249 个 function、2,195 条 import，`tests/` 58/749/584，`scripts/` 23/234/192。主要复杂度中心是 `src/kd_sensing/data`、`diagnostics`、`engine`、`models`、`baselines`、`config` 和 `cli`。这些数字只是架构审计和趋势定位信号，不能单独作为拆分或合并 KPI；真正的判定来自 owner 职责、公开 surface、导入边界、复用关系、热点预算和 focused validation。`dataset/`、`outputs/`、`outputs/cache/`、`logs/`、legacy `cache/`、`.pytest_cache/`、`__pycache__/`、`.pyc`、checkpoint 和权重文件不属于源码架构审计范围。

分层健康检查命令如下：

- OpenSpec：`openspec validate strengthen-project-health-guardrails --strict`
- 架构边界与健康护栏：`conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`
- CLI/config smoke：`conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_config_load_characterization.py -q`
- 触碰训练、数据集、诊断、CLI、配置解析或模型 forward 时，追加对应 focused tests，并在 tasks 或最终说明中记录未运行项及原因。

架构边界测试直接读取 pyproject、真实路径、OpenSpec、inventory 和少量测试常量；本节保留人类可读说明。

新增热点维护规则：

- 新增长函数、长类、manifest builder、orchestration workflow 或兼容 facade 时，先拆到窄模块；若暂缓拆分，必须在本 inventory 记录文件、符号、规模指标、拆分方向和暂缓原因。
- 兼容 facade 只保留公开 import/CLI 语义；内部实现不得从 facade 回流导入 helper，应直接依赖职责明确的窄模块。
- 普通 pytest 文件依赖 `tests/conftest.py` 或 editable install 导入 `src/`；只有架构边界 import probe、subprocess smoke 或显式隔离环境测试可以在子进程代码中局部设置 `sys.path`。

当前 AST 热点清单如下；本节说明预算理由、推荐拆分方向和暂缓原因。架构边界测试只检查关键 facade 回流和少量稳定边界，避免维护一份完整 hotspot budget 镜像。预算用于阻止热点静默扩大，不代表本 change 立即重构这些 runtime 逻辑：

热点右尺寸化规则：

- `facade-budget` / `hard-fail` 继续硬失败：公开 CLI/import facade 只能保留兼容入口和薄 orchestration，suite-specific helper 不得回流。
- `split-next`、`monitor` 和 `defer-with-rationale` 可配合 `headroom_lines` 表达有限业务 headroom；超出 headroom 时必须拆分、合并低价值边界或更新本 inventory 的理由化例外。
- `right-size-accepted` 表示当前尺寸和职责边界被接受，但必须保留 validation commands，不能解释为永久免检。
- `merge-candidate` 表示低价值边界等待合并，必须写明 owner、`consolidation_targets` 和验证命令；合并不能恢复旧入口或绕过 `src/kd_sensing`。
- 小而内聚的 loss/model/helper 可采用 `keep-and-test`：例如 `src/kd_sensing/losses/jepa.py` 和 `src/kd_sensing/models/csi_encoder.py` 当前更适合保留领域内聚并用 focused tests 护住，而不是为了降行数创建包装层。

判定矩阵：

| 证据 | 默认判断 | 下一步 |
| --- | --- | --- |
| 文件数、函数数或 import 数增长 | 只作为趋势信号 | 先确认是否来自 current capability、focused tests、薄 CLI、helper 合并或热点拆分 |
| 单个函数/类超预算且混合 loader、schema、writer、checkpoint 或 evaluation 职责 | `split-next` | 沿稳定职责边界抽 helper，并保留 public import/CLI 行为 |
| 同 owner、单调用点、只为 re-export 或降低行数存在的 helper | `merge-candidate` / `consolidate` | 合并回清晰 owner，不新增兼容 wrapper 或跨领域 `helpers.py` |
| 大 owner 负责审计型 schema、benchmark registry 或模型核心语义 | `right-size-accepted` / `keep-and-test` | 写明 accepted rationale、未来拆分触发条件和 focused tests |
| public facade 吸收 suite-specific implementation | `facade-budget` / `hard-fail` | 把实现移回窄模块，保留 facade 作为兼容入口 |

`right-size-project-architecture` remediation wave campaign：

| Wave | 目标 | 动作 | 验证边界 |
| --- | --- | --- | --- |
| Wave 0 | 治理 schema、索引、inventory、AI 导航和架构边界测试 | `hard-budget` 元数据扩展 | OpenSpec validate + architecture boundaries |
| Wave 1 | BeamBench Image AE+GPS | 拆 paper split report/checkpoint/cache/dataset orchestration，public owner 继续薄 | BeamBench focused test + architecture boundaries |
| Wave 2 | DeepSense6G/MMW dataset 与 trainer | 按 resource/scaler/target/epoch/checkpoint/finalization 稳定边界拆分 | dataset modality tests、training IO + architecture boundaries |
| Wave 3 | evaluation pass 与 diagnostics 二级热点 | evaluation schema-safe helper，MMW GPS v2/Jepa visual/run index/cleanup 先登记或小步拆分 | evaluation pass、modality difficulty、JEPA benchmark + architecture boundaries |
| Wave 4 | JEPA benchmark accepted owners | 保留 accepted rationale，active predictive 语义稳定前不强拆 | JEPA benchmark focused test + architecture boundaries |
| Wave 5 | 合并与 import 面收口 | 合并 confirmed merge-candidate，内聚小模块 `keep-and-test` | CLI/config smoke + architecture boundaries |

| 文件 | 符号 | 类型 | 当前规模 | 推荐拆分方向 | 暂缓原因 / 优先级 |
| --- | --- | --- | --- | --- | --- |
| `src/kd_sensing/data/datasets/deepsense6g.py` | `DeepSense6GDataset` | 超长 dataset 类 | 1054 行 | 已拆出 `deepsense6g_contract.py`、`deepsense6g_gps_contract.py`、`deepsense6g_columns.py`、`deepsense6g_cache_paths.py`、`deepsense6g_sample_assembly.py` 和 `deepsense6g_scalers.py`；后续继续拆 label/history adapters 与 resource reader glue | 数据契约 helper 已覆盖 GPS feature mode/scene calibration、`beam_target_source=current`、`gps_bev_xy_source`、required columns、metadata parsing、cache path、beam/auxiliary target assembly 和 streaming stats；真实资源读取仍和多条 current workflow 耦合，继续预算监控；优先级 P1 |
| `src/kd_sensing/data/datasets/deepsense6g.py` | `DeepSense6GDataset.__init__` | 超长初始化函数 | 263 行 | 已拆出 CSV path、feature mode setup、cache path、target source 和 column guard；后续拆 resource reader setup、scaler/normalizer setup 和 target provider setup | 初始化仍承担 loader/scaler orchestration，保持 265 行预算并要求新契约规则进入 helper；优先级 P1 |
| `src/kd_sensing/data/datasets/mmw.py` | `MMWDataset` | 超长 dataset 类 | 592 行 | 已拆出 `mmw_columns.py`、`mmw_geometry.py` 和 `mmw_radio_semantic.py`；后续继续拆 manifest parsing、sequence window 和 sensor feature loading | MMW group-safe split 与 label calibration 仍在演进；优先级 P1 |
| `src/kd_sensing/engine/trainer.py` | `_train_inner` | 超长训练 orchestration 函数 | 316 行 | 已拆出 `trainer_runtime_helpers.py` 的 final evaluation、CSI RMS config handoff、epoch setter 和 shutdown helper；后续继续拆 epoch loop 与 checkpoint coordination | 训练数值语义敏感；优先级 P1 |
| `src/kd_sensing/engine/mmw_town_gps_v2.py` | `run_mmw_town_gps_v2` | 超长诊断 workflow 函数 | 277 行 | protocol dispatch、label-space resolution、summary writing、plot handoff | MMW GPS v2 仍承担对照解释；优先级 P2 |
| `src/kd_sensing/baselines/beambench/image_ae_gps_training.py` | `run_image_ae_gps_training` | BeamBench Image AE+GPS workflow | 244 行 | 已拆出 config/dataset/model/AE cache/evaluation/report helper；后续只在训练流程变更时继续收口 | 复现实验入口保持 Table III 语义，package CLI 直接导入该 owner，不再维护 `image_ae_gps.py` 大聚合；优先级 P1 |
| `src/kd_sensing/baselines/beambench/image_ae_gps_paper_split.py` | `run_image_ae_gps_paper_split_training` | BeamBench Image AE+GPS paper split workflow | 267 行 | scene split orchestration、checkpoint reuse、per-scene summary 和 summary artifact 已迁出 public owner；后续可继续拆 report payload builder | 与本地 scene31-34 复现产物耦合，当前在 headroom 内并由 BeamBench focused test 覆盖；优先级 P1 |
| `src/kd_sensing/engine/evaluation_pass.py` | `run_evaluation_pass` | evaluation pass | 216 行 | metric aggregation、objective outputs、prediction metadata | evaluation schema 为多个 CLI 共享；优先级 P2 |
| `src/kd_sensing/engine/batch.py` | module helpers | batch preparation 热点 | 文件级 batch contract | modality target preparation、label adapters、history anchor inputs | 触碰会影响训练/验证公共 batch contract；优先级 P2 |
| `src/kd_sensing/diagnostics/jepa_gps_shortcut_benchmark.py` | module runner facade | JEPA shortcut benchmark 公开 import/CLI facade | 文件级 399 行，预算 450 行 | 只保留兼容 re-export；suite-specific helper 实现不得回流 | 内部 owner 已收敛为 `jepa_benchmark_common.py`、`jepa_benchmark_manifest.py`、`jepa_benchmark_scenario_c.py`、`jepa_benchmark_scenario_d.py`、`jepa_benchmark_predictive.py`、`jepa_benchmark_perturbations.py`、`jepa_benchmark_artifacts.py`、`jepa_benchmark_plots.py` 和 `jepa_benchmark_runner.py`；优先级 P0 |
| `src/kd_sensing/diagnostics/run_index.py` | module helpers | 诊断 run index 热点 | 文件级诊断 contract | process/resource collection、artifact summary、CSV/render writers | 输出 schema 已在 README 暴露；优先级 P3 |

JEPA benchmark owner 模块完整路径：`src/kd_sensing/diagnostics/jepa_benchmark_common.py`、`src/kd_sensing/diagnostics/jepa_benchmark_manifest.py`、`src/kd_sensing/diagnostics/jepa_benchmark_scenario_c.py`、`src/kd_sensing/diagnostics/jepa_benchmark_scenario_d.py`、`src/kd_sensing/diagnostics/jepa_benchmark_predictive.py`、`src/kd_sensing/diagnostics/jepa_benchmark_perturbations.py`、`src/kd_sensing/diagnostics/jepa_benchmark_artifacts.py`、`src/kd_sensing/diagnostics/jepa_benchmark_plots.py` 和 `src/kd_sensing/diagnostics/jepa_benchmark_runner.py`。`jepa_benchmark_common.py` 收纳 common types、JSON/CSV/path、scalar 和 metadata helper；`jepa_benchmark_scenario_d.py` 收纳 Scenario D/CxD normalization、metrics、phase、dominance 和 failure-mode helper；`jepa_benchmark_runner.py` 收纳 `run_jepa_gps_shortcut_benchmark` orchestration、runner summary、metric source ingestion、real-forward shard/cache、geometry diagnostics dispatch 和 runner manifest helper。

`jepa_benchmark_predictive.py` 因 active Predictive GPS-query++ change 同时维护 predictive stress curves、legacy P0-P5 compatibility、GPS-query advantage slice、claim gate 和 diagnostics，当前登记为 `right-size-accepted`，待 predictive 语义稳定后再考虑拆分。`jepa_benchmark_perturbations.py` 因 hard-negative 和 advantage perturbation 参数扩展调整监控预算，仍保留 difficulty profile、deterministic perturbation 和 legacy perturbation helper 的未来 split target。

## 配置生命周期分类

`configs/` 当前 141 个 YAML 按生命周期维护：

- canonical/current root configs：`configs/<image|radar|gps|lidar|mmwave|csi>/{strong,lightweight,supervised}.yaml`、`configs/deepsense6g_gps_adapter_v2.yaml`、`configs/mmw_town_gps_adapter_v2.yaml`。
- canonical fusion root：`configs/fusion/*.yaml` 只保留长期 supervised、token-transformer/objective-aware 和 BeamBench Image AE+GPS thin/reproducibility 入口；具体 allowlist 见下文 `configs/fusion/` 根目录分类。
- experiment reproduction：`configs/fusion/experiments/jepa_image_gps/*.yaml` 当前 30 个实体 YAML，记录 JEPA image+GPS、GPS-biased checkpoint reuse、GPS query pooling、geometry prior/safe rerank、architecture sweep、2604/BeamBench fair 和必要 supervised/random-best 对照；这些不是 root canonical 入口。
- CSI experiment matrix：`configs/csi/hardening_matrix/_base/*.yaml` 和 `configs/fusion/csi_hardening_matrix/_base/*.yaml` 是 base config；`configs/csi/hardening_matrix/*.yaml` 与 `configs/fusion/csi_hardening_matrix/*.yaml` 是 A/B/C/D/E 实验 ID 的轻量 overlay YAML；`configs/csi/hardening_matrix/debug/*.yaml` 是 debug/smoke。
- dataset preparation：`configs/preprocess/*.yaml` 服务当前数据准备、索引和 cache；DeepVerse/DT31 与 Raymobtime s008 预处理配置已删除。
- diagnostics：`configs/diagnostics/jepa_visual_analysis_2604.yaml` 服务 JEPA 离线分析；`configs/diagnostics/jepa_gps_shortcut_benchmark_*.yaml` 服务 JEPA vs GPS shortcut benchmark、Scenario D image observability 和 Predictive Robustness smoke/canonical manifest。它们不是训练入口，predictive smoke manifest 只提供 mock/smoke schema evidence；旧 `configs/diagnostics/modality_visualization.yaml` 和 viewer manifest 导出配置已退役。
- baseline/reproduction：`configs/baselines/*.yaml`、`configs/fusion/tii_vlrg_transformer_baseline.yaml`、`configs/fusion/amber_lite_missing_modality.yaml`、`configs/fusion/experiments/wcl2025_missing_modality/*.yaml` 和 `configs/pretraining/*.yaml` 只保留仍维护的 BeamBench reproduction、TII-VLRG-style / AMBER-lite / RMBP-MM local experimental baselines、可选 TII external workflow、可选 WCL source audit、GPS-conditioned JEPA pretraining 或未来明确 current workflow；AMR-Net_gps_image 和 JEPA-MSAC 实体配置已退役删除。GPS window baseline 配置已删除。
- retired history：已删除的 KD、HiST/Hist、Top8 selector、GPS residual、camera residual、BGAM、viewer manifest、CRAF/MARF/G2D/Multimodal-NF 实体配置只允许作为历史或 migration guard 说明出现，不得作为当前推荐入口。

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
| `cxd-phase-transition-analysis` | `current` | 当前 CxD phase diagram、dominance、crossing 和 failure decomposition 诊断契约。 |
| `csi-channel-data` | `current` | 当前 CSI 数据列和样本字段契约。 |
| `csi-channel-degradation` | `current` | 当前 CSI degradation profile 能力。 |
| `csi-hardening-debug-validation` | `current` | 当前 CSI hardening debug 矩阵验证。 |
| `csi-hardening-experiment-matrix` | `current` | 当前 CSI hardening 控制变量矩阵。 |
| `csi-modality-model` | `current` | 当前 CSI 模态模型契约。 |
| `dataset-directory-layout` | `current` | 当前数据集目录和本地产物边界。 |
| `dataset-runtime-contracts` | `current` | 当前 dataset descriptor/sample/adapter/target provider 契约。 |
| `deepsense6g-camera-ae-residual-correction` | `retired-tombstone` | DeepSense6G camera residual 路线已退役。 |
| `deepsense6g-gps-lidar-bgam-reranker` | `retired-tombstone` | DeepSense6G GPS+LiDAR BGAM reranker workflow 已退役；只保留防回流边界。 |
| `deepsense6g-gps-residual-fusion` | `retired-tombstone` | DeepSense6G GPS residual fusion 路线已退役。 |
| `deepsense6g-gps-top8-candidate-selector` | `retired-tombstone` | DeepSense6G Top8 selector 与 BGAM-only TopK manifest/dataset/loss 支撑已退役。 |
| `deepsense6g-scene-selection` | `current` | 当前 DeepSense6G scene 选择与输出隔离。 |
| `distillation-free-project-surface` | `current` | 当前去蒸馏化项目表面和旧 KD 拒绝契约。 |
| `experiment-artifact-registry` | `current` | 当前 checkpoint/artifact registry 与历史产物隔离。 |
| `experiment-run-index` | `current` | 当前只读 run index 能力。 |
| `experiment-workflow` | `current` | 当前配置驱动训练/评估/预处理/诊断 workflow。 |
| `first-class-prediction-tasks` | `current` | 当前 prediction objective 元数据、loss 和指标契约。 |
| `geometry-residual-beam-labels` | `retired-tombstone` | geometry residual label 路线已退役。 |
| `geometry-prior-beam-fusion` | `current` | 当前 GPS geometry prior/logit fusion、clean gate、teacher stabilization 与 diagnostics bundle 契约。 |
| `ieee-11282996-gps-image-reproduction` | `retired-tombstone` | AMR-Net_gps_image source-audit/local substitute workflow 已退役；只保留 metadata conflict tombstone 和防回流边界。 |
| `gps-coarse-anchor-prediction` | `retired-tombstone` | GPS coarse anchor prediction 已退役。 |
| `gps-conditioned-jepa-pretraining` | `current` | 当前 GPS-conditioned JEPA 预训练能力。 |
| `gps-modality-model` | `current` | 当前 GPS 模态模型契约。 |
| `gps-preprocessing` | `current` | 当前 GPS sequence/relative-polar/scaler 预处理契约。 |
| `gps-pseudo-label-bgam` | `retired-tombstone` | GPS pseudo-history BGAM 输入与评估产物已退役。 |
| `gps-query-jepa-pooling` | `current` | 当前 Image+GPS JEPA query-pool 下游能力。 |
| `hist-beam-cross-scene-adaptation` | `retired-tombstone` | HiST/Hist 跨场景适配研究线已退役。 |
| `history-anchored-residual-beam` | `retired-tombstone` | history-anchor Hist 路径已退役。 |
| `image-only-legal-crossroad-probe` | `retired-tombstone` | image-only Hist probe 已退役。 |
| `image-preprocessing-profiles` | `current` | 当前 image profile 与 RGB/ImageNet 预处理契约。 |
| `jepa-downstream-extensibility` | `current` | 当前 JEPA downstream pooler/adapter 扩展契约。 |
| `jepa-gps-shortcut-benchmark` | `current` | 当前 JEPA vs GPS shortcut benchmark 能力。 |
| `jepa-msac-reproduction` | `retired-tombstone` | JEPA-MSAC Scenario 32 mock/paper workflow 已退役；只保留 tombstone 和旧入口拒绝边界。 |
| `jepa-visual-analysis-suite` | `current` | 当前 JEPA visual analysis 离线诊断能力。 |
| `jepa-visual-architecture-sweep` | `current` | 当前 GPS-query JEPA visual architecture sweep 候选矩阵、可比性 metadata 和结果摘要契约。 |
| `legacy-kd-isolation` | `retired-tombstone` | legacy KD 入口只作为拒绝、历史读取和 migration guard 墓碑。 |
| `lidar-modality-model` | `current` | 当前 LiDAR 模态模型契约。 |
| `lidar-preprocessing` | `current` | 当前 LiDAR 点云/BEV cache/scaler 预处理契约。 |
| `maintainer-context-index` | `supporting` | 最小结构化事实清单；不再作为任务路由、源码目录镜像、entrypoint allowlist 或 hotspot budget 权威。 |
| `mainline-experiment-documentation` | `current` | 当前主线模型目录、实验协议表和结果/claim 账本治理能力；不替代 OpenSpec 行为契约。 |
| `mmw-beam-label-calibration` | `current` | 当前 MMW beam label calibration 契约。 |
| `mmw-cross-scene-adaptation-protocol` | `supporting` | MMW split/adaptation protocol 支撑；MMW HiST wording 只可作历史边界。 |
| `mmw-sensor-assisted-beam-prediction` | `current` | 当前 MMW sensor-assisted beam prediction 边界。 |
| `mmw-town-gps-adapter-v2` | `current` | 当前 MMW Town GPS-only v2 workflow。 |
| `mmw-town-gps-lidar-bgam-reranker` | `retired-tombstone` | MMW Town GPS+LiDAR BGAM reranker workflow 已退役。 |
| `mmw-town-gps-top8-candidate-selector` | `retired-tombstone` | MMW standalone Top8 manifest 与 BGAM candidate 支撑已退役。 |
| `mmw-town10-dataset-preparation` | `current` | 当前 MMW Town10 数据准备能力。 |
| `mmwave-modality-model` | `current` | 当前 mmWave 模态模型契约。 |
| `mmwave-preprocessing` | `current` | 当前 mmWave sequence/power/scaler 预处理契约。 |
| `model-architecture-extension-contract` | `current` | 当前新增 baseline、模块化组件、整模型例外、workflow reproduction 和 metadata 护栏契约。 |
| `model-architecture-summary` | `current` | 当前模型架构/参数摘要观测能力；包内入口只读配置、sweep manifest 或 startup summary，不是运行时配置或第二套 registry。 |
| `modality-aware-data-loading` | `current` | 当前按模态选择加载、normalization 和 cache 行为契约。 |
| `modality-contracts` | `current` | 当前中心化模态顺序、batch key 和 profile 拒绝边界。 |
| `modality-difficulty-pipeline` | `current` | 当前 difficulty profile/operator pipeline。 |
| `modality-visual-diagnostics` | `current` | 当前诊断入口收敛到 JEPA visual analysis、GPS shortcut benchmark 和其它非 viewer 诊断；viewer manifest/Gradio viewer 已退役。 |
| `modular-sequence-model` | `current` | 当前模块化序列模型结构。 |
| `multi-task-occlusion-position-learning` | `current` | 当前 occlusion/position 辅助监督能力。 |
| `observability-aware-fusion` | `current` | 当前 image/GPS reliability metadata 自适应融合、uncertainty gating 和 JEPA fallback 契约。 |
| `original-code-compatibility` | `supporting` | 历史 checkpoint/config 读取兼容支撑，不是旧训练入口。 |
| `path-prototype-hist-beam-adaptation` | `retired-tombstone` | P3/HiST path prototype 已退役。 |
| `project-architecture` | `current` | 当前包结构、入口、轻量导入和退役边界。 |
| `project-health-guardrails` | `current` | 当前健康护栏、inventory 和静态检查契约。 |
| `project-surface-cleanup` | `current` | 当前源码表面清理、退役路线和本地产物边界。 |
| `predictive-jepa-robustness` | `current` | 当前 JEPA Predictive Robustness workflow 契约；pending/unverified，不代表真实 stress-curve 数值 claim 已完成。 |
| `spec-lifecycle-boundaries` | `current` | 当前 OpenSpec capability lifecycle 分类和读取边界。 |
| `radar-student-model` | `current` | 当前 radar student 模型契约。 |
| `radar-teacher-model` | `current` | 当前 radar teacher/复现兼容模型契约。 |
| `radio-semantic-hist-beam-adaptation` | `retired-tombstone` | radio-semantic Hist 路线已退役。 |
| `raymobtime-s008-retirement` | `retired-tombstone` | Raymobtime s008 退役墓碑。 |
| `real-perturbation-forward-evaluation` | `current` | 当前 benchmark real-forward perturbation evaluation、logits cache、resume/shard 和 leakage guard 契约。 |
| `resnet18-image-encoder` | `current` | 当前 ResNet-18 ImageNet encoder 能力。 |
| `runtime-artifact-cleanup` | `current` | 当前只读 cleanup manifest 和显式删除 workflow。 |
| `safe-residual-beam-rerank-fusion` | `current` | 当前 opt-in anchor-safe residual beam reranker、candidate set、fallback gate 和 rerank loss 契约。 |
| `scenario-d-image-observability-benchmark` | `current` | 当前 Scenario D 图像可观测性等级、CxD benchmark 矩阵和输出产物边界。 |
| `snapshot-next-frame-baselines` | `current` | 当前 snapshot next-frame baseline 契约。 |
| `soft-beam-label-training` | `current` | 当前 circular/soft beam label supervised training。 |
| `target-shot-domain-splitting` | `current` | 当前 source-target domain 和 target-shot split 契约。 |
| `tinyvit-image-encoder` | `current` | 当前 TinyViT-5M/11M opt-in RGB/ImageNet image encoder 组件能力；不替换默认 ResNet-18。 |
| `training-throughput-optimization` | `current` | 当前训练吞吐 profiling 与建议。 |
| `vision-position-baseline-suite` | `current` | 当前 Vision-Position baseline preset 矩阵。 |

退役/支撑例外摘要：

- 本次 `prune-overengineered-project-surface` 审计保留现有 `retired-tombstone` specs：它们仍承担旧 CLI、配置路径、registry 名称、文档 wording 或 migration guard 的防回流价值。只剩背景说明且无 guard 价值的墓碑后续可归档或折叠到本节集中清单，但不得恢复旧入口。
- HiST/Hist、Raymobtime s008、GPS coarse anchor、GPS residual、camera residual、geometry residual、image-only Hist probe、P3/Radio-semantic Hist、legacy KD、CRAF/MARF/G2D 和 Multimodal-NF 不属于当前推荐入口；只能在退役墓碑、migration guard、历史说明或拒绝边界中出现。
- Top8/TopK 的旧 standalone selector 训练、plot、compare CLI/config 已退役；BGAM-only candidate manifest、dataset 和 loss 支撑也已删除。普通 Top-K 指标、circular metrics、MMW GPS v2 和 CSI candidate ranking 不因字符串命中而退役。
- 通用 LOSO/few-shot split、beamspace physical label、distribution diagnostics、runtime cleanup 和历史 checkpoint 读取属于 supporting 或 current guardrail；它们不得恢复 Hist、Raymobtime、旧 KD 或 residual 研究线的旧 CLI、root config、console script 或实体 YAML。

模型扩展路径分类：

- config-only baseline：只改 YAML、virtual recipe、overlay 或 hyperparameter；普通 supervised/adaptation baseline 的首选路径。
- component baseline：新增或替换 `ENCODERS`、`PROJECTORS`、`REPRESENTATION_CORES` 或 `HEADS` 子组件，并通过 `model.primary.type: modular_sequence` 选择。
- whole-model exception：新增完整 `@MODELS.register(...)` 时必须有 current spec、active change artifact、inventory 或测试 allowlist 中的明确例外说明，并覆盖 registry build、forward/output adaptation 和 `training_strategy_metadata()`；JEPA-MSAC 的 `jepa_msac` 例外已退役，不再作为 current registry surface。
- `pinn_multimodal_beam` 是 `add-physics-informed-mmw-beam-baseline` 登记的 current whole-model exception：owner 为 `src/kd_sensing/models/pinn_multimodal_beam.py`，物理 helper 位于 `src/kd_sensing/models/physics/`，监督 adapter 位于 `src/kd_sensing/data/datasets/mmw_physics_adapter.py`，训练接入为 opt-in `loss.physics.enabled` extension，inspection 入口为 `kd-sensing-inspect-mmw-physics`。输出边界仍是 ignored `outputs/`；验证命令为 `conda run -n kd_mm_beam pytest tests/test_physics_informed_mmw.py -q`、配置/CLI/架构边界 focused tests 和相关 OpenSpec validate；paper-style frontend 复用 JEPA image tokenizer、sparse pilot CSI 和 shared Transformer，声明边界限定为窄带阵列信道。
- workflow/paper reproduction：BeamBench AE+GPS、官方协议包装、多阶段训练或特殊报告产物应位于 `src/kd_sensing/baselines/<family>/`、package console script 或包内 CLI；它们不是普通模块化 baseline，不得复制通用训练循环或恢复旧入口。BGAM 属于 retired-tombstone，不再作为 workflow/paper reproduction 入口。

`retire-legacy-model-registry-surface` 将普通 strong/lightweight whole-model 注册名和 feature-extractor `MODELS` 注册名移出 current discovery。`configs/<radar|gps|mmwave>/{strong,lightweight,supervised}.yaml`、`configs/gps/ablation_relative_polar.yaml` 和 `configs/fusion/radar_gps_supervised.yaml` 保留文件名/run name，但 `model.primary.type` 均为 `modular_sequence`；差异由 encoder/core/head 参数表达。仍有当前迁移价值的旧名可保留 removed guard；完全退役旧名可回落为普通 unknown-name 错误。暂缓保留的 whole-model exceptions 是 `cls_token_transformer_fusion`、`token_transformer_fusion`、`vision_position_late_fusion`、`vision_position_transformer_fusion` 和 `gps_sequence_baseline`，因为它们仍有 current spec、实体 config、vision-position workflow 或 focused tests 覆盖。

模型架构摘要是只读观测层，owner helper 位于 `src/kd_sensing/models/architecture_summary.py`，包内入口为 `python -m kd_sensing.cli.model_architecture_summary`，人类可读组件目录维护在 `docs/model_architecture_inventory.md`。它复用已解析配置、真实 `nn.Module`、startup summary artifact 或 JEPA visual sweep manifest 生成 JSON/Markdown/CSV 参数摘要；它不新增 registry、不改变 `model.primary` 构建语义、不读取真实 `dataset/`，也不写训练产物。显式 `--output` 应指向 ignored `outputs/analysis/model_architecture_summary/` 或用户指定路径。

## 文档生命周期分类

根目录 Markdown：

- current quickstart and short index：`README.md`；只保留安装、主要入口、quickstart、数据/产物边界和文档索引，不复制完整模型目录、协议表或结果账本。
- agent/developer operating rules：`AGENTS.md`。
- environment/data setup：`ENVIRONMENT.md`、`DATASET_STRUCTURE.md`。
- current reproducibility/reporting：`README_REPRODUCE.md` 提供 BeamBench/Arnold22 当前推荐命令并指向 current summary；`BASELINE_REPORT.md` 开头维护 current summary、claim status 和 caveat；`results/reproduce_baseline.md` 是历史流水账，不能覆盖 current summary。
- historical research notes：`TODO_FOR_ATTENTION_MODULE.md`、`deep-research-report.md`、`PATCH_NOTES.md`、`知乎问答下载.md`、`跨场景自适应方案.md`、`跨场景自适应方案_融合推理修改版.md`。这些文档只能作为历史背景，不得把退役 KD/HiST/Top8/residual/camera residual 路线重新描述为当前推荐入口。

`docs/` Markdown：

- current agent/maintainer navigation：`docs/agent_navigation.md`；它只提供修改前权威来源、当前状态、任务路由、误读边界和验证选择导航，不替代 README、AGENTS 或 OpenSpec specs，也不维护完整项目表面积审计。
- current architecture/health inventory：`docs/project_surface_inventory.md`；记录 capability lifecycle、文档生命周期和源码/配置/入口表面积，不承担实验参数横向比较。
- current model/workflow facts：`docs/model_architecture_inventory.md` 维护当前 registry-backed model、encoder、projector、representation core 和 head 目录；`docs/mainline_model_catalog.md` 维护当前主线模型、baseline/control、诊断和 benchmark 目录；`docs/experiment_protocols.md` 维护 formal/lowmem/smoke/debug/upper-bound/historical ablation 参数口径；`docs/result_claims_registry.md` 维护可引用结果、blocked official、本地 substitute、upper-bound、mock/smoke 和 historical ablation 的 claim provenance。
- current workflow quickstart：`docs/experiment_matrix.md`、`docs/extension_guide.md`、`docs/training_throughput.md`；`docs/experiment_matrix.md` 只保留推荐顺序、入口命令和关键 caveat，并指向三份 current mainline 文档。
- dataset/diagnostic focused notes：`docs/research_notes.md`。
- historical analysis：`docs/p3_v7_multisource_crossroad_analysis.md`，只保留研究背景，不作为当前长期入口。

## 源码热点模块

本批次拆分的热点 facade 与职责模块如下：

- 模型扩展热点优先落在 `src/kd_sensing/models/modular.py` 的模块化子组件、`src/kd_sensing/models/image_encoders.py` / `jepa.py` 等已有 encoder 或窄模型文件，以及 `src/kd_sensing/registries.py` 的轻量 registry。普通 baseline 不应新增 dataset 解析、训练循环、validation loop、专用 `prepare_*` 或 `forward_task_model` 分支。
- `src/kd_sensing/models/architecture_summary.py` 保留模型架构摘要 helper；它只读 `nn.Module`、配置和声明 metadata，供 startup summary、JEPA sweep summary 和包内 CLI 复用。新增参数口径或 warning 时优先扩展该 helper 与 focused tests，不把观察逻辑塞入 registry 或训练循环。
- BeamBench Image AE+GPS 不再维护 `image_ae_gps.py` 大聚合 owner；package CLI 直接导入 `image_ae_gps_training.py` 和 `image_ae_gps_paper_split.py`，脚本/测试直接导入 `image_ae_gps_config.py`、`image_ae_gps_datasets.py`、`image_ae_gps_models.py`、`image_ae_gps_ae.py`、`image_ae_gps_evaluation.py` 和 `image_ae_gps_reports.py` 等具体 owner。`src/kd_sensing/baselines/beambench/__init__.py` 只保留轻量 package marker，不再 re-export heavy training/dataset/model symbols。
- `ObservabilityAwareFusion` / reliability-aware fusion 长期视为显式 opt-in adaptive fusion helper 或可组合 representation core 候选；普通 early-concat、CLS-token transformer、JEPA 和 Vision-Position baseline 不应被静默替换语义。消费 reliability metadata 的模型必须在 run metadata 或 `training_strategy_metadata()` 中标记。

- Raymobtime s008 dataset、预处理器、selection 模型、配置和 focused test 已退役删除；旧 registry 名称和配置路径只保留 migration guard 错误信息。
- `src/kd_sensing/models/csi.py` 已删除；CSI pilot estimation、hardening、view tokenizer/fusion、debug helpers 和 encoder registry glue 分别由 `csi_estimation.py`、`csi_hardening.py`、`csi_views.py`、`csi_debug.py`、`csi_encoder.py` 直接承载。
- `src/kd_sensing/engine/objective_metadata.py`、`src/kd_sensing/engine/objectives/registry.py` 和 `src/kd_sensing/engine/objectives/history.py` 已删除；objective 名称、默认 metric、metric alias、mode、history fields、TensorBoard scalar schema、runtime metadata 和 validation helper 都由 `src/kd_sensing/engine/objectives/metadata.py` 维护。内部代码和测试不得再从旧 facade 或 `kd_sensing.engine.objectives` package re-export 导入。
- `src/kd_sensing/data/datasets/deepsense6g.py` 保留 DeepSense6G runtime dataset orchestration；GPS contract、target source、metadata parsing、beam label cache mode、required columns 和 image/LiDAR cache path resolution 分别迁移到 `deepsense6g_gps_contract.py`、`deepsense6g_contract.py`、`deepsense6g_columns.py` 和 `deepsense6g_cache_paths.py`。后续新增 GPS feature mode、beam target source、column guard 或 cache path rule 必须优先进入这些 helper，并使用 synthetic tests 避免读取真实 `dataset/`。
- viewer manifest 导出、`viewer_manifest_*` helper、viewer prediction export 和 `kd-sensing-visualize-modalities` alias 已退役并删除；它们只作为 retired-tombstone 防回流说明保留，不再是公开 orchestration/import 入口。
- `src/kd_sensing/diagnostics/jepa_gps_shortcut_benchmark.py` 保留 JEPA vs GPS shortcut benchmark 公开 facade 和兼容导出；内部职责收敛到 `src/kd_sensing/diagnostics/jepa_benchmark_common.py`、`jepa_benchmark_manifest.py`、`jepa_benchmark_scenario_c.py`、`jepa_benchmark_scenario_d.py`、`jepa_benchmark_predictive.py`、`jepa_benchmark_perturbations.py`、`jepa_benchmark_artifacts.py`、`jepa_benchmark_plots.py` 和 `jepa_benchmark_runner.py`。新增 Scenario C/D/CxD、Predictive、manifest/schema、artifact writer、plotting 或 runner helper 必须落在这些 owner 模块，不得把 suite-specific implementation 写回 facade；默认输出限定在 `outputs/analysis/`，不读取真实 `dataset/`，除非用户显式用真实 config/weights 执行评估计划。
- DeepVerse/DT31 generator、label builder、split、sanity check 和 focused test 已退役删除；MMW beam power 所需的通用 ULA/DFT codebook helper 保留在 `src/kd_sensing/data/beam_codebook.py`。
- `src/kd_sensing/data/mmw/preparation.py` 保留 Town10/Skybridge MMW preparation 公开 orchestration 和兼容导出；配置 schema、默认常量和 override loading 迁移到 `src/kd_sensing/data/mmw/preparation_config.py`，zip/input audit、extract marker、availability report 迁移到 `src/kd_sensing/data/mmw/preparation_audit.py`，sensor/channel indexing 与 path parsing 迁移到 `src/kd_sensing/data/mmw/preparation_index.py`，sequence row、group-safe split、guard band 和 leakage diagnostics 迁移到 `src/kd_sensing/data/mmw/preparation_splits.py`，channel payload、DFT/codebook beam power 和 power validation 迁移到 `src/kd_sensing/data/mmw/preparation_beam_power.py`，manifest/split/report 写出迁移到 `src/kd_sensing/data/mmw/preparation_writers.py`，relative geometry、pose/proxy features 和 azimuth bin helper 迁移到 `src/kd_sensing/data/mmw/preparation_geometry.py`。

新增内部代码不得从 `kd_sensing.engine.objective_metadata`、`kd_sensing.data`、`kd_sensing.data.datasets`、`kd_sensing.models.fusion`、`kd_sensing.data.mmw.preparation` 或 `kd_sensing.diagnostics.jepa_gps_shortcut_benchmark` 回流导入窄 helper；应直接使用上面的窄模块。`kd_sensing.data.mmw.preparation` 和 `kd_sensing.diagnostics.jepa_gps_shortcut_benchmark` 可作为公开 orchestration/import 入口，但内部 helper 引用应分别指向 `preparation_*` 与 `jepa_benchmark_*` 窄模块。viewer manifest 相关模块不得作为兼容 facade 回流。

## 第二梯队热点

第二梯队热点先纳入 inventory 和架构 review 清单，不在本批次做大规模行为改写：

- HiST-Beam engine/model/evaluation 专用源码已退役并从当前支持面删除；旧 registry 名称和配置路径只保留 migration guard 错误信息。
- `src/kd_sensing/diagnostics/jepa_visual_analysis.py`：登记为 `monitor`；职责包含 report payload、table writer、figure writer、cache metadata 和 analysis manifest。若后续拆分，优先抽内部 report/table/figure/cache owner，并运行 viewer/visual diagnostics 与架构边界测试，保持 CLI 输出 schema 兼容。
- `src/kd_sensing/diagnostics/run_index.py`：后续优先抽出 process/resource collection、artifact summary、CSV/render writers；当前保持诊断输出 schema 兼容。
- `src/kd_sensing/diagnostics/runtime_artifact_cleanup.py`：登记为 `monitor`；manifest/apply/render/organize 边界已列入索引。任何拆分都必须保持显式删除确认、dry-run manifest 和 ignored runtime artifact roots 不变。
- `src/kd_sensing/models/modular.py`：登记为 `keep-and-test`；模块化模型构建、representation core dispatch、forward preparation 和 metadata handoff 当前比机械拆文件更易审计，下一次模型语义变更时再考虑抽窄 helper。
- `src/kd_sensing/config/canonical.py`：登记为 `monitor`；canonical recipe、virtual config overlay 和 path alias/retired-route guard 是稳定拆分候选，但不因行数单独重构。
- `src/kd_sensing/data/difficulty/operators/image.py`：登记为 `monitor`；image difficulty operator、weather/occlusion/geometry-aware transform 和 deterministic seed helper 需要 `tests/test_modality_difficulty.py` 护住后再拆。
- `src/kd_sensing/data/transform_ops/csi.py`：后续优先抽出 CSI parsing、hardening feature transforms 和 temporal window helpers；当前避免同时改动数据契约。
- `src/kd_sensing/engine/batch.py`：后续优先抽出 modality target preparation、label adapters 和 history anchor input helper；当前保持训练 batch contract。
- `src/kd_sensing/engine/evaluation_pass.py`：后续优先抽出 metrics aggregation、objective-specific outputs 和 prediction metadata helper；当前保持 evaluation result schema。
- `src/kd_sensing/engine/loso_data.py`：当前保留 LOSO source/target dataloader helper。结构检查未发现当前内部调用，但该模块通过 `__all__` 暴露 `build_loso_dataloaders`、`build_loso_source_train_loader` 和 `build_loso_target_stage_loader`，且当前 OpenSpec 仍保留 cross-scene LOSO/few-shot sampling 语义；若后续完全退役，应另起 change 同步 specs、docs 和外部兼容说明。

本次已删除高置信孤立源码：`src/kd_sensing/evaluation/flops.py`、`src/kd_sensing/evaluation/latency.py` 和 `src/kd_sensing/data/transform_ops/cache.py`。删除前检查确认它们不属于 console script、package `__init__` 公开导出、注册入口、README/docs/OpenSpec 当前声明或测试依赖；image/LiDAR cache 的当前实现入口继续分别位于 `src/kd_sensing/data/transform_ops/image_cache.py` 和 `src/kd_sensing/data/transform_ops/lidar.py`。

`simplify-overengineered-surfaces` 收敛分类：

- deleted: `src/kd_sensing/diagnostics/communication_state_features.py`、`tests/test_communication_state_features.py` 和 Python thin alias 脚本 `scripts/train.py`、`scripts/evaluate.py`、`scripts/preprocess.py`、`scripts/check_dataset.py`、`scripts/eval_baseline.py`、`scripts/train_baseline.py`、`scripts/train_beambench_image_ae_gps.py`、`scripts/run_beambench_image_ae_gps_tableiii.py`。
- no-current-surface: `src/kd_sensing/models/lidar_pillar_encoder.py` 已删除；当前 LiDAR 支持面仍是点云读取、BEV 伪图像/cache、normalization、质量摘要和 dataset flat sample。
- remove-internal-only: `src/kd_sensing/data/dataset_runtime.py` 已删除；target-shot split 直接消费 `Mapping[str, Any]` rows，并继续解析 JSON metadata/resource_refs/target_ref 字符串。当前 dataset runtime contract 由实际 dataset、data factory、metadata helper 和 batch runtime 满足。
- `prune-remaining-overengineered-surface` deleted: 退役整模型类和旧 alias（GPS/image/radar/LiDAR/mmWave whole-model class）、`src/kd_sensing/models/fusion/networks.py`、`src/kd_sensing/engine/objectives/{registry.py,history.py}`、`src/kd_sensing/config/canonical_recipes/`、`src/kd_sensing/cli/beambench_check_dataset.py`、旧手写 YAML fallback、runtime cleanup 的 HiST/P3/V8/V9 目录命名规则，以及前序已删的 `src/kd_sensing/_typing.py`、`src/kd_sensing/engine/objective_metadata.py`、`src/kd_sensing/data/dataset_runtime.py`、`src/kd_sensing/baselines/beambench/image_ae_gps.py`、`scripts/analyze_csi_hardening_sweep.py` 和 `tests/test_csi_hardening_sweep_analysis.py`。
- `prune-remaining-overengineered-surface` merged: canonical recipe 常量表迁入 `kd_sensing.config.canonical`，objective metadata/history/registry 常量迁入 `kd_sensing.engine.objectives.metadata`，dataset profile/sample/fusion key 信息复用 `kd_sensing.modalities`，TinyViT 四个注册名改为 preset 表循环，run-index 默认扫描只覆盖 current canonical layout，CLI help smoke 优先跑已安装脚本、缺失时按 `pyproject.toml` target 调用源码 owner。
- `prune-audit-followup-overengineering` deleted: `src/kd_sensing/config/source.py` 的单用途 config source 包装已合并回 `src/kd_sensing/config/io.py`；`src/kd_sensing/data/transform_ops/normalization.py` 的 normalization re-export 已删除，调用方直接导入 `gps.py`、`lidar.py` 和 `mmwave.py` owner。
- `prune-audit-followup-overengineering` retained-with-reason: registry removed-name guard 只剩仍有 current migration value 的 scene alias、fusion old class/whole-model alias 和 KD loss token；训练 extension 框架保留为 `right-size-accepted`，因为它连接 JEPA base loss/target encoder update、teacher guidance checkpoint loads、batch-step after-forward hook 和 epoch metadata，删除会触碰训练语义；`jepa_visual_analysis.py`、`gps_query_evidence.py`、MMW GPS v2 CLI/engine 等诊断 helper 语义和输出字段不完全一致，暂不合并为跨领域 `utils`。
- retained-with-reason: `src/kd_sensing/data/dataset_descriptors.py` 仍保留轻量 `DatasetDescriptor` 查询，因为 `profile_for()`、`to_dict()`、family/storage/split/artifact boundary metadata 和 validation 错误信息提供实际行为；profile 名称、sample key、fusion input key、shape/metadata 不再在 descriptor 中重复维护，而是来自 `modalities.py`。
- merged: duplicate `OutputRegistry` 合并到 `src/kd_sensing/diagnostics/jepa_benchmark_artifacts.py` owner，`src/kd_sensing/diagnostics/jepa_visual_analysis.py` 复用该 helper；没有新增通用 registry 抽象。
- dependency-audit: dev extra 删除 `thop` 与 `pytorch-model-summary`；默认 runtime 删除 `scikit-image`，图像读取改用 Pillow；`h5py` 改为 optional `hdf5` extra，仅真实 HDF5 path semantics 读取时需要。若后续需要 FLOPs、model summary 或 HDF5 默认读取，需在对应 change 中说明当前使用点和验证命令。

## 配置 YAML

当前 `configs/fusion/` 根目录有 12 个实体 YAML，只保留长期 canonical 或当前明确入口；架构边界测试直接扫描这些真实文件。`configs/fusion/experiments/jepa_image_gps/` 有 30 个 JEPA image+GPS 实验特化 YAML，分类为：canonical/current root 保留 0 个，recipe 可无损生成 0 个，recipe 有差异 3 个 architecture sweep（`architecture_sweep_{smoke,lowmem,strict}.yaml`，由 manifest 消费但仍有显式差异），人工样例/正式实验 21 个（2604、BeamBench-fair、geometry prior 和 safe residual rerank 系列），debug/smoke 2 个（`image_gps_jepa_k_token_pooler_smoke.yaml`、`safe_residual_rerank_clean_smoke_2604_s32_s34_lowmem.yaml`），diagnostics manifest 0 个，删除/归档 0 个。本轮未删除这些 YAML：没有现成 generator 能无损生成完整 experiment name、objective、dataset split、modalities、model/loss/training/output/checkpoint 语义。

CSI hardening 当前是 base+overlay：`configs/csi/hardening_matrix/_base/csi_only.yaml` 支撑 13 个 CSI-only overlay ID，`configs/csi/hardening_matrix/debug/*.yaml` 保留 5 个 debug/smoke parity 配置，`configs/fusion/csi_hardening_matrix/_base/{gps_only,gps_csi}.yaml` 支撑 E0-E3 GPS+CSI validation overlay；`tests/test_config_load_characterization.py` 验证 A0/E1 的关键解析语义与 base 等价并保留 `config_resolution`。BEV-Fusion 2604 分类为 formal `paper_full.yaml`、lowmem approximation、smoke、ablation matrix 和 9 个人工 ablation overlay；pretraining smoke 分类为 GPS-conditioned JEPA current/pending 与 JEPA visual architecture sweep smoke；diagnostics YAML 分类为 hand-maintained manifest 或 strict diagnostic config；difficulty YAML 分类为 current reliability profile。删除/归档 0 个。

`configs/diagnostics/jepa_gps_shortcut_benchmark_smoke.yaml` 是不读真实数据的 benchmark smoke manifest，`configs/diagnostics/jepa_gps_shortcut_benchmark_beambench_fair.yaml` 是引用现有 Vision-Position baseline 与 JEPA downstream 配置的 canonical benchmark manifest，checkpoint 路径为本地占位；Scenario D 和 Predictive Robustness smoke manifest 只验证 schema、strict comparability 字段和 claim gating，不产生真实数值 claim。AMR-Net_gps_image 与 JEPA-MSAC 的实体 YAML 已退役删除；旧路径由 migration guard 拒绝。

`configs/fusion/` 根目录保留分类如下：

- canonical strong/current supervised: `all_modalities_lidar_supervised.yaml`、`all_modalities_supervised.yaml`、`image_gps_supervised.yaml`、`image_gps_resnet18_modular_supervised.yaml`、`mmwave_csi_supervised.yaml`、`mmwave_csi_medium_degraded_supervised.yaml`、`radar_gps_supervised.yaml`、`radar_lidar_supervised.yaml`。
- current thin/reproducibility entry: `beambench_image_ae_gps_direct.yaml`。
- current token-transformer/objective-aware entries: `token_transformer_all_modalities_supervised.yaml`、`token_transformer_all_modalities_multitask_supervised.yaml`、`token_transformer_image_radar_supervised.yaml`。

已迁移到 `configs/fusion/experiments/jepa_image_gps/` 的实验特化配置如下：

- fair/2604 当前文档复核配置：`image_gps_jepa_gps_biased_best_beambench_fair_lowmem.yaml`、`image_gps_jepa_gps_biased_best_2604_s32_s34_lowmem.yaml`、`image_gps_jepa_gps_query_pool_best_beambench_fair_lowmem.yaml`、`image_gps_jepa_gps_query_pool_best_2604_s32_s34_lowmem.yaml` 和 `image_gps_jepa_gps_query_pool_best_2604_s32_s34_fasttrain.yaml` 是主线或快速复核主线；`image_gps_supervised_beambench_fair_lowmem.yaml`、`image_gps_jepa_random_best_beambench_fair_lowmem.yaml`、`image_gps_supervised_2604_s32_s34_lowmem.yaml`、`image_gps_jepa_random_best_2604_s32_s34_lowmem.yaml` 是对照。保留 `beambench_fair` 文件名的配置现在对齐 BeamBench Table III 的输入/split/target/metric 口径：`seq_len=1`、`num_pred=1`、`beam_target_source=current`、GPS `paper_distance_angle`、scene paper calibration angle、S32-S34 train、S31-S34 test 和 linear DBA；它们仍是 Image+GPS/JEPA 下游模型，不是 Table III Camera AE+GPS Direct 模型。
- Predictive Robustness pending 配置：`image_gps_jepa_predictive_hybrid_beambench_fair_lowmem.yaml` 是 BeamBench-fair 派生训练 profile，默认训练 legacy `P4_joint_predictive_recovery`；单个 P4 train/curriculum profile 不等价于完整 stress-curve benchmark，真实 claim 只能来自本地 real manifest 的 strict comparable clean + `image_missing` / `image_noise` / `gps_noise` train-then-evaluate。
- BeamBench fair 保留复查配置：`image_gps_jepa_random_last_beambench_fair_lowmem.yaml` 和 `image_gps_jepa_gps_biased_pooler_param_groups_beambench_fair_lowmem.yaml`。
- 已退役删除配置：scene31-only low-memory/best-last 配置、非 BeamBench last-checkpoint 配置，以及 `jepa_gru.yaml`、`jepa_snapshot.yaml`、`jepa_plain_token_transformer.yaml`、`jepa_next_query_transformer.yaml` next-beam downstream ablation 配置。

已退役的 CRAF、MARF、G2D、Multimodal-NF 和 KD 实体 YAML、overlay recipe 与 virtual alias 不再作为支持入口存在。删除实体文件后，配置加载器只为当前 strong/lightweight canonical、snapshot、objective-aware、Vision-Position baseline preset 和保留 overlay 生成 virtual config，不接管退役路径；旧 `logits_kd` / `rkd` 路径只作为 migration guard 的拒绝命中保留。Vision-Position 当前 virtual preset 为 `configs/fusion/{camera_ae_gps,resnet_gps,transformer_image_gps,gps_only_neural}.yaml`，默认使用 BeamBench-style `seq_len=1`、`num_pred=1`、`paper_distance_angle`、`beam_target_source=current` 和 linear DBA 口径；这些 preset 只是项目对照，不得作为 Arnold22 Table III row 的数值复现入口。`gps_only_neural` 不是论文 GPS `Classical*` 或 `Dense†` 行；Table III Camera AE+GPS row 只能走 `configs/fusion/beambench_image_ae_gps_direct.yaml` 和 `kd-sensing-run-beambench-image-ae-gps-tableiii`。

## 脚本入口 Allowlist

保留入口按 lifecycle 分类如下；`pyproject.toml` 是 package console script 权威，`tests/test_architecture_boundaries.py` 直接读取 pyproject 和真实脚本路径。新增 `scripts/` 或 `tools/analysis/` 下的 Python/shell 文件必须在本 inventory、README/docs 或 OpenSpec tasks 中保留职责、输出边界和 caveat 说明。

职责边界：package CLI 只做 argparse、配置/override glue、轻量 IO、调用 owner module 和 user-facing exit code；真实训练、评估、benchmark、dataset preparation 或诊断主逻辑必须位于 `baselines/`、`diagnostics/`、`engine/`、`data/` 或对应窄模块。Python thin alias 已删除；`scripts/` 只保留 research diagnostic、dataset preparation 和 shell orchestration，不再作为训练/评估/预处理兼容入口。所有入口默认输出只能落在 ignored 的 `outputs/`、`logs/`、cache/checkpoint、本地 dataset preparation target、`docs/figures/` 或显式用户路径中，不得把新生成产物写回源码目录。

- package_cli: 核心 `kd-sensing-train`、`kd-sensing-evaluate`、`kd-sensing-preprocess`、run-index、runtime cleanup、target-shot split、distribution-shift、MMW GPS v2、JEPA visual analysis、GPS shortcut benchmark、BeamBench Image AE+GPS training、Table III orchestration、可选 TII VLRG Transformer external workflow 和可选 WCL 2025 missing-modality source-audit 都只承担 parser/config glue 或明确诊断 CLI，owner module 位于 engine、data、diagnostics 或 baselines。`kd_sensing.cli.run_jepa_msac` 和 `kd_sensing.cli.run_amr_net_gps_image` 已退役删除；JEPA visual analysis 与 GPS shortcut benchmark 是只读模型/benchmark 产物的诊断入口，TII/WCL wrappers 写 manifest、dry-run/execute 命令 logs 或 condition summary row，输出限定在 ignored 的 `outputs/visual_analysis/` 或 `outputs/analysis/`。TII/WCL/AMBER 的本地实验 baseline 主入口是 `kd-sensing-train --config <local baseline yaml>`。GPS coarse anchor、Top8 selector、DeepSense6G residual、camera residual、BGAM、viewer manifest、Gradio viewer、AMR-Net_gps_image 和 JEPA-MSAC 入口已退役，不再作为当前 package CLI。
- research_diagnostic: `scripts/analysis/beambench_ae_gps_diagnostics.py`、`scripts/analysis/deepsense_gps_v2_support_sweep_artifacts.py`、`scripts/analysis/visualize_deepsense_beambench_correspondence.py`、`scripts/debug_eval_consistency.py`、`scripts/figures/draw_jepa_architecture.py`、`scripts/profile_training_io.py`、`scripts/recommend_parallel_training.py`。`scripts/analyze_csi_hardening_sweep.py` 已退役删除，CSI hardening 解释阈值保留在 `docs/research_notes.md`。旧模态子集/扰动研究脚本和 MMW GPS v2 旁支 `scripts/mmw/visualize_gps_*` 脚本不再作为长期入口；通用 subset/mask 验证保留在 `kd-sensing-evaluate` 使用的共享 evaluation pass 与配置化 `evaluation.modality_subsets` 中，MMW current 图表使用 `kd-sensing-plot-mmw-town-gps-v2`。
- dataset_preparation: `scripts/inspect_dataset.py`、`scripts/mmw/prepare_town10_skybridge.py`、`scripts/mmw/build_sequence_splits_from_manifest.py`、`scripts/mmw/visualize_town_label_distribution.py`。
- shell_orchestration: `scripts/run_csi_hardening_matrix.sh`。DeepSense GPS circular soft-label、MMW GPS circular soft-label ablation 和 MMW sunny modal15 shell wrappers 已退役，历史本地输出只通过 runtime cleanup/organize manifest 审计。
- deleted thin aliases: `scripts/train.py`、`scripts/evaluate.py`、`scripts/preprocess.py`、`scripts/check_dataset.py`、`scripts/eval_baseline.py`、`scripts/train_baseline.py`、`scripts/train_beambench_image_ae_gps.py`、`scripts/run_beambench_image_ae_gps_tableiii.py` 和包内旧 wrapper `kd_sensing.cli.beambench_check_dataset` 已删除。训练、评估、预处理使用 `kd-sensing-train`、`kd-sensing-evaluate`、`kd-sensing-preprocess`；BeamBench baseline 审计/mock smoke 如需直接运行，使用包内 CLI module `python -m kd_sensing.cli.beambench_train_baseline`、`python -m kd_sensing.cli.beambench_eval_baseline` 或 owner module `python -m kd_sensing.baselines.beambench.dataset_check`。

MMW 入口生命周期说明：

- `kd-sensing-train-beambench-image-ae-gps` 属于 package_cli。职责是委托 `kd_sensing.baselines.beambench.image_ae_gps_training` 中的论文 row 专用实现：从本地 DeepSense6G scene31-34 sequence CSV 读取 camera/GPS/current beam target，先训练或加载 Camera AE，再冻结 AE encoder，使用官方 BeamBench `dense_model` 等价 head（Camera AE latent + GPS Direct、Sigmoid+BCE）训练 fusion classifier，输出 checkpoint、history、predictions 和 BeamBench DBA/top-k metrics；输出限定在 `outputs/scene<id>/` 或显式用户路径下，不得提交新 checkpoint、日志或 predictions。
- `kd-sensing-run-beambench-image-ae-gps-tableiii` 属于 package_cli。职责是委托 `kd_sensing.cli.run_beambench_image_ae_gps_tableiii`，顺序运行 scene31-34 的 Camera AE + GPS Direct 本地复现实验并输出 Table III 风格 CSV/Markdown/JSON 汇总；默认输出限定在 `outputs/scenegroup_s32_s34/`，评估-only 汇总可写入 `outputs/evaluations/`，不得提交新 checkpoint、feature cache、predictions 或 summary runtime artifact。
- `kd-sensing-tii-vlrg-transformer` 属于 package_cli。职责是委托 `kd_sensing.baselines.tii_vlrg_transformer`，记录 TII VLRG Transformer external repo、source commit、checkpoint、prediction/metrics artifact、dry-run/execute 外部命令、logs 和统一 DBA summary row；默认输出限定在 `outputs/analysis/tii_vlrg_transformer_reproduction/`，不得提交外部源码副本、checkpoint、cache、prediction、metrics 或日志。
- `kd-sensing-wcl2025-missing-modality-audit` 属于 package_cli。职责是委托 `kd_sensing.baselines.rmbp_mm.workflow`，记录 IEEE WCL 2025 source audit、official/local-substitute branch、claim status、strict comparability gate 和 condition-level summary adapter；默认输出限定在 `outputs/analysis/wcl2025_missing_modality_reproduction/`，不得提交外部源码副本、checkpoint、cache、prediction、metrics 或日志。
- `scripts/analysis/beambench_ae_gps_diagnostics.py` 属于 research_diagnostic。职责是读取本地 BeamBench AE+GPS 复现实验产物，汇总训练历史、预测和指标诊断，辅助分析 Camera AE + GPS Direct row 的本地复现差异；输出限定为 `outputs/analysis/` 等显式诊断路径，不得提交生成统计、图表或 checkpoint。
- `scripts/analysis/visualize_deepsense_beambench_correspondence.py` 属于 research_diagnostic。职责是读取本地 DeepSense6G scene31-34 原始 scenario CSV、GPS 和 beam labels，输出 BeamBench Fig.2 风格的 calibrated GPS angle 与 centered beam index 空间对应图；输出限定为 `outputs/analysis/` 等显式本地诊断路径，不得提交生成图片或统计产物。
- `scripts/analysis/deepsense_gps_v2_support_sweep_artifacts.py` 属于 research_diagnostic。职责是汇总 DeepSense6G GPS v2 support sweep 本地 artifact，输出只限 `outputs/analysis/` 等显式诊断路径，不得提交生成统计或图表产物。
- `scripts/figures/draw_jepa_architecture.py` 属于 research_diagnostic。职责是生成 JEPA pretraining/downstream reuse 架构示意图，输出限定为 `docs/figures/` 或显式本地图形路径；不得提交由临时运行产生的缓存、checkpoint 或训练产物。
- `scripts/mmw/build_sequence_splits_from_manifest.py` 属于 dataset_preparation。职责是在已有 `Prepared/<scene>/manifests/frame_manifest.csv` 基础上生成指定 `seq_len`/`pred_len` 的 sequence split CSV 和 `split_metadata.json`，服务于已完成 manifest 准备但需要补建 split 的本地数据准备流程。推荐长期入口仍是包内 MMW 数据准备能力或 `scripts/mmw/prepare_town10_skybridge.py`；该脚本是短期可审计的补充入口。输出仅允许写入 dataset 或显式本地数据根下的 `Prepared/<scene>/splits/<split_tag>/`，不得写入源码目录。删除/收敛条件是包内公开 split materialization utility 或 preprocessor CLI 覆盖同等参数、metadata 和错误提示后，将该脚本降级为 thin alias 或移除。
- `scripts/mmw/visualize_town_label_distribution.py` 属于 dataset_preparation。职责是读取本地 MMW Town split/manifest 数据并输出标签分布诊断图或摘要，辅助确认场景标签偏移；输出限定为显式本地诊断路径，不得提交生成图片或统计产物。
- MMW GPS v2 旁支 `scripts/mmw/visualize_gps_angle_beam_correspondence.py`、`scripts/mmw/visualize_gps_prediction_trajectory.py` 和 `scripts/mmw/visualize_prediction_error_label_distribution.py` 已退役删除。它们只作为 historical exploratory plots 说明保留；当前图表和不可用说明由 `kd-sensing-plot-mmw-town-gps-v2` / `kd-sensing-compare-mmw-town-gps-v2` 负责。
- `scripts/run_mmw_sunny_modal15_l5p3_h123.sh`、`scripts/run_mmw_sunny_modal15_l5p6_h246.sh`、`scripts/run_mmw_gps_circular_soft_label_ablation.sh` 和 `scripts/run_deepsense_gps_circular_soft_label.sh` 已退役删除。需要复跑相关实验时使用 current `kd-sensing-train`、MMW GPS v2 package CLI、保留的 diagnostics 或 CSI hardening runner，而不是恢复 shell wrapper。
- `scripts/run_csi_hardening_matrix.sh` 属于 shell_orchestration。默认 CSI A0 配置为 `configs/csi/hardening_matrix/A0_clean_full_strong.yaml`，分析基线 run name 为 `csi_A0_clean_full_strong`；脚本不再调用已退役的 CSI sweep analyzer，也不得重新引用已不存在的 `A0_clean_full_teacher.yaml` 作为默认入口。
已退役的 image-only legal crossroad probe、P3/V8 批处理和等待式 shell wrapper 已从 allowlist 删除；历史本地输出只通过 runtime cleanup manifest 作为候选审计，不再作为当前入口维护。

`tools/visualization/` viewer support、`kd-sensing-export-viewer-manifest`、`kd-sensing-visualize-modalities` 和 `python -m kd_sensing.cli.export_viewer_manifest` 不得回流；当前诊断入口使用 JEPA visual analysis、GPS shortcut benchmark 和其它明确 current 的非 viewer 诊断。

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
