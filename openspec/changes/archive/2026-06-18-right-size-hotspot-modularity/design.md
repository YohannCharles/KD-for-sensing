## Context

项目已经有一套维护性热点治理：`docs/maintainer_context_index.yaml` 记录 budgets，`docs/project_surface_inventory.md` 解释热点原因，`tests/test_architecture_boundaries.py` 阻止 facade 回流和未登记长函数继续膨胀。此前的保守方案只调整治理语义；用户现在明确可以接受高风险，因此本设计改为完整源码表面修复。

当前主要热点：

- `src/kd_sensing/engine/data_factory.py` 约 989 行，混合 dataset 构建、loader kwargs、protocol split、stratified split、internal validation split、GPS scaler/normalizer 协调。
- `src/kd_sensing/preprocessing/sequences.py` 约 670 行，混合 column plan、window materialization、split selection、metadata writer 和 label distribution。
- `src/kd_sensing/baselines/beambench/image_ae_gps.py` 约 2438 行，承载 config、dataset、model、AE cache、training/evaluation、paper split 和 report writer。
- `DeepSense6GDataset`、`MMWDataset`、`trainer._train_inner`、`jepa_benchmark_common.py`、`jepa_benchmark_scenario_d.py`、`jepa_benchmark_runner.py` 仍是 P0/P1/P2 维护热点。
- `losses/jepa.py`、`losses/gps_lidar_bgam_losses.py` 和 `models/csi_encoder.py` 当前内聚且规模可控，作为“不要为了拆而拆”的样板保留。

## Goals / Non-Goals

**Goals:**

- 完整修复热点源码表面：拆分真实稳定职责，合并低价值边界，保留内聚小模块。
- 为每个热点指定 owner、目标模块边界、public import 策略、validation commands 和回滚方式。
- 将 `data_factory.py`、`sequences.py`、BeamBench Image AE+GPS、dataset/trainer、JEPA benchmark 作为分阶段 wave 处理。
- 更新治理索引和测试，让后续 Codex 能自动理解“这里该拆/该合/该保留”的判断。
- 保持公开 CLI、推荐入口和 runtime 行为稳定。

**Non-Goals:**

- 不改训练数学、loss 公式、数据 split 语义、beam label 口径、checkpoint schema、输出目录或本地产物清理策略。
- 不恢复退役路线、root-level wrapper、旧兼容入口、退役实体 YAML 或绕过 `src/kd_sensing` 的运行方式。
- 不把小而内聚的 loss/model 文件拆成无意义 wrapper。
- 不一次性把所有 wave 混在一个不可回滚的提交里实现。

## Decisions

### Decision 1: 采用 remediation waves，而不是单次大爆破

实施按 wave 推进，每个 wave 必须能独立验证：

1. **Wave 0: Baseline 与治理 schema**  
   记录当前热点行数、public imports、focused tests、已知红点；扩展索引 metadata 和架构测试。

2. **Wave 1: engine/preprocessing 公共构建面**  
   先处理 `data_factory.py` 和 `sequences.py`，因为它们是多个 workflow 的共享依赖，越早收敛越能减少后续重构冲突。

3. **Wave 2: BeamBench Image AE+GPS**  
   拆分 2400+ 行 baseline 文件，修复当前 paper split 预算红点，同时保持 BeamBench reproduction CLI/import 语义。

4. **Wave 3: Dataset 与 trainer orchestration**  
   继续拆 DeepSense6G/MMW dataset 和 `_train_inner` 的稳定 orchestration 边界。

5. **Wave 4: JEPA benchmark 第二层拆分**  
   拆 `common/scenario_d/runner` 中已经稳定的 schema、CxD、dominance、failure mode、summary 和 manifest writer。

6. **Wave 5: 合并/收敛与最终护栏**  
   合并低价值 helper/facade，更新 inventory，运行完整 focused + architecture 验证。

备选方案是一次性大改所有热点。它速度快，但回归定位困难；wave 方案更啰嗦，但能让每一步有证据。

### Decision 2: `data_factory.py` 保留公开 owner，私有 helper 迁入 sibling modules

`data_factory.py` 不应变成旧式兼容 facade，也不应继续承载所有细节。目标是保留公开函数：

- `build_dataset`
- `build_dataloaders`
- `build_split_dataset`
- `build_protocol_split_datasets`
- `build_dataloader`
- `build_dataloader_kwargs`
- `resolve_dataloader_split_config`
- `shutdown_dataloader_workers`
- `prepare_lidar_normalizer`

候选 sibling modules：

- `engine/data_factory_loaders.py`: dataloader kwargs、split config、worker shutdown。
- `engine/data_factory_protocols.py`: protocol role、scene union dataset、stratified 2604 split。
- `engine/data_factory_groups.py`: label/group stratification、sequence group keys、proportional counts。
- `engine/data_factory_validation.py`: validation-from-train split、subset annotation。
- `engine/data_factory_scalers.py`: GPS scaler fit/apply、multi-scene scaler harmonization、normalization kwargs。

`data_factory.py` 继续作为构建 owner，内部实现委托新模块；内部代码不得从旧 facade 或 deprecated builder 聚合层导入。

### Decision 3: `sequences.py` 拆成 preprocessing sequence 子职责

目标保留 `generate_sequence_data` 和 `SequencePreprocessor` 的公共语义，拆出稳定 helper：

- `preprocessing/sequence_columns.py`: `SequenceColumnPlan`、source column resolution、required column validation。
- `preprocessing/sequence_windows.py`: window materialization、window columns。
- `preprocessing/sequence_splits.py`: `SequenceSplit`、`SplitProtocolPlan`、balanced split selection、split scoring。
- `preprocessing/sequence_metadata.py`: metadata writer、label distribution summary、json-ready helper。

如果某 helper 只有一个调用点且没有领域语义，保留在 owner 内部，不为了行数拆出文件。

### Decision 4: BeamBench Image AE+GPS 拆成一个包内 mini-subsystem

目标 public owner 仍是 `kd_sensing.baselines.beambench.image_ae_gps` 或已登记 CLI owner；实现分散到明确职责模块。候选结构：

- `image_ae_gps_config.py`: config dataclass、normalization、device/performance metadata。
- `image_ae_gps_datasets.py`: direct dataset、image-only dataset、feature dataset、loader helpers。
- `image_ae_gps_models.py`: dense/fusion model。
- `image_ae_gps_ae.py`: AE train/load、latent encoding、feature cache signature/path。
- `image_ae_gps_training.py`: direct training loop、classifier epoch、optimizer/runtime helpers。
- `image_ae_gps_evaluation.py`: evaluation pass、prediction CSV/report rows。
- `image_ae_gps_paper_split.py`: scene-specific cfg、checkpoint reuse、paper split train/eval orchestration。
- `image_ae_gps_reports.py`: summary artifacts、markdown/csv/json writers。

`image_ae_gps.py` 可保留为 public re-export/orchestration owner，但必须纳入 facade/owner metadata，且不能重新吸收 suite-specific helper。

### Decision 5: Dataset 和 trainer 只拆稳定边界，不引入 mixin 迷宫

DeepSense6G/MMW dataset 的目标是减少类体积和初始化复杂度，但不引入多层 mixin 继承。优先使用普通 helper module 和小 dataclass：

- `deepsense6g_sample_assembly.py`
- `deepsense6g_resource_readers.py`
- `deepsense6g_scalers.py`
- `deepsense6g_target_provider.py`
- `mmw_columns.py`
- `mmw_geometry.py`
- `mmw_radio_semantic.py`

`trainer._train_inner` 拆成函数级 orchestration：

- runtime plan/builders
- epoch loop
- validation/checkpoint coordination
- final test evaluation
- artifact finalization

### Decision 6: JEPA benchmark 第二层按 schema/analysis 结果边界拆

JEPA benchmark 已有第一层模块化，下一步避免继续堆在 `common` 和 `scenario_d`：

- `jepa_benchmark_scalars.py`: numeric conversion、drop/slope/AUC。
- `jepa_benchmark_metadata.py`: batch metadata/sample id helpers。
- `jepa_benchmark_io.py`: json/csv/path/hash helpers。
- `jepa_benchmark_cxd_phase.py`: CxD grid、heatmap、phase artifacts。
- `jepa_benchmark_dominance.py`: modality dominance、crossing detection、pairing。
- `jepa_benchmark_failure_modes.py`: failure decomposition。
- `jepa_benchmark_runner_summary.py`: robustness summary、shortcut reliance、case studies。
- `jepa_benchmark_runner_manifest.py`: runner manifest builder。

### Decision 7: 明确保留小而内聚模块

`losses/jepa.py`、`losses/gps_lidar_bgam_losses.py` 和 `models/csi_encoder.py` 不进入默认拆分清单。修复策略是：

- 若发现重复通用 helper，合并到已有 shared loss/model helper。
- 若测试缺口存在，补 focused tests。
- 若未来增长超过阈值，再登记 hotspot，而不是提前拆。

## Risks / Trade-offs

- 高风险重构引入行为回归 -> 每个 wave 前先跑 baseline focused tests，wave 后跑同一组测试和架构边界测试。
- import 循环 -> 新模块只允许从低层 helper 向上被 owner 导入，不允许 helper 导入 public owner。
- public import 破坏 -> 对已登记公开符号添加 import smoke 或现有 CLI help smoke。
- 拆分过度 -> 每个新模块必须有明确 owner/responsibility；单调用点且无领域语义的 helper 不创建新文件。
- 合并导致 owner 再变大 -> 合并后若 owner 超出预算，必须登记为 `monitor` 或 `split-next`，不能静默扩大。
- 大量文件移动影响 review -> 每个 wave 独立提交/验证；纯移动和行为修改尽量分开。

## Migration Plan

1. **Baseline capture**: 记录当前 `wc -l`、hotspot metadata、public imports 和 focused tests 状态。
2. **Governance first**: 扩展索引字段和架构测试，让后续 wave 有稳定规则。
3. **Shared builders**: 重构 `data_factory.py` 和 `sequences.py`，先降低共享模块变更半径。
4. **Workflow split**: 拆 BeamBench Image AE+GPS，修复 paper split 红点。
5. **Runtime/data split**: 拆 dataset/trainer 稳定边界。
6. **Diagnostics split**: 拆 JEPA benchmark 第二层模块。
7. **Consolidation pass**: 合并低价值 helper/facade，更新 docs/inventory。
8. **Final verification**: 运行 OpenSpec、architecture boundaries、相关 focused tests；必要时跑全量 `conda run -n kd_mm_beam pytest -q`。

## Open Questions

- Wave 1 是否先拆 `data_factory.py` 还是 `sequences.py`：建议先 `data_factory.py`，因为当前 IDE active file 就在这里，且它影响训练/验证 runtime。
- BeamBench `image_ae_gps.py` 最终是否保留 public re-export：建议保留，以保护当前 CLI/import 语义，但架构测试必须阻止 helper 回流。
- 是否在本 change 内跑全量 pytest：建议最终尝试；若时间或环境限制，必须至少跑 wave 对应 focused tests 和 architecture boundaries。
