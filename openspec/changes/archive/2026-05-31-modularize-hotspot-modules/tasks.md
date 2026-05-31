## 1. 基线梳理与行为锁定

- [x] 1.1 记录当前热点文件行数、公开函数/类、内部 helper 分布和现有测试覆盖，形成实施前基线。
- [x] 1.2 扫描 `hist_beam_loso_execution.py`、`data/mmw/preparation.py` 的外部 import，区分必须保留的公开入口和可迁移的 private helper。
- [x] 1.3 为 HiST-Beam LOSO executor 补齐或确认 focused characterization tests，覆盖 run metadata、execution progress JSONL、summary JSON/CSV、quick validation conclusion、preflight error 和 checkpoint reuse metadata。
- [x] 1.4 为 MMW preparation 补齐或确认 focused characterization tests，覆盖 `prepare_town10_skybridge`、frame manifest、sequence split CSV、split metadata、beam power artifact、availability/report JSON 和保留公开 helper import。

## 2. 拆分 HiST-Beam LOSO Executor

- [x] 2.1 新增或完善 `hist_beam_loso_records.py`，迁移 run id、run dir、base/missing run record、stage record 和 run identity helper。
- [x] 2.2 新增或完善 `hist_beam_loso_artifacts.py`，迁移 run metadata 写出、summary CSV/JSON 写出、JSONL progress event 和 JSON serialization helper。
- [x] 2.3 新增或完善 `hist_beam_loso_config.py`，迁移 scene cfg、stage cfg、enabled modalities、prototype/reuse/source cache key 和 throughput metadata 派生 helper。
- [x] 2.4 将仍留在 `hist_beam_loso_execution.py` 的 preflight、stage、summary/conclusion 和 matrix helper 分别迁入已有窄模块，避免重复实现。
- [x] 2.5 调整 `hist_beam_loso_execution.py` 为公开 orchestration facade，只保留公开入口、常量、兼容导出和顶层调用顺序。
- [x] 2.6 更新内部 import，使训练、评估、诊断和测试新增代码优先依赖 `hist_beam_loso_*` 窄模块，而不是从 executor facade 回流导入 helper。

## 3. 拆分 MMW Preparation

- [x] 3.1 新增 `preparation_config.py`，迁移 `MMWPreparationConfig`、配置加载、override normalization 和默认常量。
- [x] 3.2 新增 `preparation_audit.py`，迁移 zip/input 校验、extract marker、source hash、data availability 和 audit report helper。
- [x] 3.3 新增 `preparation_index.py`，迁移 `SensorFrame`、`PreparedFrame`、`ChannelFile`、sensor/channel indexing、scenario root 和 path parsing helper。
- [x] 3.4 新增 `preparation_splits.py`，迁移 sequence rows、group-safe split、guard band、split metadata 和 leakage diagnostics。
- [x] 3.5 新增 `preparation_beam_power.py`，迁移 channel payload 读取、DFT/codebook beam power 派生、power vector 写前校验和相关 field summary helper。
- [x] 3.6 新增 `preparation_writers.py`，迁移 manifest CSV、sequence row CSV、artifact path、report JSON 和 JSON-safe path 写出 helper。
- [x] 3.7 新增 `preparation_geometry.py`，迁移 relative geometry、proxy features、pose 解析、vehicle proxy 和 azimuth bin helper。
- [x] 3.8 调整 `data/mmw/preparation.py` 为公开 orchestration facade，保留 `prepare_town10_skybridge`、`build_sequence_splits_from_manifest` 等现有公开入口兼容。

## 4. Inventory 与架构防回流

- [x] 4.1 更新 `docs/project_surface_inventory.md`，记录第一批 hotspot facade 到窄模块映射、第二梯队热点清单、推荐 import 路径和禁止回流路径。
- [x] 4.2 更新 `tests/test_architecture_boundaries.py` 的 hotspot facade 断言，降低 `hist_beam_loso_execution.py` 的行数上限并加入 `data/mmw/preparation.py` 的行数上限、禁止片段和 helper 所属模块检查。
- [x] 4.3 增加内部 import 扫描断言，拒绝新增对第一批 facade 中已迁移 helper 的内部依赖，并在失败信息中指向推荐窄模块。
- [x] 4.4 确认第二梯队热点 `models/fusion/hist_beam.py`、`diagnostics/run_index.py`、`tools/visualization/gradio_multimodal_viewer.py`、`data/transform_ops/csi.py`、`engine/batch.py` 和 `engine/evaluation_pass.py` 已进入 inventory，并标明后续拆分方向或暂缓原因。

## 5. 第二梯队轻量优化

- [x] 5.1 评估 `models/fusion/hist_beam.py` 是否可先抽出 config/adapters/hierarchical helpers，若风险低则完成轻量拆分并补 focused tests。
- [x] 5.2 评估 `diagnostics/run_index.py` 是否可先抽出 proc/resource collection、artifact summary、render/csv writer helper，若风险低则完成轻量拆分并补 focused tests。
- [x] 5.3 评估 `tools/visualization/gradio_multimodal_viewer.py` 是否可先抽出 render cache/filter/status helper，若风险低则完成轻量拆分并补 viewer/import smoke。
- [x] 5.4 对未在本轮拆分的第二梯队热点只更新 inventory 和架构 guardrail，不做行为改写。

## 6. 验证与收尾

- [x] 6.1 运行 OpenSpec 校验：`openspec validate modularize-hotspot-modules --strict`。
- [x] 6.2 运行架构边界测试：`conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
- [x] 6.3 运行 HiST-Beam 和 MMW focused tests：`conda run -n kd_mm_beam pytest tests/test_hist_beam_loso.py tests/test_mmw_town10_preparation.py -q`。
- [x] 6.4 若触碰 CLI 或 viewer 入口，运行对应 smoke，例如 `conda run -n kd_mm_beam kd-sensing-export-viewer-manifest --help` 和 `conda run -n kd_mm_beam kd-sensing-visualize-modalities --help`。
- [x] 6.5 确认未新增、移动、删除或提交 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、下载压缩包或临时验证产物。
- [x] 6.6 运行最终回归：`conda run -n kd_mm_beam pytest -q`；若因环境或本地数据缺失无法完成，在最终说明中列出原因和已完成的 focused 验证。
