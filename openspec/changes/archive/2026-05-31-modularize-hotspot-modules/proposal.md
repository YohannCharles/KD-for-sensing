## Why

当前源码中仍存在多个高变更频率的大文件，最大热点包括 `src/kd_sensing/engine/hist_beam_loso_execution.py`（约 2500 行）和 `src/kd_sensing/data/mmw/preparation.py`（约 2168 行）。这些文件把 orchestration、preflight、split、artifact 写出、summary/conclusion、IO 解析和领域算法混在一起，导致后续改动容易跨职责扩散，也不利于架构边界测试发现回流。

## What Changes

- 拆分 HiST-Beam LOSO executor 热点文件，将 preflight、stage 执行、run record/progress、summary/conclusion、matrix metadata 和配置派生移动到职责明确的窄模块；保留现有公开入口作为薄编排 facade。
- 拆分 MMW Town10 preparation 热点文件，将配置 schema、zip/input audit、sensor/channel indexing、sequence split、beam power 派生、manifest/CSV 写出、report 和 proxy geometry helper 分离。
- 梳理第二梯队热点模块，包括 `models/fusion/hist_beam.py`、`diagnostics/run_index.py`、`tools/visualization/gradio_multimodal_viewer.py`、`data/transform_ops/csi.py` 和训练/evaluation batch helper，按风险和变更频率纳入分阶段拆分清单。
- 补充架构回流防护：新增或更新 inventory/allowlist，标记兼容 facade、推荐窄模块和禁止内部新增依赖的路径。
- 保持公开 CLI、公开 import、manifest schema、run metadata、训练/评估输出和数据目录策略兼容；本变更不移动、不删除、不压缩本地数据或训练产物。
- 不新增旧入口、二级兼容聚合层或绕过 `src/kd_sensing` 包结构的运行方式。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `project-architecture`: 细化源码热点模块拆分、兼容 facade、内部依赖防回流和分层验证要求。

## Impact

- 主要影响 `src/kd_sensing/engine/hist_beam_loso_execution.py`、`src/kd_sensing/data/mmw/preparation.py` 及其新拆出的同包窄模块。
- 次级影响 `src/kd_sensing/models/fusion/hist_beam.py`、`src/kd_sensing/diagnostics/run_index.py`、`tools/visualization/gradio_multimodal_viewer.py`、`src/kd_sensing/data/transform_ops/csi.py`、`src/kd_sensing/engine/batch.py`、`src/kd_sensing/engine/evaluation_pass.py` 等热点模块的后续拆分计划。
- 测试影响包括架构边界测试、HiST-Beam LOSO focused tests、MMW Town10 preparation tests、viewer/import smoke 和 OpenSpec strict validation。
- 不应改变模型数值语义、数据样本契约、现有配置加载行为、CLI 参数语义、默认数据目录或本地产物边界。
