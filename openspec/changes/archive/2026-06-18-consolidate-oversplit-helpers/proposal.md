## Why

上一轮 hotspot right-size 拆分后，JEPA benchmark、data factory、sequence preprocessing 和 BeamBench baseline 出现了不少只承载少量内部 helper 的 Python 文件，阅读和跳转成本高于它们带来的边界收益。当前项目主要服务个人论文实验，不需要把每个内部不变量都包装成外部级防御检查；应回收过度拆分的小文件，并删除已由调用链保证的重复断言和二次安全检查。

## What Changes

- 合并同一 owner 下的内部 helper 小文件，优先处理当前工作树中新拆出的 `jepa_benchmark_*`、`data_factory_*`、`sequence_*` 和 `image_ae_gps_*` helper，而不是全仓库无差别合并。
- 保留公开 CLI、console script、包级 import facade 和本地产物边界；不得恢复旧入口、兼容聚合层或仓库根脚本。
- 在 JEPA GPS shortcut benchmark 中，将 common/types/io/scalar/metadata、Scenario D/CxD、runner summary/source/manifest 等过细 helper 回收到清晰 owner 模块，保持 manifest、metrics CSV、图表和 runner 输出 schema 不变。
- 在 data factory、sequence preprocessing 和 BeamBench Image AE+GPS 中，只合并单调用点或低复用 helper；保留会影响数据契约、配置解析或训练流程的稳定 owner 模块。
- 精简冗余防御式代码：删除内部重复 `assert`、重复 `isinstance`、重复 `None` 检查和只包一层再抛出的异常包装，让底层错误直接暴露。
- 保留必要边界检查：manifest/config/CLI 用户输入、文件路径存在性、split/label space/metric comparability、no-future-leak、产物写入边界和测试 fixture 契约仍要给出清晰错误。
- 更新 `docs/maintainer_context_index.yaml`、`docs/project_surface_inventory.md` 和架构边界测试中的 hotspot/line-budget/owner metadata，使治理事实与合并后的文件布局一致。
- 不改变训练指标、扰动语义、数据 split、checkpoint schema、真实输出目录或论文 claim 口径；真实实验产物继续只写入 ignored `outputs/`、`logs/` 或 manifest 指定目录。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `project-architecture`: 明确同 owner 内部 helper 可以在收益更高时合并，且内部冗余防御检查可删除；同时要求公开入口、轻量导入边界、产物边界和用户输入边界检查保持稳定。
- `jepa-gps-shortcut-benchmark`: 明确 benchmark 内部模块布局可以收敛为更少 owner 模块，只要 manifest 契约、扰动语义、comparability metadata、metrics/manifest 输出和 facade 入口行为不变。

## Impact

- 主要源码候选：
  - `src/kd_sensing/diagnostics/jepa_benchmark_common.py`
  - `src/kd_sensing/diagnostics/jepa_benchmark_scenario_d.py`
  - `src/kd_sensing/diagnostics/jepa_benchmark_runner.py`
  - `src/kd_sensing/diagnostics/jepa_benchmark_artifacts.py`
  - `src/kd_sensing/engine/data_factory.py`
  - `src/kd_sensing/preprocessing/sequences.py`
  - `src/kd_sensing/baselines/beambench/image_ae_gps.py`
- 可能删除的内部 helper 文件包括当前工作树中新拆出的 `jepa_benchmark_common_types.py`、`jepa_benchmark_io.py`、`jepa_benchmark_metadata.py`、`jepa_benchmark_scalars.py`、`jepa_benchmark_cxd_*.py`、`jepa_benchmark_runner_*.py`、`data_factory_validation.py`、`data_factory_scalers.py`、部分 `sequence_*.py` 和部分 `image_ae_gps_*.py`，最终以调用关系和测试结果决定。
- 受影响测试：`tests/test_jepa_gps_shortcut_benchmark.py`、`tests/test_architecture_boundaries.py`、data factory/sequence/BeamBench 相关 focused tests。
- 受影响文档治理：`docs/maintainer_context_index.yaml`、`docs/project_surface_inventory.md`，必要时补充 README 或导航文档中的 right-size 说明。
