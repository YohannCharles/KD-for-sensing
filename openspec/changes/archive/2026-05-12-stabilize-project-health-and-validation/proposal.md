## Why

当前项目已经具备完整的训练、诊断和可视化链路，但项目体检暴露出一个测试红点和几类架构债：Phase 1.5 总决策会在 checkpoint matrix 未完成时过早标记为 final，包级导入会牵出重依赖，builder/transform 拆分仍停留在 facade 层，CLI entry point 与安装元数据不一致。

这会降低实验结论可信度，也会让后续多模态诊断、互补分析和训练改动更容易互相影响；需要先做一次项目健康度稳定化，让验证基线重新变绿，并把核心边界收紧。

## What Changes

- 修正 Phase 1.5 final decision gate：当 checkpoint matrix 或 baseline matrix 存在 `pending` / `missing` 时，总决策保持探索性 `pending`，不输出 final 路线结论。
- 收紧轻量导入边界：导入单个 `engine`、`diagnostics`、`distillation` 子模块时，不应因为包级 `__init__` 预加载而导入训练器、可视化渲染、dataset、`pandas`、`scipy` 或 `matplotlib`。
- 将 `engine._builders_impl` 中的实现真正迁移到对应职责模块，保留 `kd_sensing.engine.builders` 作为兼容 facade。
- 将 `data.transform_ops._legacy` 中仍被使用的 image / GPS / LiDAR / mmWave / radar / IO / normalization 实现迁移到对应模态模块，保留 `kd_sensing.data.transforms` 旧导入兼容。
- 校正 Python console scripts 和安装元数据，使 `kd-sensing-visualize-modalities`、`kd-sensing-export-viewer-manifest` 等 README 中声明的入口可用且指向正确 CLI。
- 明确源码、内置复现权重和本地产物边界：已跟踪的 `All_models/*.pth` 需要有明确保留理由和校验；新生成 checkpoint 继续保持忽略，不再误入源码变更。
- 增加分层健康检查：最小导入 smoke、entry point help、Phase 1.5 runner、互补分析核心测试、以及项目环境下的快速回归命令。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `project-architecture`: 强化轻量导入边界、真实职责拆分、CLI entry point 可用性、源码与实验产物边界和健康检查要求。
- `phase-1-5-utility-validation`: 明确 Phase 1.5 final decision gate 必须同时考虑 bootstrap、dedicated baseline 和 checkpoint matrix 的完成状态，缺失项只能产生探索性输出。

## Impact

- 影响代码：`src/kd_sensing/engine/`、`src/kd_sensing/data/transform_ops/`、`src/kd_sensing/diagnostics/`、`src/kd_sensing/distillation/`、`src/kd_sensing/config/`、`pyproject.toml`、`README.md`、相关测试。
- 影响测试：需要在 `kd_mm_beam` 环境中恢复全量 pytest 绿灯，并新增轻量导入和 entry point regression。
- 影响使用方式：不改变现有训练、评估、预处理、Gradio viewer 或互补分析命令语义；旧 import facade 和旧脚本入口继续兼容。
- 影响实验产物：不移动用户本地 `dataset/`、`outputs/`、`logs/`；如调整已跟踪 `All_models` 策略，需要保留可复现实验路径和文档说明。
