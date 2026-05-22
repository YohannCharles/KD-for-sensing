## 1. OpenSpec 与架构检查准备

- [x] 1.1 调整 `tests/test_architecture_boundaries.py` 的 OpenSpec purpose 检查，只解析每个 spec 的 `## Purpose` 段落，避免全文字符串自引用误伤。
- [x] 1.2 补齐或加长当前 `openspec validate --all --strict` 报告中过短的 spec purpose，不改变 requirement 语义。
- [x] 1.3 增加架构 probe，验证 `import kd_sensing.config` 不导入 `torch`、dataset、model、diagnostics visualization core 或训练主循环。

## 2. Objective 元数据轻量化

- [x] 2.1 新增轻量 objective metadata 模块，承载 objective 列表、默认 metric、metric mode、aliases、available metrics、history fields、TensorBoard scalar、required targets/outputs 和 runtime metadata。
- [x] 2.2 改造 `kd_sensing.engine.prediction_objectives`，让 torch target/loss helper 复用轻量 objective metadata，并保持既有公开 helper 和 `__all__` 兼容。
- [x] 2.3 改造 `kd_sensing.config.normalization`、config validation、training state 或 training metrics 中只需要元数据的导入路径，避免配置路径导入 torch loss/runtime。
- [x] 2.4 运行 objective 相关 focused 测试：`conda run -n kd_mm_beam pytest tests/test_prediction_objectives.py tests/test_raymobtime_s008_selection.py -q`；若不存在其中某个测试文件，则运行覆盖 objective 的现有相关测试。

## 3. 诊断可视化 import 边界

- [x] 3.1 清理 `kd_sensing.diagnostics.visualization.config` 的不必要重依赖 import，仅保留配置解析、路径、模态契约和 JSON helper 所需依赖。
- [x] 3.2 清理 `sampling`、`writers` 等轻量 helper 的 import，使其不导入 dataset builder、model builder、matplotlib、PIL 或 visualization core。
- [x] 3.3 将 pandas、torch、PIL、matplotlib、dataset builder 等重依赖保留在 `datasets`、`render`、manifest/prediction 导出路径或实际使用函数内部。
- [x] 3.4 增加架构 probe，验证导入 `diagnostics.visualization.config`、`sampling`、`writers` 不触发渲染栈或 dataset builder。

## 4. 验证与收尾

- [x] 4.1 运行 OpenSpec change 校验：`openspec validate refine-architecture-boundaries --strict`。
- [x] 4.2 运行架构边界快速检查：`conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
- [x] 4.3 运行 viewer/诊断相关回归：`conda run -n kd_mm_beam pytest tests/test_modality_visual_diagnostics.py tests/test_gradio_complementarity_explorer.py -q`。
- [x] 4.4 运行最终回归：`conda run -n kd_mm_beam pytest -q`。
