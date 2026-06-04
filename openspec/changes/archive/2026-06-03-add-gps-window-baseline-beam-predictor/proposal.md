## Why

当前 HiST-Beam few-shot 适配结果显示，神经网络方案在跨场景时仍可能受到 source prior、target oracle 使用边界和数值稳定性的影响。需要先建立一个不依赖神经网络、只使用 GPS/位姿滑动窗口的强诊断基线，用最简单可解释的几何与滤波假设覆盖所有可用场景，判断波束预测任务的下界、时序连续性收益和跨场景误差来源。

## What Changes

- 新增 GPS 滑动窗口非神经网络波束预测能力：从样本历史 GPS/pose 字段估计 RSU-CAV 相对方位、速度趋势和未来短期位置，映射到 beam codebook 后输出 Top-K 预测。
- 新增可配置算法族：支持 last-observed geometry、constant-velocity extrapolation、Savitzky-Golay/移动平均平滑、角速度平滑、beam-transition fallback 和候选参数网格。
- 新增全场景评估 CLI/配置：可对所有本地 MMW scenario 或指定 source/target 场景矩阵运行 GPS-only baseline，并输出与现有评估兼容的 `metrics.json`、预测直方图、collapse/误差诊断和调参日志。
- 新增迭代调参 artifact：每轮运行记录参数、场景、指标、误差分桶和推荐的下一轮参数候选，便于“根据每次结果调整算法”。
- 保持防泄漏边界：预测阶段不得使用 future beam label、beam power argmax、target_test 标签、path/radio/channel oracle 字段或神经网络训练权重。
- 不引入破坏性变更；现有 HiST-Beam、history-anchored 和 sensor-assisted 配置默认行为不变。

## Capabilities

### New Capabilities

- `gps-window-baseline-beam-prediction`: 定义仅使用 GPS/位姿滑动窗口的非神经网络 beam prediction baseline、评估产物、调参闭环和防泄漏契约。

### Modified Capabilities

- 无。

## Impact

- 影响代码：预计新增 `src/kd_sensing/baselines/` 或 `src/kd_sensing/engine/` 下的 GPS window baseline 实现、预测/评估 CLI、参数搜索与结果汇总模块。
- 影响配置：新增 `configs/hist_beam/` 或 `configs/baselines/` 下的 GPS window baseline 快速验证和全场景评估配置。
- 影响测试：新增几何映射、滤波/外推、防泄漏、CLI help、metrics artifact 和小型 synthetic scenario 测试。
- 影响产物：输出目录新增 GPS baseline 专用 metrics、prediction artifacts、iteration report 和 parameter sweep summary；不提交运行产物、日志或 checkpoint。
