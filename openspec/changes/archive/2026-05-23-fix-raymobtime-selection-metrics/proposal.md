## Why

当前 Raymobtime s008 selection 实验存在两个会误导结果解读的问题：ray-tracing zip 中的真实 `.hdf5` 路径文件未被解析，导致 `link_quality` 全部回退为 `-120 dBm`；同时单任务 objective 的训练日志和 TensorBoard 会写出未训练辅助 head 的 LOS/link/beam 指标，造成“当前任务结果”和“诊断输出”混在一起。

此外，current snapshot beam selection 与历史 future beam DBA 语义不同。用户仍需要一个 beam 距离敏感指标，但它必须使用当前 beam 的明确命名，避免复用 `val_adba` 带来的语义混淆。

## What Changes

- 修复 Raymobtime s008 ray-tracing 预处理：支持从官方 `ray_tracing_data_s008_carrier60GHz.zip` 中解析 `.hdf5` 文件，构建真实 path-level ray 特征和 `link_quality` target。
- 增加 Raymobtime s008 cache 数据质量校验：当 ray path 全部缺失、`link_quality` 全常数或回退值占比异常时，预处理 MUST 报出清晰错误或在审计报告中显式标记不可用于 link 训练。
- 收紧 Raymobtime selection 单任务的正式指标边界：`current_beam_selection` 只暴露 beam 指标，`current_los_classification` 只暴露 LOS 指标，`current_link_quality` 只暴露 link 指标；非当前任务 head 的输出只能作为诊断信息，不得写入正式 validation metrics、history 或 TensorBoard 主图。
- 为 current beam selection 新增距离敏感指标，使用明确命名 `val_beam_dba` 和 TensorBoard tag `beam/val_dba_current`，基于当前 beam label 计算，不复用 legacy future `val_adba`。
- 保持 `selection_multitask` 的多任务语义：该 objective 仍应同时输出 beam Top-K、current beam DBA、LOS 指标、link 指标和总 loss。
- 更新测试与验证命令，覆盖 HDF5 解析、cache 质量门禁、objective metric 过滤、TensorBoard tag 映射和 current beam DBA 语义。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `raymobtime-s008-selection`: 修改 Raymobtime s008 ray-tracing cache 构建、link target 数据质量要求、selection 指标集合和 current beam DBA 指标契约。
- `first-class-prediction-tasks`: 修改 prediction objective 元数据、available metrics、history fields 和 TensorBoard scalar 映射，确保单任务 objective 只暴露当前 objective 指标，并新增 current beam DBA 的 primary/secondary metric 可用性。

## Impact

- 影响 `src/kd_sensing/preprocessing/raymobtime_s008.py` 的 ray-tracing zip 解析、cache metadata、unmatched report 和质量校验。
- 影响 `src/kd_sensing/evaluation/metrics.py`、`src/kd_sensing/engine/evaluation_pass.py`、`src/kd_sensing/engine/objective_metadata.py`、`src/kd_sensing/engine/training_metrics.py` 和 TensorBoard 写入逻辑。
- 影响 Raymobtime s008 相关 `metrics.json`、`train_log.json`、`training_outputs.npz` 和 TensorBoard tag；历史已生成 event 文件不会自动重写。
- 旧的全 `-120 dBm` Raymobtime cache 属于坏数据产物，需要用户重新运行预处理生成；该目录仍属于本地 ignored 产物，不纳入源码变更。
- 可能新增或声明 HDF5 读取依赖；实现应优先使用项目环境中可维护的读取方式，并在依赖缺失时报出清晰安装/环境错误。
