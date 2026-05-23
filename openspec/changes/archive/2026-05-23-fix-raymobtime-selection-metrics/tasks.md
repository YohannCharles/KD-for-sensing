## 1. HDF5 ray-tracing 解析与 cache 质量

- [x] 1.1 使用 `conda run -n kd_mm_beam` 对官方 `ray_tracing_data_s008_carrier60GHz.zip` 中的 `.hdf5` 样本做 schema spike，记录可用的 episode、vehicle、path、power、ToA、AoD、AoA 和 phase 字段映射。
- [x] 1.2 在项目依赖或环境校验中声明 HDF5 读取依赖，并确保缺失依赖时 Raymobtime s008 预处理给出清晰错误。
- [x] 1.3 扩展 `src/kd_sensing/preprocessing/raymobtime_s008.py` 的 ray table 加载逻辑，支持 `.hdf5/.h5` zip 条目并转换为 canonical ray rows。
- [x] 1.4 复用现有 ray feature 聚合路径生成 `ray_features_no_los`、`ray_features_with_los` 和真实 `link_quality`，避免 LOS 泄漏到模型输入。
- [x] 1.5 在 cache metadata 和 unmatched report 中记录 matched/unmatched/fallback 统计、link target 分布和 cache 质量摘要。
- [x] 1.6 添加质量门禁：ray path 全部缺失、link target 全 fallback 或训练 split link target 标准差为 0 时拒绝生成可训练 cache。

## 2. Current beam DBA 与 objective 指标边界

- [x] 2.1 在 evaluation metrics 中实现 current snapshot beam DBA 计算，输出 `beam_dba_current` 和 `val_beam_dba`，不写 `val_adba`。
- [x] 2.2 更新 objective metadata：`current_beam_selection` 的 available metrics、aliases、metric mode、history fields 和 TensorBoard scalar 包含 `val_beam_dba` / `beam/val_dba_current`。
- [x] 2.3 拆分 Raymobtime selection 单任务 TensorBoard scalar 映射，使 beam、LOS、link 单任务只写各自正式指标，`selection_multitask` 才写三类指标。
- [x] 2.4 收紧 evaluation pass 的正式指标提升逻辑，非当前 objective 的 head 输出不得写入正式 `val_*` metrics、history 或 TensorBoard 主指标。
- [x] 2.5 保持 `selection_multitask` 同时输出 beam Top-K、`val_beam_dba`、LOS、link 和 `val_selection_multitask_loss`。
- [x] 2.6 更新训练日志、checkpoint metric 提取和 `training_outputs.npz` payload，确保新增 `val_beam_dba` 可记录且旧 `val_adba` 不出现在 Raymobtime current objective 中。

## 3. 测试覆盖

- [x] 3.1 为 Raymobtime HDF5 fixture 或真实结构适配添加预处理单元测试，验证 `.hdf5` 条目可解析并产生非全 fallback 的 `link_quality`。
- [x] 3.2 添加 cache 质量门禁测试，覆盖全 missing ray path、全 `-120 dBm` fallback 和 link target std 为 0 的失败路径。
- [x] 3.3 添加 current beam DBA 指标测试，验证命名、公式、alias、metric mode 和不产生 `val_adba`。
- [x] 3.4 添加 objective metadata/TensorBoard 测试，验证三个 Raymobtime 单任务不会写入其它任务的正式 tag。
- [x] 3.5 添加 evaluation pass/training metrics 测试，验证 `available_metrics`、history 和 epoch log 只暴露当前 objective 指标。
- [x] 3.6 更新或新增 selection multitask 测试，验证多任务仍同时暴露 beam、LOS、link 和总 loss 指标。

## 4. 验证与迁移说明

- [x] 4.1 运行 `openspec validate fix-raymobtime-selection-metrics --strict`，确认 proposal、design、spec 和 tasks 可归档。
- [x] 4.2 运行 `conda run -n kd_mm_beam pytest tests/test_raymobtime_s008_selection.py tests/test_prediction_objectives.py tests/test_training_io_workflow.py -q`。
- [x] 4.3 运行 `conda run -n kd_mm_beam pytest tests/test_evaluation_pass.py -q`，覆盖共享 evaluation pass 行为。
- [x] 4.4 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`，确认依赖和导入边界未被 HDF5 支持破坏。
- [x] 4.5 使用 `conda run -n kd_mm_beam python scripts/preprocess.py --help` 和 Raymobtime s008 预处理 dry run 或小样本 fixture 验证 CLI 路径。
- [x] 4.6 在变更说明中记录旧的全 `-120 dBm` Raymobtime cache 需要重新生成，历史 TensorBoard event 文件不会自动重写。
