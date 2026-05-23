## Context

Raymobtime s008 selection 流程已经把 current beam、LOS 和 link quality 作为一组 snapshot 任务接入统一训练入口。当前问题集中在两个边界：

- 数据边界：官方 ray-tracing zip 实际包含 `.hdf5` 文件，现有预处理只读取 `.csv/.txt/.json`，导致 ray table 为空，所有 `link_quality` 回退为 `-120 dBm`。
- 指标边界：selection 模型总是带有 beam、LOS、link 三个 head，现有 evaluation 和 TensorBoard 写入只要看到 head 输出就计算并写出指标，导致单任务实验混入未训练辅助 head 的曲线。

current snapshot beam selection 还需要一个距离敏感指标，但它不是 legacy future beam ADBA。沿用 `val_adba` 会让 DeepSense future horizon 语义和 Raymobtime current snapshot 语义混在一起。

## Goals / Non-Goals

**Goals:**

- 从官方 `.hdf5` ray-tracing zip 生成真实 ray 特征和非退化 `link_quality` target。
- 对 Raymobtime cache 增加质量门禁，避免全 fallback 或全常数 target 静默进入训练。
- 让 `current_beam_selection`、`current_los_classification`、`current_link_quality` 的正式 metrics、history 和 TensorBoard 只包含当前 objective 的指标。
- 为 current beam selection 增加 `val_beam_dba` 和 `beam/val_dba_current`，作为当前 beam 距离敏感指标。
- 保持 `selection_multitask` 继续同时输出 beam、LOS、link 三类正式指标。

**Non-Goals:**

- 不重写历史 TensorBoard event 文件或已生成的 `outputs/` 目录。
- 不把 Raymobtime selection 任务改回 future horizon 预测。
- 不把 `val_adba` 重新用于 Raymobtime current beam selection。
- 不新增绕过 `src/kd_sensing` 包结构的预处理入口。

## Decisions

### 1. HDF5 解析进入 Raymobtime 预处理模块

实现应在 `src/kd_sensing/preprocessing/raymobtime_s008.py` 内扩展 `_load_ray_table()`，支持 `.hdf5/.h5` zip 条目。解析结果先转换成 canonical ray rows，再复用现有 `_features_for_index()` 和 `_ray_feature_from_rows()`。

canonical row 至少包含：

- `sample_id` 或可构造 `sample_id` 的 `EpisodeID`、`SceneID`、`VehicleArrayID`
- power 字段，标准化为 `power_dbm`
- 可选 ToA、AoD、AoA、phase 字段，用于填充现有 ray feature

这样可以把 HDF5 支持限制在预处理边界内，不影响 dataset、model 和 training runtime。

替代方案：在 dataset runtime 中按样本读取 HDF5。拒绝该方案，因为它会把重 IO 引入训练循环，并让 cache 可复现性变差。

### 2. HDF5 依赖显式化并清晰失败

如果环境缺少 HDF5 读取依赖，预处理必须在遇到 `.hdf5` 条目时给出清晰错误。实现可以选择声明 `h5py` 依赖，也可以使用项目已有可维护方案；关键是不能跳过 `.hdf5` 后继续生成全 fallback cache。

替代方案：把 `.hdf5` 读取作为 optional best-effort。拒绝该方案，因为当前 bug 就来自静默跳过真实 ray source。

### 3. cache 质量门禁放在生成阶段

cache builder 应记录 matched/unmatched/fallback 统计和 link target 分布。当训练 split 的 `link_quality` 全常数、全部 fallback 或 ray path 全部缺失时，预处理失败。质量摘要写入 `cache_metadata.json` 和 unmatched report，便于复现实验时定位数据问题。

替代方案：训练时检测 link target 是否异常。该方案可作为二级保护，但不能替代预处理门禁，因为坏 cache 会污染所有后续实验和分析。

### 4. objective 过滤发生在正式指标提升层

evaluation pass 可以为了诊断临时计算模型输出，但正式 `val_*` metrics、`available_metrics`、history 和 TensorBoard 必须由 objective metadata 控制：

```
current_beam_selection ──► val_beam_top{1,3,5}, val_beam_dba
current_los_classification ──► val_los_accuracy, val_los_f1, val_los_auc
current_link_quality ──► val_link_mae, val_link_rmse, val_link_r2
selection_multitask ──► 上述三类 + val_selection_multitask_loss
```

非当前 head 的结果如果保留，只能放在内部诊断 payload，不能写入正式曲线。这样 TensorBoard 横向比较时不会把未训练辅助 head 当成实验结果。

替代方案：继续写全部 head 指标，但依赖 `available_metrics` 解释。拒绝该方案，因为 TensorBoard 图表不会理解 `available_metrics`，用户仍会看到误导曲线。

### 5. current beam DBA 使用新命名

新增指标命名为：

- flat metric: `beam_dba_current`
- validation metric: `val_beam_dba`
- TensorBoard tag: `beam/val_dba_current`

公式可以复用现有 DBA 的 top-3 距离衰减思想，但输入只包含当前 snapshot `[B, 1]` label。`val_adba` 继续保留给 legacy/future beam prediction，不在 Raymobtime current beam objective 中产生。

替代方案：直接恢复 `val_adba`。拒绝该方案，因为它会违反 current snapshot spec，且和 future horizon average DBA 语义冲突。

## Risks / Trade-offs

- [HDF5 内部 schema 可能与测试 fixture 不同] -> 先用真实官方文件做最小 spike，解析函数采用字段 alias 和结构遍历，但对缺失 power/sample 对齐字段清晰失败。
- [新增 HDF5 依赖会影响环境] -> 在 `pyproject.toml` 或环境文档中显式声明，并用预处理错误提示指导用户安装到 `kd_mm_beam`。
- [旧 dashboard 依赖单任务里的辅助 tag] -> 这是有意收紧；保留历史 event 文件，但新训练只写正式指标。若确需辅助曲线，可后续新增明确的 diagnostics 分组。
- [坏 cache 已存在于本地 outputs] -> 不自动删除用户产物；实现时通过 cache metadata 版本或质量校验提示用户重新运行预处理。
- [current beam DBA 与 legacy DBA 公式相近但命名不同] -> 在 metrics metadata 和测试中固定命名，避免将 `val_beam_dba` 汇总到旧 `val_adba` 管线。

## Migration Plan

1. 更新 HDF5 解析和 cache 质量门禁后，用户重新运行 Raymobtime s008 预处理，生成新的 cache。
2. 新训练会写出收紧后的 TensorBoard tags 和 metrics；旧 event 文件保留但不再作为新结果口径。
3. 实验汇总脚本若需要 beam 距离敏感指标，应读取 `val_beam_dba` 或 `beam/val_dba_current`，不再寻找 Raymobtime current objective 的 `val_adba`。

## Open Questions

- 官方 `.hdf5` 文件内字段名称和层级已用本地官方 `dataset/Raymobtime/s008/raw_data/ray_tracing_data_s008_carrier60GHz.zip` spike：zip 包含 2086 个 `.hdf5` 条目，每个条目包含 `allEpisodeData` dataset，样本 shape 为 `[1, 10, 25, 9]`；文件名 `_e<episode>` 映射 `EpisodeID`，第一维映射 `SceneID=0`，第二维映射 `VehicleArrayID`，第三维映射 path。字段映射为 `0=power_dbm`、`1=toa`、`2=elev_aod`、`3=az_aod`、`4=elev_aoa`、`5=az_aoa`、`8=phase`，字段 6 作为 path flag 记录但不进入模型输入，字段 7 在样本中不可用。
- 是否需要为辅助 head 诊断新增单独 TensorBoard 分组，例如 `diagnostics/link/mae`，本 change 暂不引入，避免继续混淆正式指标。
