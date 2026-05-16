## Context

当前仓库的训练数据路径围绕 DeepSense6G CSV 展开，Scenario 31 的现有文件不能替代 DeepVerse6G-DT31 generator 输出。`deepverse6g.md` 将完整方案拆为多个阶段，本轮只实现 Phase 1：通过 DeepVerse dataset object 读取通信 channel、radar channel、LoS、mobility location、camera path 和 LiDAR path，生成 manifest 与 labels。运行约束是所有 Python 命令使用 `conda run -n kd_mm_beam ...`。

本机当前没有 `deepverse` 包，也没有 DT31 场景目录；因此代码必须在真实依赖缺失时清楚失败，并允许用测试替身验证 cache 构建逻辑。

## Goals / Non-Goals

**Goals:**

- 提供 `scripts/deepverse/generate_dt31_cache.py`，能从 DT31 场景生成 Phase 1 cache。
- 输出 `metadata.json`、`samples.csv`、`labels.npz`、`radar_features.npz`、`weak_wireless.npz`、`noisy_position.npz`、`camera_index.json`、`lidar_index.json`、`split.json`、`sanity_report.json` 和 `used_generation_params.json`。
- 标签语义固定为 future `t+1` beam 和 `t+1:t+K` trajectory；blockage 仅在 LoS/status 语义可验证且类别可用时作为监督标签启用，同时保留多 horizon arrays 和 raw LoS/status。
- 默认按 sequence/segment 做 80/20 train/val split；若真实 DT31 只暴露单条连续轨迹，则使用 contiguous temporal split 并在 split 边界施加 purge/embargo，确保滑窗 history/future 原始时间索引不跨 split 重叠。

**Non-Goals:**

- 不实现 DeepVerse 训练 Dataset、collate、模型、loss、metrics 或实验矩阵。
- 不把 full comm channel、clean position 或 LoS history 作为默认模型输入。
- 不强制安装 `deepverse` 或下载 DT31 场景数据。
- 不处理复杂 radar range-Doppler 表示；本轮使用低维 radar 统计特征。

## Decisions

1. **将 DeepVerse 适配层与 DeepSense6G 代码隔离。**
   新增 `src/kd_sensing/data/deepverse/`，不改现有 DeepSense6G dataset 主流程。这样可以避免 DT31 generator 的外部依赖影响已有训练和测试。

2. **使用稳健的反射式 DeepVerse 访问。**
   DeepVerse 不同版本的 sample 名称、字段和参数结构可能不完全一致。Generator 只负责导入 `ParameterManager`/`Dataset`、设置存在的参数、保存最终参数；Label builder 通过候选 sample 名称和字段读取 comm/mobility/camera/lidar，失败时记录跳过原因。

3. **Phase 1 只保存轻量索引与 numpy artifacts。**
   Camera/LiDAR 默认保存路径索引，不在 cache 阶段强制加载图像或点云 tensor。Radar、labels、weak wireless 和 noisy position 保存为 `.npz`，manifest 使用 CSV，便于在缺少 parquet 依赖时仍可运行。

4. **Beam 标签由 DFT/ULA codebook 派生。**
   `compute_beam_gain()` 对 DeepVerse 示例中常见的 `[N_ant, N_ue, N_sc]` channel 做 shape 检查，输出 gain vector、top-k power、entropy 和 label。该路径不把 full channel 写为默认输入。

5. **Radar 默认作为输入模态保存为低维统计特征。**
   `radar_features.npz` 保存每个历史帧的 `abs_mean`、`abs_std`、`abs_max`、`phase_diff_mean`、`phase_diff_std` 和 `path_count`，避免 Phase 1 直接绑定复杂 range-Doppler 表示。

6. **DT31 默认使用 sequence/segment 级 split，禁止随机滑窗作为默认评估。**
   滑窗样本共享大量相邻原始帧。当前 cache 中随机 sample split 使 train/val 的 raw history/future time index 大面积重叠，不能作为严肃验证口径。默认 `split_by` 改为 `sequence`：优先按 DeepVerse scene/pass/object/segment id 分组，在每个 split 内单独生成窗口或至少按完整窗口组分配 split。若 dataset API 只能暴露单条连续轨迹，则退化为 `time_contiguous`，按时间顺序切 80/20，并丢弃或屏蔽 split 边界附近 `seq_len + pred_horizon - 1` 范围内会跨界复用 raw frames 的窗口。`sample_random` 仅保留为显式 debug 模式，metadata 和 sanity report 必须标记 `leakage_risk: high`。

7. **Blockage 是可选监督标签，必须带 valid mask 和可用性检查。**
   DeepVerse/DeepMIMO 风格 LoS status 至少存在 `1=LoS`、`0=NLoS/indirect`、`-1=no path or unavailable` 等语义差异。当前真实 cache 的 `los_status_future` 全为 `-1`，直接按 `LoS_status != 1` 映射为 blockage 会把所有样本变成正类，既可能误读 sentinel，也无法训练/评估二分类。Phase 1.1 改为保存 raw `los_status_future`、`link_state_future`、`blockage_valid_mask`；仅当 status 语义明确且 `{0,1}` 两类都达到最低样本数/比例时，metadata 才将 `blockage` 标为可训练 objective。否则 `blockage_label` 使用 ignore sentinel，sanity report 写明 `usable: false` 和原因。

## Risks / Trade-offs

- [Risk] 本地缺少 `deepverse` 包或 DT31 场景，无法真实生成 cache。→ Mitigation: 脚本启动时显式检查并给出缺失路径/包名，单元测试用 fake dataset 覆盖核心构建逻辑。
- [Risk] DeepVerse DT31 sample API 与公开示例不完全一致。→ Mitigation: 读取路径采用候选名称和字段 fallback，所有失败原因进入 `skip_counts` 与 sanity report。
- [Risk] 只保存 camera/lidar path 不能保证训练阶段可直接读 tensor。→ Mitigation: Phase 1 的验收只要求 path/index 存在性记录；训练 Dataset 属于后续 Phase 2。
- [Risk] Radar raw/FMCW tensor 字段随 DeepVerse 版本变化。→ Mitigation: Phase 1 使用基于 `coeffs` 的稳定低维统计特征，并在 sanity report 中检查 NaN/Inf。
- [Risk] Beam codebook 假设 antenna 维度位置错误。→ Mitigation: `compute_beam_gain()` 在两个常见维度上匹配 codebook antenna 数，否则抛出带 shape 的错误。
- [Risk] 随机 sample split 会让相邻滑窗或同一 raw frame 同时出现在 train/val，验证指标偏高。→ Mitigation: 默认改为 sequence/segment split；单连续轨迹使用 contiguous split + purge/embargo；sanity report 必须输出 raw frame overlap 检查。
- [Risk] DT31 的 LoS/status 字段可能是 tri-state 或 sentinel，导致 blockage 全为单类。→ Mitigation: blockage 标签必须带 valid mask、raw status distribution 和 class balance gate；无法满足 gate 时默认禁用 blockage objective。
