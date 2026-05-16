## 1. OpenSpec 与配置

- [x] 1.1 创建仅覆盖 DeepVerse DT31 Phase 1 数据生成的 proposal、design 和 spec artifact。
- [x] 1.2 新增 `configs/deepverse/dt31_generation.yaml`，写明 DT31 场景、序列、beam、radar、split 和 dataset 下输出默认参数。

## 2. 核心生成模块

- [x] 2.1 新增 `src/kd_sensing/data/deepverse/` 包结构和 `DeepVerseDT31Generator`，支持缺包、缺 `config.m` 和参数保存的明确错误。
- [x] 2.2 实现 ULA DFT codebook、beam gain、LoS 到 blockage、位置噪声和 JSON/array 序列化辅助逻辑。
- [x] 2.3 实现 Phase 1 label builder，输出 manifest rows、labels、radar features、weak wireless、noisy position、camera/lidar index 和 skip counts。
- [x] 2.4 实现 split 与 sanity report 生成，默认使用 `split_by=sample` 80/20 产出可直接训练和验证的数据，同时支持显式 `split_by=ue`。

## 3. CLI 入口

- [x] 3.1 新增 `scripts/deepverse/generate_dt31_cache.py`，支持 CLI 参数与 YAML 配置合并。
- [x] 3.2 脚本输出 `metadata.json`、`samples.csv`、`labels.npz`、`radar_features.npz`、`weak_wireless.npz`、`noisy_position.npz`、`camera_index.json`、`lidar_index.json`、`split.json`、`sanity_report.json` 和 `used_generation_params.json`。

## 4. 验证

- [x] 4.1 添加 fake DeepVerse dataset 单元测试，覆盖 manifest/labels/cache/split/sanity 输出。
- [x] 4.2 使用 `conda run -n kd_mm_beam pytest tests/test_deepverse_dt31_generation.py` 验证 Phase 1 逻辑。
- [x] 4.3 使用 `conda run -n kd_mm_beam python scripts/deepverse/generate_dt31_cache.py --config configs/deepverse/dt31_generation.yaml --dry-run` 验证缺失外部依赖时的错误信息。

## 5. Phase 1.1 标签与 split 修正

- [x] 5.1 修正 LoS/blockage 语义：保留 raw `los_status_future` 分布，新增 `link_state_future`、`blockage_valid_mask` 和 ignore sentinel；不得把 `LoS_status=-1` 或未知值默认映射为 blockage 正类。
- [x] 5.2 增加 blockage 可用性 gate：当 valid blockage 不同时包含两类，或 minority class 低于最低样本数/比例时，metadata/sanity report 标记 `blockage.usable=false`，默认训练 objective 不启用 blockage。
- [x] 5.3 在 manifest 中记录可用于 split 的 `scene_id`、`sequence_id`/`segment_id`、`object_id` 或等价 group key；sample_id 应包含足够信息避免不同 scene/pass 的 time index 碰撞。
- [x] 5.4 将默认 split 从 `sample` 改为无泄漏的 `sequence`；若只有单连续轨迹，则使用 `time_contiguous` + purge/embargo，确保 train/val/test 的 raw history/future time index overlap 为 0。
- [x] 5.5 保留随机滑窗 split 仅作显式 debug 模式 `sample_random`，并在 metadata/sanity report 标记 `leakage_risk: high`；默认配置和 CLI help 不得推荐它作为验证口径。
- [x] 5.6 扩展 sanity report：输出 raw LoS/status 分布、blockage valid label 分布、blockage usability、split protocol、group counts、embargo span、discarded boundary windows 和跨 split raw frame overlap。
- [x] 5.7 添加单元测试：全 `LoS_status=-1` 时 blockage 被禁用；LoS/NLoS 两类充足时 blockage 可用；sequence split 无 group/raw-frame overlap；单连续轨迹 fallback 通过 purge/embargo 消除 overlap；`sample_random` 被标为 high leakage risk。
- [x] 5.8 重新生成真实 DT31 cache 并复核 `sanity_report.json`：blockage 不再作为全 1 监督标签静默通过，且默认 split 的 cross-split raw frame overlap 为 0。
