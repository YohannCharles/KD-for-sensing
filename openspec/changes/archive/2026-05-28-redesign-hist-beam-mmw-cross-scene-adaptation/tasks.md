## 1. 数据准备与可用性登记

- [x] 1.1 修复 `src/kd_sensing/data/mmw/preparation.py` 的 channel scenario/agent 解析，使 `Town10/Town10_skybridge/cav_N/<frame>_paths.npz` 能与 sensor `Town10_skybridge_seed24/cav_N/<frame>` 按同一 CAV agent 匹配。
- [x] 1.2 为 sensor/channel scenario alias 增加配置字段、metadata 记录和 sanity report 统计，覆盖 `Town10_skybridge_seed24` ↔ `Town10_skybridge` 的显式匹配。
- [x] 1.3 增加 prepared artifact validity checks：finite beam power、非空 window、frame 连续性、CAV/channel agent 一致性、required modality coverage 和 mismatch examples。
- [x] 1.4 用 fixture 扩展 `tests/test_mmw_town10_preparation.py`，覆盖 channel agent 错配会失败、alias 匹配会成功、已有 sunny zip 路径可通过 override 解析。
- [x] 1.5 修复后使用 `conda run -n kd_mm_beam python scripts/mmw/prepare_town10_skybridge.py --config configs/preprocess/mmw_town10_skybridge.yaml --force -o mmw.sensor_zip=dataset/_downloads/MMW/sunny/Sensor_Data/Town10_skybridge_seed24.zip -o mmw.channel_zip=dataset/_downloads/MMW/sunny/Channel_Data/Town10.zip` 重建 sunny prepared 产物，并确认 manifest 中 `agent` 与 `channel_path` 的 CAV agent 一致。
- [x] 1.6 新增 MMW data availability metadata 生成逻辑，记录 ready/pending/single_scene_ready 状态、zip fingerprint、prepared root、frame/window 数和 claim guard 所需字段。

## 2. MMW enriched manifest 与几何特征

- [x] 2.1 扩展 frame manifest schema，记录 condition、town、sensor scenario、channel scenario、channel agent、sample id、CAV/RSU 可用模态路径和 modality availability。
- [x] 2.2 实现 CAV/RSU YAML pose 解析与 relative geometry builder，输出 relative range、azimuth、elevation、heading difference、local-frame 坐标和 direct geometry 标记。
- [x] 2.3 为 LiDAR occupancy、bbox、depth、radar point cloud、channel path count/energy spread 增加 proxy feature hooks，并在 metadata 中标记 direct/proxy 来源。
- [x] 2.4 扩展 `MMWDataset` 或相关 dataset descriptor，使 MMW batch 能按配置返回 geometry fields、channel-derived fields、CAV/RSU modality availability 和现有 beam/mmWave 序列字段。
- [x] 2.5 增加 MMW loader smoke 测试，使用 `conda run -n kd_mm_beam pytest tests/test_mmw_town10_preparation.py -q` 验证 prepared CSV、manifest、geometry fields 和 batch shape。

## 3. MMW LOSO 与 target adaptation workflow

- [x] 3.1 扩展 LOSO planner 的数据集 descriptor 输入，使 DeepSense6G 31-34 默认 folds 保持不变，同时支持 MMW scenario/town/condition fold 规划。
- [x] 3.2 实现 MMW single-scene smoke guard：当 ready scenario 少于两个时只生成 smoke/within-scenario sanity plan，并在 summary 中写入 `cross_scene_claim_allowed: false`。
- [x] 3.3 实现 MMW target_adapt/target_test deterministic split，按 sample id 和 sequence segment 防泄漏，并写出 split metadata。
- [x] 3.4 扩展 few-shot sampler，优先按 coarse sector 与 relative azimuth bin 分层采样 budgets `0,5,10,20,50`，不可用时记录 deterministic random fallback。
- [x] 3.5 扩展 LOSO summary 和 quick validation conclusion，记录 claim_scope、dataset family、condition/town/scenario、prototype diagnostics、geometry coverage 和 unavailable reasons。

## 4. Geometry-aware HiST-Beam 模型

- [x] 4.1 扩展 HiST-Beam 配置解析，新增 geometry-aware 开关、geometry field list、angular smoothing、geometry consistency、private prototype alignment 和 coarse-conditioned adapter 配置。
- [x] 4.2 在 `src/kd_sensing/models/fusion/hist_beam.py` 中加入 geometry token/projection，并输出 shared geometry representation、scene-private representation、adapter representation 和 geometry diagnostics。
- [x] 4.3 实现 coarse-sector aware fine mapping adapter，使 adapter 可由 coarse sector embedding 或 coarse context 条件化，并保持 zero-init 等价 source model。
- [x] 4.4 保持旧 DeepSense6G HiST-Beam 默认行为 opt-in 兼容，新增 geometry-aware 配置不影响未启用 geometry 的旧模型。

## 5. Loss、prototype 与 adaptation 修正

- [x] 5.1 在 `src/kd_sensing/engine/hist_beam_losses.py` 实现 angular smoothing loss，支持 linear/circular topology、sigma/temperature 和有效样本 diagnostics。
- [x] 5.2 实现 multimodal geometry consistency loss，按可用字段 mask 子 loss，并输出每个子 loss 的 coverage/unavailable reason。
- [x] 5.3 修改 source prototype 生成逻辑，使 artifact 支持 coarse sector 条件下的 private/adapter prototype、counts、field metadata 和 direct/proxy 摘要。
- [x] 5.4 修改 target adaptation，使 prototype consistency 默认使用 adapted private representation，按 coarse confidence threshold 选择 target，并记录 confidence、coverage、used sample count、prototype loss mean。
- [x] 5.5 在 `v5_adapter_proto` summary 中记录 prototype status；当权重为 0、artifact 缺失、coverage 为 0 或 prediction delta 为 0 时标记 no-op/inconclusive。
- [x] 5.6 增加模型/loss/adaptation 单元测试，使用 `conda run -n kd_mm_beam pytest tests/test_hist_beam*.py -q` 或新增等价测试文件验证新增 loss 与 prototype 梯度路径。

## 6. 配置、文档与实验入口

- [x] 6.1 新增 MMW geometry-aware smoke 配置，默认使用 sunny single-scene smoke，不声称跨场景结论。
- [x] 6.2 新增 MMW scenario-LOSO 配置模板，在至少两个 ready scenario 时启用，覆盖 source-only、adapter、adapter+prototype、full fine-tune 和 budgets。
- [x] 6.3 更新 README 或相关 docs，说明 MMW 本地下载路径、prepare 命令、single-scene smoke 限制、跨场景 claim guard 和推荐验证顺序。
- [x] 6.4 确保所有新增入口通过包内 CLI、console script 或现有 `scripts/mmw/prepare_town10_skybridge.py` 薄入口暴露，不新增绕过 `src/kd_sensing` 的旧式根目录脚本。

## 7. 验证与回归

- [x] 7.1 运行 `openspec validate redesign-hist-beam-mmw-cross-scene-adaptation --strict`。
- [x] 7.2 运行 `conda run -n kd_mm_beam pytest tests/test_mmw_town10_preparation.py -q`。
- [x] 7.3 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
- [x] 7.4 运行 HiST-Beam 相关单测或新增测试，验证旧 DeepSense6G 配置不受 geometry-aware opt-in 改动影响。
- [x] 7.5 对 MMW sunny 执行 smoke plan，确认 loader、forward、loss、summary、claim guard 和 prototype diagnostics 均能写出。
- [x] 7.6 在其它下载完成的 MMW 场景上增量 prepare，并在至少两个 ready scenario 后执行最小 scenario-LOSO 验证。
