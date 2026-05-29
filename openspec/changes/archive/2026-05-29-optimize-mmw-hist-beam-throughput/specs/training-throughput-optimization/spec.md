## ADDED Requirements

### Requirement: MMW HiST-Beam image-heavy 吞吐 profile
训练吞吐 profile MUST 支持 MMW HiST-Beam LOSO 配置，并输出足以定位 RGB/ImageNet image 序列解码、DataLoader wait、worker 内存和 GPU step 关系的结构化字段。profile MUST 不把 image decode/resize 等待误判为 CUDA 显存或模型结构问题。

#### Scenario: 输出 MMW image-heavy profile 字段
- **WHEN** 用户运行 `conda run -n kd_mm_beam python scripts/profile_training_io.py --config configs/hist_beam/mmw_scenario_loso.yaml`
- **THEN** profile 输出 MUST 记录 enabled modalities、seq_len、batch size、num_workers、prefetch_factor、pin_memory 和 persistent_workers
- **AND** profile 输出 MUST 记录 image、GPS、mmWave、beam label 和 radio-semantic 或 beam power 相关 `__getitem__` 耗时汇总
- **AND** profile 输出 MUST 记录 DataLoader wait、transfer、forward、backward/optimizer、step、samples/s 和 CUDA peak memory

#### Scenario: 标记 loader wait 支配 step
- **WHEN** DataLoader wait 的 P95 或总耗时显著高于 GPU step
- **THEN** profile 输出 MUST 标记 `loader_wait_dominates_step`
- **AND** 输出 MUST 指出优先检查 image decode/cache、batch size、worker 数和并行 run 数

### Requirement: MMW 并行训练内存风险推荐
并行训练推荐器 MUST 对 MMW image-heavy HiST-Beam 运行输出 memory-aware 覆盖建议。推荐 MUST 同时考虑 enabled modalities、seq_len、batch size、DataLoader worker 数、parallel run 数、系统内存和已有 profile 结果。

#### Scenario: image-heavy 并行运行推荐保守 worker
- **WHEN** 用户为包含 image modality 且 `seq_len >= 8` 的 MMW HiST-Beam 配置请求 4 路并行推荐
- **THEN** 推荐 MUST 限制每个 run 的 train worker 和 batch size，使预计 worker RSS 不超过可用内存预算
- **AND** 推荐 MUST 说明 AMP 不能解决 image decode 或 worker RSS 问题
- **AND** 推荐 MUST 包含 `output.progress.enabled=false` 或等价后台低噪声设置

#### Scenario: OOM 风险给出降载路径
- **WHEN** profile 或运行日志显示进程被系统 killed、退出码 137 或 worker RSS 过高
- **THEN** 推荐 MUST 输出降低 parallel runs、降低 batch size、降低 num_workers、禁用 persistent workers 或启用 image-derived cache 的建议
- **AND** 推荐 MUST 不默认建议继续增加 worker 数

### Requirement: MMW throughput 回归验证
项目 MUST 提供 focused 验证，确保 MMW image-heavy profile 和推荐字段稳定，并能在无全量真实数据提交的情况下运行。

#### Scenario: profile 字段稳定
- **WHEN** 测试或小样本 fixture 运行 MMW HiST-Beam profile helper
- **THEN** 输出 MUST 包含分模态 getitem、DataLoader wait、GPU/CPU step、loader config、seq_len、enabled modalities 和 IO-risk 字段
- **AND** 测试 MUST 可通过 `conda run -n kd_mm_beam pytest ... -q` 运行

#### Scenario: 推荐器识别 image-heavy 风险
- **WHEN** 推荐器收到包含 image modality、`seq_len=8` 和多个并行 run 的 MMW 配置
- **THEN** 推荐结果 MUST 包含 memory risk 或 image-heavy risk 诊断
- **AND** 推荐覆盖参数 MUST 优先限制 worker、batch size 或并行度，而不是只启用 AMP
