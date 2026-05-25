## ADDED Requirements

### Requirement: Multimodal-NF 吞吐 profile
训练吞吐 profile MUST 支持 Multimodal-NF image、LiDAR、GPS、CSI 和 fusion 配置，并输出足以定位 image/LiDAR HDF5 解压、派生缓存、DataLoader wait、CPU 到 GPU transfer 和模型 step 的结构化字段。

#### Scenario: 输出 Multimodal-NF 模态级 getitem
- **WHEN** 用户运行 `conda run -n kd_mm_beam python scripts/profile_training_io.py --config <multimodal-nf-config>`
- **THEN** profile 输出 MUST 记录每个启用 Multimodal-NF 模态的 `__getitem__` 均值、P50、P95、最小值和最大值
- **AND** 输出 MUST 包含 dataset 总 `__getitem__`、auxiliary targets、DataLoader wait、transfer、forward、backward/optimizer、step、samples/s 和 CUDA peak memory 字段

#### Scenario: 输出 Multimodal-NF 数据源和缓存策略
- **WHEN** profile 的配置使用 Multimodal-NF dataset
- **THEN** profile 输出 MUST 记录每个 split 和启用模态的数据源类型、派生缓存策略、缓存路径、缓存命中或缺失状态
- **AND** 如果 profile 过程中触发缓存生成，输出 MUST 记录生成行为或生成耗时摘要

#### Scenario: 对比 GPS 与 image/LiDAR 瓶颈
- **WHEN** 用户分别 profile Multimodal-NF GPS-only、image-only、LiDAR-only 和 fusion 配置
- **THEN** 输出字段 MUST 足以比较各模态 `__getitem__` 与 GPU step 的相对耗时
- **AND** profile MUST 不把 image/LiDAR 的数据读取耗时归入 GPS 或 target 字段

### Requirement: Multimodal-NF 吞吐配置推荐
项目 MUST 提供面向 Multimodal-NF image/LiDAR/fusion 训练的配置推荐，帮助用户选择 DataLoader worker、prefetch、pin memory、test worker、AMP、progress 和派生缓存策略。推荐 MUST 以命令行覆盖或说明形式输出，除非用户明确要求，不得直接修改用户配置文件。

#### Scenario: 含 image 和 LiDAR 的 fusion 推荐
- **WHEN** 用户请求 Multimodal-NF image+LiDAR+GPS fusion 的吞吐推荐
- **THEN** 推荐 MUST 包含启用或预热 image/LiDAR 派生缓存的建议
- **AND** 推荐 MUST 包含合理的 `num_workers`、`prefetch_factor`、`pin_memory`、`persistent_workers`、`test_num_workers`、`training.amp.enabled` 和 `output.progress.enabled` 覆盖建议
- **AND** 推荐 MUST 说明这些建议用于吞吐优化，用户仍可按机器资源调整

#### Scenario: 缓存未准备时提示预热
- **WHEN** 推荐器或 profile metadata 发现 Multimodal-NF image/LiDAR 派生缓存不存在或覆盖率不足
- **THEN** 输出 MUST 提示先运行缓存预热或使用 `auto`/`rebuild` 策略
- **AND** 输出 MUST 不默认建议把缺失缓存的配置强行改成 `read_only`

#### Scenario: GPS-only 不推荐重缓存
- **WHEN** 用户运行或请求 GPS-only Multimodal-NF 配置推荐
- **THEN** 推荐 MUST 不要求 image 或 LiDAR 派生缓存
- **AND** 推荐 MUST 说明 GPS-only 主要受模型 step 和普通 DataLoader 参数影响

### Requirement: Multimodal-NF 吞吐回归验证
实现 Multimodal-NF 吞吐优化后，项目 MUST 提供 focused 验证，确保派生缓存路径不改变样本契约，并且 profile 能捕获 image/LiDAR 的性能差异。验证 MUST 使用 fixture 或小样本本地数据，不能要求提交真实全量缓存。

#### Scenario: 派生缓存与原始读取等价
- **WHEN** 测试使用小型 Multimodal-NF fixture 同时构建原始 HDF5 dataset 和派生缓存 dataset
- **THEN** 两条路径返回的 image/LiDAR tensor shape、target fields、metadata 关键字段和样本顺序 MUST 等价
- **AND** 未启用模态 MUST 不出现在样本中

#### Scenario: profile 输出字段稳定
- **WHEN** 测试运行 focused profile 或直接调用 profile helper
- **THEN** 输出 MUST 包含 Multimodal-NF 模态级 getitem、数据源、缓存策略、DataLoader split 参数和 samples/s 字段
- **AND** 该测试 MUST 可通过 `conda run -n kd_mm_beam pytest ... -q` 在无真实全量数据时运行

#### Scenario: 配置兼容性
- **WHEN** 用户未配置 Multimodal-NF 派生缓存字段
- **THEN** 现有 Multimodal-NF 配置 MUST 继续构建 dataset 和 DataLoader
- **AND** focused tests MUST 覆盖旧配置默认行为
