# multimodal-nf-dataset Specification

## Purpose
定义 Multimodal-NF 作为独立本地数据集家族的运行契约：覆盖默认目录与审计、HDF5 frame index、flat dict sample、按 profile 懒加载 image/LiDAR/GPS/CSI、近场三维 codebook flattened beam target、LOS/link 等辅助标签，以及可用小型 fixture 验证的 dataset smoke workflow。该 capability 同时约束本地真实数据、codebook、cache、审计报告、训练输出和 checkpoint 的产物边界。
## Requirements
### Requirement: Multimodal-NF 本地数据布局与审计
系统 MUST 支持 `multimodal_nf` dataset type，并将默认数据集家族目录解析为 `dataset/MultimodalNF`。系统 MUST 提供配置驱动审计流程，用于检查 Multimodal-NF HDF5、image/lidar 压缩包或 HDF5、codebook 文件、city 列表、HDF5 keys、shape、dtype 和 split 可用性。

#### Scenario: 默认目录解析
- **WHEN** 用户配置 `data.dataset.type: multimodal_nf` 且未设置 `data.dataset.data_root`
- **THEN** 系统 MUST 使用 `dataset/MultimodalNF` 作为默认数据根目录
- **AND** 该路径 MUST 通过项目根路径解析工具解析

#### Scenario: 审计官方数据文件
- **WHEN** 用户运行 Multimodal-NF 审计预处理
- **THEN** 系统 MUST 检查 channel HDF5、image 数据、LiDAR 数据和 `upa64x64_NF_codebook*.pkl` 是否存在
- **AND** 审计输出 MUST 记录每个 HDF5 文件的 keys、shape、dtype、city id、样本数和缺失项
- **AND** 审计输出 MUST 写入 `outputs/`、`dataset/MultimodalNF/cache` 或用户配置的 ignored 目录

#### Scenario: 不自动迁移真实数据
- **WHEN** 本地存在 Hugging Face 下载目录、外部数据目录或旧目录结构
- **THEN** 系统 MUST 不自动移动、复制或删除真实数据文件
- **AND** 用户 MUST 能通过显式 `data_root`、`channel_root`、`image_root`、`lidar_root` 或 `codebook_path` 继续使用该路径

### Requirement: Multimodal-NF HDF5 index 构建
系统 MUST 从 Multimodal-NF HDF5 和 metadata 构建 frame-wise sample index。每个样本 MUST 对应单个 UAV trajectory frame，并 MUST 可追踪到 city、trajectory 和 frame。

#### Scenario: 构建 frame-wise index
- **WHEN** index builder 读取 `City_###` 数据
- **THEN** 每个 frame MUST 生成一个稳定 `sample_id`
- **AND** `sample_id` MUST 包含或可反查 city id、trajectory id 和 frame id
- **AND** index row MUST 包含 channel、image、LiDAR、position、beam target 和辅助标签引用

#### Scenario: city-level split
- **WHEN** 用户使用默认 split 策略
- **THEN** 系统 MUST 按 city 或配置定义的 city 集合划分 train/validation/test
- **AND** 同一 city 的样本 MUST 不同时出现在多个 split 中，除非配置显式选择 frame-level debug split
- **AND** split metadata MUST 记录 city 列表、样本数、beam label 分布、LoS/NLoS 分布和 NF/FF 分布

### Requirement: Multimodal-NF dataset sample 契约
`multimodal_nf` dataset MUST 返回 flat dict sample，并根据启用模态和 profile 懒加载输入。dataset MUST 不读取未启用模态对应的大数组或 HDF5 dataset。

#### Scenario: GPS+CSI 取样
- **WHEN** 用户启用 `gps` profile `uav_xyz_snapshot` 和 `csi` profile `xl_mimo_nf`
- **THEN** 样本 MUST 包含 `gps` 和 `csi`
- **AND** `gps` MUST 具有 `[1, 3]` 当前 UAV 位置语义
- **AND** `csi` MUST 具有 `[1, 4096, K, 2]` 或配置选择后的等价 near-field XL-MIMO channel 语义
- **AND** 样本 MUST 不包含 `image` 或 `lidar`

#### Scenario: Image+LiDAR 取样
- **WHEN** 用户启用 `image` 和 `lidar` profile `point_cloud_xyz_10000`
- **THEN** `image` MUST 返回 `[1, 3, H, W]` RGB tensor
- **AND** `lidar` MUST 返回 `[1, P, 3]` 点云 tensor，默认 P 为 10000
- **AND** 如果原始 image 为 `[H, W, 3]`，dataset 或 adapter MUST 转换为 channel-first RGB tensor

#### Scenario: Metadata 字段
- **WHEN** 用户设置 `return_metadata: true`
- **THEN** 样本 metadata MUST 至少包含 `sample_id`、`dataset_type`、`city_id`、`trajectory_id`、`frame_id`、`split`、启用 profile 和源文件路径或引用

#### Scenario: split-specific 验证抽样
- **WHEN** 用户为 Multimodal-NF dataset 配置 `eval_portion` 或 split-specific portion
- **THEN** 系统 MUST 只对对应 validation/test split 应用该抽样
- **AND** train split MUST 默认保持 `portion` 或 `train_portion` 指定的样本范围
- **AND** run metadata MUST 记录实际应用的 selected portion

### Requirement: Multimodal-NF 近场 beam target 契约
Multimodal-NF dataset MUST 支持三维 codebook Top-5 beam target。主训练标签 MUST 以 `target_beam` 暴露为 flattened class，同时保留 triplet 与 power 信息用于指标、诊断和后续结构化方法。

#### Scenario: Top-5 target 输出
- **WHEN** 样本包含 `BeamIdx` 和 `BeamPower`
- **THEN** dataset MUST 返回 `target_beam` 为 Top-1 triplet flatten 后的一维当前标签
- **AND** dataset MUST 返回 `beam_triplet_topk`，形状为 `[5, 3]`
- **AND** dataset MUST 返回 `beam_power_topk`，形状为 `[5]`
- **AND** dataset metadata 或 dataset 属性 MUST 记录 codebook shape 和 flatten 规则

#### Scenario: codebook 文件缺失
- **WHEN** 用户启用 near-field beam target 但 codebook 文件或 codebook shape 不可解析
- **THEN** 系统 MUST 拒绝构建 dataset
- **AND** 错误信息 MUST 指出缺失的 codebook 配置和可用路径

### Requirement: Multimodal-NF 辅助标签契约
Multimodal-NF dataset MUST 暴露 LoS/NLoS、NF/FF、trajectory mode 或等价辅助标签，用于分析、过滤和可选辅助任务。

#### Scenario: 辅助标签输出
- **WHEN** 原始 HDF5 包含 `Has_LoS`、`Is_NF`、`Traj_Is_NLoS` 或 `Mode_Idx`
- **THEN** dataset MUST 将可用标签以 `los_label`、`nf_label`、`traj_nlos_label` 和 `mode_idx` 或等价规范字段输出
- **AND** 缺失某个辅助标签时系统 MUST 在 metadata 中记录不可用状态，而不是静默填入错误值

### Requirement: Multimodal-NF 配置和 smoke workflow
系统 MUST 提供 Multimodal-NF 预处理、dataset smoke、单模态 near-field beam selection 和 fusion 配置样例。smoke workflow MUST 可在无真实全量数据的情况下通过小型 fixture 验证核心契约。

#### Scenario: 预处理配置
- **WHEN** 用户运行 `conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/multimodal_nf_audit.yaml`
- **THEN** 命令 MUST 能输出审计报告或清晰的数据缺失错误
- **AND** 命令 MUST 不要求提交任何真实数据或生成 cache 到源码变更

#### Scenario: smoke 测试
- **WHEN** 开发者运行 Multimodal-NF focused tests
- **THEN** 测试 MUST 使用小型 fixture 验证 HDF5 index、dataset fields、target flatten、Top-5 metadata、profile shape 和 data factory 构建
- **AND** 测试命令 MUST 使用 `conda run -n kd_mm_beam pytest ... -q`

### Requirement: Multimodal-NF helper 拆分兼容
Multimodal-NF preprocessing 和 dataset helper 拆分后，审计、index 构建、split assignment、codebook metadata、HDF5 inspection 和 dataset sample 契约 MUST 保持兼容。公开 preprocessor registry 名称和配置入口 MUST 不改变。

#### Scenario: 审计入口保持
- **WHEN** 用户运行 `conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/multimodal_nf_audit.yaml`
- **THEN** 命令 MUST 继续调用 Multimodal-NF 审计流程
- **AND** 输出字段和缺失数据错误语义 MUST 保持兼容

#### Scenario: index 构建入口保持
- **WHEN** 用户运行 Multimodal-NF index 构建配置
- **THEN** 系统 MUST 继续输出相同语义的 sample index 和 metadata
- **AND** split assignment、sample_id、codebook metadata 和 target 引用 MUST 保持兼容

#### Scenario: dataset sample 保持
- **WHEN** 测试从 Multimodal-NF dataset 取样
- **THEN** flat sample keys、metadata、enabled modality lazy loading 和 target fields MUST 与拆分前兼容
- **AND** focused tests MUST 不依赖真实全量数据

### Requirement: Multimodal-NF capability purpose 明确
`multimodal-nf-dataset` spec MUST 使用真实目的说明描述当前 capability，覆盖本地数据布局、审计、HDF5 index、flat sample、profile 懒加载、近场 codebook target、辅助标签和 smoke workflow。该 spec MUST 不长期保留 archived TBD Purpose 文案。

#### Scenario: spec purpose 不再是 TBD
- **WHEN** 开发者阅读 `openspec/specs/multimodal-nf-dataset/spec.md`
- **THEN** Purpose MUST 描述 Multimodal-NF dataset capability 的当前职责
- **AND** Purpose MUST NOT 包含 `TBD - created by archiving`

### Requirement: Multimodal-NF objective runtime 语义
Multimodal-NF run metadata MUST 根据当前 objective 记录准确的 task semantics 和 target schema。系统 MUST 区分 dataset family 能力与当前 run 实际使用的 objective，不得把所有 Multimodal-NF run 都描述为同一个 future beam task。

#### Scenario: near-field beam selection runtime
- **WHEN** 用户运行 `data.dataset.type: multimodal_nf` 且 `experiment.objective: near_field_beam_selection`
- **THEN** runtime metadata MUST 记录 objective 为 `near_field_beam_selection`
- **AND** target schema MUST 表达 Multimodal-NF 近场三维 codebook flattened beam class
- **AND** metadata MUST 记录 codebook shape、flatten order 和 num beam classes

#### Scenario: LOS runtime
- **WHEN** 用户运行 `data.dataset.type: multimodal_nf` 且 `experiment.objective: current_los_classification`
- **THEN** runtime metadata MUST 记录 objective 为 `current_los_classification`
- **AND** target schema MUST 表达 LOS/NLOS binary classification
- **AND** metadata MUST 不把该 run 的主任务描述为 beam-only prediction

#### Scenario: selection multitask runtime
- **WHEN** 用户运行 `data.dataset.type: multimodal_nf` 且 `experiment.objective: selection_multitask`
- **THEN** runtime metadata MUST 记录 beam selection、LOS 和 link quality targets 均启用
- **AND** metadata MUST 记录每个 head 或 output 字段的 target 语义

### Requirement: Multimodal-NF codebook consistency
系统 MUST 校验 Multimodal-NF codebook metadata 与模型输出类别数的一致性。若配置解析出的 codebook `num_beam_classes` 与 beam head `num_classes` 不一致，系统 MUST 在启动阶段抛出清晰错误或拒绝写出自相矛盾的 final config。

#### Scenario: codebook 类别数一致
- **WHEN** Multimodal-NF dataset 解析出 codebook shape 和 `num_beam_classes`
- **THEN** 模型 beam head 的输出类别数 MUST 与 `num_beam_classes` 一致
- **AND** `final_config.yaml` 或 runtime metadata MUST 记录该一致性来源

#### Scenario: codebook 类别数不一致
- **WHEN** `data.dataset.codebook_metadata.num_beam_classes` 与模型 beam head 类别数不一致
- **THEN** 系统 MUST 抛出包含两个实际值和相关配置路径的清晰错误
- **AND** 系统 MUST 不继续启动一个会产生不可解释指标的训练 run

### Requirement: Multimodal-NF image/LiDAR 派生缓存
Multimodal-NF dataset MUST 支持对 image 和 LiDAR 模态使用显式配置的派生缓存或重打包数据源，以减少训练时对原始 gzip HDF5 大数组的重复解压。派生缓存 MUST 是本地产物，默认写入 `dataset/MultimodalNF/cache`、`outputs/` 或用户显式配置的 ignored 目录；系统 MUST 不自动改写、删除或移动原始 Multimodal-NF HDF5 文件。

#### Scenario: 派生缓存关闭时保持原始读取
- **WHEN** 用户未启用 Multimodal-NF image/LiDAR 派生缓存或设置策略为 `off`
- **THEN** dataset MUST 继续从原始 HDF5 文件读取启用模态
- **AND** 返回的 sample keys、tensor shape、dtype 语义和 target 字段 MUST 与现有契约兼容

#### Scenario: 派生缓存命中
- **WHEN** 用户为 image 或 LiDAR 设置派生缓存策略且缓存 metadata 与原始文件 fingerprint、profile、split、`seq_len` 和 `num_pred` 匹配
- **THEN** dataset MUST 从派生缓存读取对应启用模态
- **AND** 返回 tensor MUST 与原始 HDF5 读取路径在 shape、dtype 语义和样本顺序上等价
- **AND** runtime metadata MUST 记录该模态实际使用派生缓存、缓存路径、策略和匹配的原始 fingerprint

#### Scenario: read_only 缓存缺失
- **WHEN** 用户设置派生缓存策略为 `read_only` 且所需 image 或 LiDAR 缓存不存在或 metadata 不匹配
- **THEN** dataset 构建 MUST 失败
- **AND** 错误信息 MUST 包含模态名、策略、期望缓存路径和缺失或不匹配原因

#### Scenario: auto 或 rebuild 生成缓存
- **WHEN** 用户设置派生缓存策略为 `auto` 或 `rebuild`
- **THEN** 系统 MUST 能通过显式预处理流程或受控 dataset 初始化流程生成所需派生缓存
- **AND** 生成结果 MUST 写出 metadata sidecar，记录原始文件路径或 fingerprint、profile、shape、dtype、样本数、生成时间、`seq_len` 和 `num_pred`
- **AND** 生成过程 MUST 使用原子写入或等价机制，避免多进程读取半成品缓存

#### Scenario: 未启用模态不访问缓存
- **WHEN** 用户运行 GPS-only、CSI-only 或不包含 image/LiDAR 的 fusion 配置
- **THEN** dataset MUST 不解析、不创建、不读取 image 或 LiDAR 派生缓存路径
- **AND** 对应原始 image/LiDAR 文件缺失不得阻止当前任务运行

### Requirement: Multimodal-NF 派生缓存审计与可追踪性
Multimodal-NF 审计、runtime metadata 和 profile 输出 MUST 能说明 image/LiDAR 数据来自原始 HDF5 还是派生缓存。系统 MUST 将缓存策略、缓存覆盖率和缓存有效性记录为机器可读字段。

#### Scenario: 审计报告包含缓存状态
- **WHEN** 用户运行 Multimodal-NF 审计或缓存预热流程
- **THEN** 输出 MUST 记录每个启用缓存模态的缓存路径、存在状态、样本数、profile、shape、dtype 和原始 fingerprint 匹配结果
- **AND** 输出 MUST 不要求真实大缓存文件进入源码变更

#### Scenario: runtime metadata 记录实际数据源
- **WHEN** 训练或评估构建 Multimodal-NF dataset
- **THEN** runtime metadata MUST 按 split 和模态记录 `source_kind`，其值区分原始 HDF5 和派生缓存
- **AND** metadata MUST 记录派生缓存策略、缓存路径和是否发生生成或回退

#### Scenario: 缓存失效清晰失败或回退
- **WHEN** 派生缓存 metadata 与当前原始文件、profile 或窗口参数不一致
- **THEN** `read_only` 策略 MUST 清晰失败
- **AND** `auto` 策略 MUST 清晰记录回退到原始 HDF5 或重新生成缓存的行为

### Requirement: Multimodal-NF 派生缓存轻量校验
Multimodal-NF image/LiDAR 派生缓存 MUST 支持轻量运行时校验和显式强校验。默认训练或 `read_only` cache 读取路径 MUST 不在每次 dataset 构建时重新扫描原始 HDF5 大文件计算 fingerprint；强 fingerprint 校验 MUST 通过显式预处理、审计、重建或配置选项触发，并 MUST 在 metadata 中记录结果。

#### Scenario: read_only cache 初始化不扫描原始大文件
- **WHEN** 用户配置 `data.cache.multimodal_nf.image.policy=read_only` 或 `data.cache.multimodal_nf.lidar.policy=read_only`，且对应 sidecar 的轻量校验字段有效
- **THEN** dataset 初始化 MUST 使用 sidecar 中的路径、大小、mtime、profile、split、`seq_len`、`num_pred`、shape、dtype 和 cache version 进行轻量校验
- **AND** dataset 初始化 MUST NOT 重新读取整个原始 HDF5 文件计算 SHA256 fingerprint
- **AND** runtime metadata MUST 记录 validation mode、是否执行 source fingerprint scan 和校验耗时

#### Scenario: 显式强校验
- **WHEN** 用户通过预处理、审计或配置显式请求 Multimodal-NF 派生缓存强校验
- **THEN** 系统 MUST 重新计算原始源文件 fingerprint 并与 sidecar 记录值比较
- **AND** 系统 MUST 在校验报告或 runtime metadata 中记录强校验耗时、结果和不匹配原因
- **AND** 若 policy 为 `read_only` 且强校验不匹配，系统 MUST 拒绝读取该 cache 并输出包含 cache path、source path 和 mismatch 字段的错误

#### Scenario: 旧 sidecar 缺少轻量校验字段
- **WHEN** sidecar 缺少轻量校验必需字段或 cache version 不可识别
- **THEN** `read_only` policy MUST 给出清晰错误，提示用户执行强校验、重建或切换 policy
- **AND** `auto` policy MAY 按现有策略重建或回退到原始 HDF5，但 MUST 在 metadata 中记录 fallback/rebuild 原因

### Requirement: Multimodal-NF 派生缓存 IO 布局元数据
Multimodal-NF image/LiDAR 派生缓存 sidecar MUST 记录足以诊断和优化随机窗口读取的 IO 布局信息。dataset 和 profile MUST 能从 sidecar/runtime metadata 中报告 cache 的 storage kind、layout、分片或 source key、样本数、字节数、shape、dtype 和推荐访问模式。

#### Scenario: sidecar 记录 IO 布局
- **WHEN** 系统生成或重建 Multimodal-NF image/LiDAR 派生缓存
- **THEN** sidecar MUST 记录 `storage_kind`、`layout`、`source_key` 或等价 shard 标识、`sample_count`、`bytes`、shape、dtype、cache version 和推荐访问模式
- **AND** sidecar MUST 继续记录原始 source path 或等价 source identity，便于审计和回退

#### Scenario: runtime metadata 暴露 cache 读取计划
- **WHEN** dataset 使用 Multimodal-NF image/LiDAR 派生缓存构建 train/test split
- **THEN** runtime metadata MUST 记录每个启用模态的 source kind、cache policy、validation mode、cache path 数量、总字节数和是否可能产生随机读风险
- **AND** 未启用 image/LiDAR 的配置 MUST 不解析、不校验、不打开对应派生 cache

#### Scenario: 派生 cache 样本等价
- **WHEN** 同一样本可从原始 HDF5 路径和派生 cache 路径读取
- **THEN** 派生 cache 路径 MUST 返回与原始路径等价的 sample keys、tensor shape、dtype 语义和 target 字段
- **AND** 任何新的 shard 或布局格式 MUST 保持该等价契约并提供可回退到原始 HDF5 的策略

### Requirement: Multimodal-NF 派生缓存 lazy 读取边界
Multimodal-NF dataset adapter MUST 以 worker-local lazy 方式打开派生 cache，并 MUST 避免在 worker 初始化或首次样本读取前 eager mmap 所有 city/source cache 文件。实现 MAY 提供打开文件数或映射字节数上限，但 MUST 保持读取结果与现有 sample 契约兼容。

#### Scenario: worker 只打开当前样本需要的 cache
- **WHEN** DataLoader worker 第一次读取某个 Multimodal-NF image/LiDAR 样本
- **THEN** adapter MUST 只打开该样本 source 对应的 cache 文件或 shard
- **AND** adapter MUST NOT 因启用该模态而立即 mmap 当前 split 下所有 cache 文件

#### Scenario: cache 打开状态可诊断
- **WHEN** profile 或 runtime metadata 请求 cache IO 诊断
- **THEN** 系统 MUST 能报告已打开 cache 文件数、映射字节数或等价计数
- **AND** 该诊断 MUST 不改变样本读取结果

### Requirement: Multimodal-NF 旧派生缓存 sidecar 迁移
Multimodal-NF image/LiDAR 派生缓存 MUST 支持将可验证的旧 sidecar 元数据升级为当前轻量校验 schema，而不重写对应 `.npy` 数据文件。系统 MUST 只有在旧 sidecar、`.npy` header、当前 source identity 和配置参数足以确认 cache 仍适用于当前 profile、split、`seq_len` 和 `num_pred` 时执行 metadata-only upgrade。

#### Scenario: 预处理升级旧 sidecar 不重写数据
- **WHEN** 用户运行 Multimodal-NF derived cache 预处理且 `rebuild=false`，并且某个 image 或 LiDAR `.npy` 文件存在、旧 sidecar 可迁移
- **THEN** 系统 MUST 补齐当前 cache schema 所需的 lightweight metadata 字段
- **AND** 系统 MUST 使用原子写入更新 sidecar JSON
- **AND** 系统 MUST NOT 重写对应 `.npy` 数据文件
- **AND** 预处理输出 MUST 记录该 source 的结果为 metadata upgraded 或等价机器可读状态

#### Scenario: auto 策略优先执行 metadata-only upgrade
- **WHEN** 训练或 dataset 构建使用 Multimodal-NF image/LiDAR cache `policy=auto`，并且所需 cache 的 `.npy` 文件存在但 sidecar 为可迁移旧 schema
- **THEN** 系统 MUST 优先执行 metadata-only sidecar upgrade
- **AND** 系统 MUST 在 upgrade 后重新执行 lightweight cache status 校验
- **AND** 系统 MUST 只有在 metadata-only upgrade 不安全或失败时才按现有策略重建 cache 或回退到原始 HDF5

#### Scenario: read_only 不自动写 sidecar
- **WHEN** 用户配置 Multimodal-NF image/LiDAR cache `policy=read_only`，并且所需 cache 的 `.npy` 文件存在但 sidecar 为可迁移旧 schema
- **THEN** dataset 构建 MUST 失败并保持 sidecar 不变
- **AND** 错误信息 MUST 明确说明 cache data exists but sidecar migration is pending 或等价语义
- **AND** 错误信息 MUST 包含可执行的预处理升级或强校验命令提示

#### Scenario: 不安全旧 sidecar 拒绝迁移
- **WHEN** 旧 sidecar 与当前 source path、source fingerprint、profile、split、`seq_len`、`num_pred`、shape、dtype 或 sample count 不匹配
- **THEN** 系统 MUST 拒绝 metadata-only upgrade
- **AND** `read_only` policy MUST 清晰失败
- **AND** `auto` policy MUST 记录 mismatch 原因并按现有安全策略重建 cache 或回退

#### Scenario: 强校验升级记录 fingerprint scan
- **WHEN** 用户显式请求 strong validation 迁移旧 sidecar
- **THEN** 系统 MUST 重新计算原始 source fingerprint 并与 sidecar 记录值比较
- **AND** 系统 MUST 在 sidecar 或预处理输出中记录 strong validation 耗时、结果和是否扫描 source fingerprint
- **AND** fingerprint 不匹配时系统 MUST 拒绝 metadata-only upgrade

### Requirement: Multimodal-NF 派生缓存迁移状态可追踪
Multimodal-NF cache status、runtime metadata 和预处理输出 MUST 能区分 valid cache、migration pending cache、invalid cache 和 missing cache。该状态 MUST 以机器可读字段暴露，便于 profile、推荐器和训练错误信息复用。

#### Scenario: cache status 暴露 migration pending
- **WHEN** 系统检查一个存在 `.npy` 数据文件且 sidecar 为可迁移旧 schema 的 Multimodal-NF image/LiDAR cache
- **THEN** cache status MUST 记录 `migration_pending=true` 或等价机器可读字段
- **AND** status MUST 记录 sidecar schema version、cache path、source path、validation mode 和待补齐字段摘要

#### Scenario: runtime metadata 记录升级行为
- **WHEN** dataset 构建过程中因 `auto` policy 执行 metadata-only sidecar upgrade
- **THEN** runtime metadata MUST 记录该模态发生 metadata upgrade
- **AND** metadata MUST 区分 `cache_generated=false`、`cache_rebuilt=false` 和 `metadata_upgraded=true` 或等价字段

#### Scenario: 预处理汇总迁移结果
- **WHEN** 用户运行 Multimodal-NF derived cache 预处理
- **THEN** 输出 MUST 按模态和 split 汇总 valid/skipped、metadata upgraded、rebuilt/generated、failed 和 missing 数量
- **AND** 输出 MUST 不包含真实大 cache 内容

