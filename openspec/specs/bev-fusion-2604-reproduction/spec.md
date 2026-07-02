# bev-fusion-2604-reproduction Specification

## Purpose
定义 arXiv:2604.05668 BEV-Fusion 复现实验的配置、模型、数据契约、训练评估、报告、ablation 和本地产物边界要求，使 2604 复现能力能够以可审计、可测试且不依赖真实数据的方式进入当前项目规范。

## Requirements

### Requirement: 2604 paper-aligned 实验协议配置
系统 SHALL 提供 arXiv:2604.05668 BEV-Fusion 复现实验配置族。主配置 MUST 使用 DeepSense6G scenarios 32、33、34，启用 image、radar、gps、lidar 四个模态，使用 5 帧历史输入、`future_beam1` 单 horizon 标签、64 类 beam 分类和 linear DBA metric profile。smoke 或 low-memory 配置 MAY 缩小 BEV grid 或模型维度，但 MUST 在配置和 runtime metadata 中标记为 paper approximation。

#### Scenario: 加载 paper full 配置
- **WHEN** 开发者加载 BEV-Fusion 2604 paper full 配置
- **THEN** 配置 MUST 设置 `data.dataset.type: deepsense6g`
- **AND** 配置 MUST 覆盖 scenarios 32、33、34 的训练和评估协议
- **AND** 配置 MUST 设置 `seq_len: 5`、`num_pred: 1`、`num_classes: 64`
- **AND** 配置 MUST 启用 image、radar、gps、lidar 四个输入模态
- **AND** 配置 MUST 使用 linear DBA 或明确命名的 2604 metric profile

#### Scenario: 加载 smoke 配置
- **WHEN** 开发者加载 BEV-Fusion 2604 smoke 配置
- **THEN** 配置 MUST 能在不读取真实 DeepSense6G 数据的 synthetic 或 mock batch 上完成模型构建和 forward
- **AND** 配置 MUST 将 `mock_data` 或 `paper_approximation` metadata 标记为 true
- **AND** smoke 指标 MUST NOT 被报告为真实 DeepSense6G 复现结果

### Requirement: BEV-Fusion 2604 模型注册与输出契约
系统 SHALL 注册 `bev_fusion_2604` 模型。该模型 MUST 能通过 `MODELS.build()` 从配置构建，MUST 接收现有 fusion batch preparation 提供的模态输入，并 MUST 返回当前 engine 可适配的 dict 输出，其中 `logits` 形状为 `[B, 1, 64]`，`input_features` 和 `output_features` 可用于诊断或后续 metric 记录。

#### Scenario: 构建 BEV-Fusion 模型
- **WHEN** 配置指定 `model.primary.type: bev_fusion_2604`
- **THEN** `MODELS.build()` MUST 返回可调用的 PyTorch 模型实例
- **AND** 模型 MUST 记录 `modalities`、`bev_size`、`d_model`、camera backbone、temporal core 和 GPS pathway 配置

#### Scenario: 标准四模态 forward
- **WHEN** 模型收到 batch-aligned 的 image、radar、gps、lidar 输入
- **THEN** 模型 MUST 输出 `logits`，形状为 `[batch_size, 1, 64]`
- **AND** 输出 MUST 能被 `adapt_model_output()` 适配
- **AND** 输出 diagnostics MUST 包含 BEV feature shape、effective modalities 和 GPS pathway 状态

#### Scenario: 缺少启用模态
- **WHEN** 模型配置启用某个模态但 forward 未收到对应输入
- **THEN** 模型 MUST 抛出清晰错误
- **AND** 错误信息 MUST 指出缺失模态名称和当前启用模态列表

### Requirement: Camera-to-BEV learned cross-attention
系统 SHALL 为 paper full 配置实现 learned camera-to-BEV transformation。camera 分支 MUST 保留二维视觉 feature map，将其序列化为视觉 tokens，并使用 learnable BEV queries 通过 cross-attention 生成 `[B, T, D, H_bev, W_bev]` camera BEV feature。paper full 配置 MUST 支持 ResNet-34 backbone；smoke 配置 MAY 使用轻量 backbone，但 MUST 标记为 approximation。

#### Scenario: Camera BEV 输出 shape
- **WHEN** camera-to-BEV 分支接收 `[B, T, 3, H_img, W_img]` RGB 输入
- **THEN** 分支 MUST 输出 `[B, T, d_model, bev_height, bev_width]`
- **AND** 输出的 `bev_height` 和 `bev_width` MUST 与模型配置一致

#### Scenario: paper full 使用 spatial camera features
- **WHEN** paper full 配置构建 camera-to-BEV 分支
- **THEN** 分支 MUST 使用二维 feature map tokens 参与 cross-attention
- **AND** 分支 MUST NOT 只把全局 pooled image embedding broadcast 到 BEV grid

#### Scenario: camera-to-BEV 参数可审计
- **WHEN** 训练或评估写出 runtime metadata
- **THEN** metadata MUST 记录 camera backbone、是否使用预训练权重、BEV query 数、attention 层数、attention heads 和 `d_model`

### Requirement: LiDAR、Radar 与 GPS spatial BEV 分支
系统 SHALL 将 LiDAR、Radar 和 GPS spatial pathway 对齐到与 camera 相同的 BEV grid。LiDAR 分支 MUST 接收 dataset 返回的 LiDAR BEV tensor 并投影到 `d_model` 通道；Radar 分支 MUST 接收现有 radar RA/DA 表示并投影到 BEV grid；GPS spatial pathway MUST 使用未被训练集标准化破坏的局部 XY 或等价可审计坐标生成 spatial mask。

#### Scenario: LiDAR BEV 对齐
- **WHEN** LiDAR input 的空间尺寸与模型 `bev_size` 不一致
- **THEN** 模型 MUST 使用显式 interpolation 或配置化 projection 对齐尺寸
- **AND** diagnostics MUST 记录原始 LiDAR BEV shape 和对齐后的 BEV shape

#### Scenario: Radar BEV 对齐
- **WHEN** radar 分支接收 RA/DA 或等价 radar batch
- **THEN** 模型 MUST 将 radar feature 投影到 `[B, T, d_model, H_bev, W_bev]`
- **AND** diagnostics MUST 记录 radar mapping profile

#### Scenario: GPS spatial 坐标缺失
- **WHEN** 配置启用 GPS spatial pathway 但 batch 中缺少 `gps_bev_xy` 或等价未标准化坐标来源
- **THEN** 模型或 batch preparation MUST 抛出清晰错误
- **AND** 错误信息 MUST 指出可通过配置提供 GPS BEV 坐标或改用 `gps_global_only` ablation

### Requirement: Dual-path GPS-to-BEV 编码
系统 SHALL 支持论文式 dual-path GPS 编码。GPS spatial pathway MUST 生成 BEV spatial feature 并参与每个时隙的 BEV fusion；GPS global pathway MUST 使用全精度 GPS 序列生成 global embedding，并在 temporal fusion 后通过 gated residual 注入最终 beam representation。系统 MUST 支持 `dual_path`、`spatial_only` 和 `global_only` ablation。

#### Scenario: dual-path GPS forward
- **WHEN** GPS pathway 配置为 `dual_path`
- **THEN** 模型 MUST 同时计算 GPS spatial BEV feature 和 GPS global embedding
- **AND** 模型 MUST 在 diagnostics 中记录 gated residual 标量或等价可学习 gate 值

#### Scenario: spatial-only GPS ablation
- **WHEN** GPS pathway 配置为 `spatial_only`
- **THEN** 模型 MUST 只使用 GPS spatial BEV feature 参与 BEV fusion
- **AND** 模型 MUST NOT 在 temporal fusion 后注入 GPS global embedding

#### Scenario: global-only GPS ablation
- **WHEN** GPS pathway 配置为 `global_only`
- **THEN** 模型 MUST 跳过 GPS spatial BEV feature
- **AND** 模型 MUST 保留 GPS global embedding gated residual 或等价全局注入路径

### Requirement: BEV spatial fusion 与 temporal aggregation
系统 SHALL 在 BEV 空间融合各模态 feature，并在 spatial fusion 后执行 temporal aggregation。paper full 配置 MUST 使用 temporal transformer；ablation 配置 MUST 支持 1D fusion、single-frame 和 mean-pooling temporal aggregation，并在 metadata 中记录 ablation 名称。

#### Scenario: paper full spatial fusion
- **WHEN** 四个模态 BEV feature 已对齐到同一 grid
- **THEN** 模型 MUST 在 BEV 空间融合这些 feature
- **AND** temporal transformer 的输入 MUST 来自 fused BEV representation，而不是各模态全局 pooled 1D feature 的直接拼接

#### Scenario: 1D fusion ablation
- **WHEN** ablation 配置选择 `one_d_fusion`
- **THEN** 模型 MAY 绕过 BEV spatial fusion 并使用全局 pooled feature 拼接
- **AND** runtime metadata MUST 将该运行标记为 BEV representation ablation

#### Scenario: temporal ablation
- **WHEN** ablation 配置选择 `single_frame` 或 `mean_pool_temporal`
- **THEN** 模型 MUST 替换 temporal transformer 为对应 aggregation 策略
- **AND** report MUST 将该结果与 paper full model 分开汇总

### Requirement: Paper-style 训练、loss 与 augmentation 配置
系统 SHALL 提供 paper-style 训练配置。默认优化器 MUST 为 AdamW，paper full 配置 MUST 使用 `lr=1e-4`、`weight_decay=1e-2` 或在 metadata 中记录偏差；loss MUST 使用 focal loss `gamma=2`，class weighting 或 alpha MUST 只从训练 split 标签统计中估计。训练增强 MUST 不破坏 beam label；beam-aware horizontal flip 只有在同步 beam index reversal 有测试覆盖时才能启用。

#### Scenario: focal loss 配置
- **WHEN** 加载 paper full 训练配置
- **THEN** loss MUST 解析为 focal loss
- **AND** `gamma` MUST 为 2 或在 metadata 中记录 paper deviation
- **AND** class frequency 统计 MUST 只来自训练 split

#### Scenario: optimizer 配置
- **WHEN** 加载 paper full 训练配置
- **THEN** optimizer MUST 解析为 AdamW
- **AND** learning rate 和 weight decay MUST 记录在最终配置和 runtime metadata 中

#### Scenario: beam-aware flip 安全边界
- **WHEN** 配置启用 horizontal flip augmentation
- **THEN** 系统 MUST 同步执行 beam index reversal
- **AND** 如果 reversal 规则缺失或未通过测试，配置解析或训练启动 MUST 拒绝该 augmentation

### Requirement: 2604 评估报告与可比性声明
系统 SHALL 生成或记录 2604 复现实验报告所需 metadata。报告 MUST 包含每个 scene 的 DBA 与 Top-K、macro DBA、weighted overall DBA 或明确选择的 overall 口径、论文目标值、差距、split protocol、seed、样本数、metric profile、mock/real 标记、paper exact split 是否可用，以及本地硬件和可选 latency/参数量统计。

#### Scenario: 真实评估报告
- **WHEN** BEV-Fusion 2604 真实数据评估完成
- **THEN** 报告 MUST 分别列出 S32、S33、S34 的 DBA 和 Top-K
- **AND** 报告 MUST 列出论文目标 DBA `86.60%`、`86.27%`、`86.70%` 和 overall `86.52%`
- **AND** 报告 MUST 标明本地 split/seed 是否与论文 exact split 一致

#### Scenario: mock 评估报告
- **WHEN** 评估使用 mock 或 synthetic 数据
- **THEN** 报告 MUST 标记 `mock_data: true`
- **AND** 报告 MUST NOT 将 mock 指标放入真实 DeepSense6G 复现结论

#### Scenario: metric 口径可审计
- **WHEN** report 或 metrics artifact 写出 DBA
- **THEN** artifact MUST 记录 DBA 是否 linear、circular、Top-K distance-based 或其它命名 profile
- **AND** 不同 DBA 口径 MUST NOT 混用同一未限定字段名

### Requirement: Ablation matrix 可复现
系统 SHALL 提供 paper-style ablation 配置或 recipe，至少覆盖移除 camera、移除 LiDAR、移除 radar、移除 GPS、1D fusion、single-frame、mean-pooling temporal、GPS spatial-only 和 GPS global-only。每个 ablation MUST 保持同一数据 split、metric profile 和训练评估报告字段，除非配置明确记录差异。

#### Scenario: 移除单个模态
- **WHEN** 加载 `without_camera`、`without_lidar`、`without_radar` 或 `without_gps` ablation 配置
- **THEN** 配置 MUST 禁用对应模态分支
- **AND** 其它训练协议、label、metric profile 和 output metadata MUST 与 full model 保持一致

#### Scenario: ablation metadata
- **WHEN** 任一 ablation 训练或评估完成
- **THEN** runtime metadata MUST 记录 `ablation_name`
- **AND** report MUST 能按 full model 与 ablation 分组展示结果

### Requirement: 测试、轻量导入与产物边界
系统 SHALL 为 BEV-Fusion 2604 复现能力提供不依赖真实 DeepSense6G 数据的快速测试，并保持现有本地产物边界。新增模型不得破坏配置、路径、模态契约等轻量导入路径；真实训练输出、cache、checkpoint、日志和报告默认写入 ignored 的 `outputs/`、`logs/` 或等价本地产物目录。

#### Scenario: 模型 forward smoke test
- **WHEN** 测试使用小型 synthetic 四模态 batch 构建 `bev_fusion_2604`
- **THEN** forward MUST 完成
- **AND** logits shape MUST 为 `[B, 1, 64]`
- **AND** diagnostics MUST 包含 BEV shape 和 effective modalities

#### Scenario: 配置加载测试
- **WHEN** 测试加载 full、smoke 和 ablation 配置
- **THEN** 配置加载 MUST 成功
- **AND** 每个配置 MUST 解析到预期模型类型、模态集合、BEV size、loss 和 metric profile

#### Scenario: 轻量导入不牵出重依赖
- **WHEN** 开发者导入 `kd_sensing.config`、路径工具或模态契约
- **THEN** 新增 BEV-Fusion 模型 MUST NOT 被 eager import 到这些轻量路径

#### Scenario: 本地产物不进入源码变更
- **WHEN** 用户运行 BEV-Fusion 2604 训练、评估、report 或 cache workflow
- **THEN** 新生成的 checkpoint、cache、metrics、TensorBoard、report 和日志 MUST 位于忽略规则覆盖的本地产物路径
- **AND** 源码变更 MUST NOT 要求提交这些本地产物

### Requirement: 2604.05668 对齐 supervised 下游验证
项目 MUST 提供 image+GPS supervised 2604 对齐配置族，用于与 arXiv:2604.05668 的 S32/S33/S34 主表口径比较。该配置族 MUST 合并 DeepSense6G scenes 32、33、34 的官方 train/test labeled CSV，MUST 在每个 scene 内按 beam label 做固定 seed 的 80/10/10 stratified train/validation/test split，MUST 使用 `seq_len: 5` 和 `num_pred: 1`，并 MUST 记录 split protocol 与每个 split 的样本数。

#### Scenario: 2604 配置使用合并后 stratified split
- **WHEN** 用户加载 2604 对齐 supervised 配置
- **THEN** `data.dataset.split_protocol` MUST 为 `stratified_80_10_10`
- **AND** `data.dataset.train_scenes`、`validation_scenes` 和 `test_scenes` MUST 包含 32、33 和 34
- **AND** train/validation/test MUST 来源于每个 scene 的 `train_seqs_RA_GPS_LIDAR.csv` 与 `test_seqs_RA_GPS_LIDAR.csv` 合并集合

#### Scenario: 2604 配置匹配历史窗口
- **WHEN** 2604 对齐 supervised 配置被加载
- **THEN** `data.dataset.seq_len` 和 `model.seq_length` MUST 为 5
- **AND** `data.dataset.num_pred` 和 `model.num_pred` MUST 为 1

#### Scenario: 2604 split 使用 train-only normalization
- **WHEN** 2604 split protocol 构建 image+GPS dataloader
- **THEN** GPS scaler MUST 只从 stratified train 子集拟合
- **AND** validation/test MUST 复用该 scaler
- **AND** runtime metadata MUST 记录 scaler 来源与 split protocol
