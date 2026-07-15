# gps-preprocessing Specification

## Purpose
定义 GPS 序列列、relative-polar 特征和 scaler artifact 生成要求。
## Requirements
### Requirement: GPS 序列列生成
系统 MUST 在 Scenario 9 序列生成流程中支持保留 GPS 路径列，使启用 GPS 的 dataset 能按历史时隙读取 UE 和 BS 经纬度数据。序列 CSV MUST 至少包含每个历史时隙对应的 UE GPS 路径和 BS GPS 路径；默认 UE GPS MUST 优先使用校准后的 `unit2_loc_cal`，BS GPS MUST 使用 `unit1_loc`。

#### Scenario: 生成携带 GPS 的序列 CSV
- **WHEN** 用户通过预处理入口从 `scenario9.csv` 生成输入长度为 8、预测长度为 3 的序列 CSV，并启用 GPS 列输出
- **THEN** 系统 MUST 生成 `gps1` 到 `gps8` 的 UE GPS 路径列
- **AND** 系统 MUST 生成 `bs_gps1` 到 `bs_gps8` 的 BS GPS 路径列或等价的 BS GPS 引用列
- **AND** 系统 MUST 保留既有 camera、radar、beam、future_beam 和 `seq_index` 列

#### Scenario: 旧序列 CSV 兼容非 GPS 实验
- **WHEN** 用户使用不包含 GPS 列的旧序列 CSV 运行 image-only、radar-only 或未启用 GPS 的 fusion 配置
- **THEN** 系统 MUST 不要求 CSV 中存在 GPS 列
- **AND** 系统 MUST 保持既有数据加载行为不变

### Requirement: GPS 经纬度读取
系统 MUST 能从 DeepSense GPS 文本文件读取纬度和经度，并将其转换为可用于特征构造的数值数组。读取失败、缺失文件或格式非法时，启用 GPS 的配置 MUST 抛出包含路径信息的清晰错误。

#### Scenario: 读取 GPS 文本
- **WHEN** dataset 读取一个包含两行经纬度数值的 GPS 文本文件
- **THEN** 系统 MUST 返回 `[lat, lon]` 浮点数组
- **AND** 系统 MUST 支持科学计数法格式的经纬度文本

#### Scenario: GPS 文件缺失
- **WHEN** 启用 GPS 的 dataset 遇到不存在的 GPS 路径
- **THEN** 系统 MUST 抛出包含缺失路径的异常

### Requirement: GPS-Rel-Polar 特征模式
系统 MUST 支持 `relative_polar` 作为默认 GPS 特征模式，并 MUST 支持 MMW YAML 专用的 `rsu_local_relative_polar` opt-in 模式。两个模式 MUST 对每个历史时隙输出 `[dist, sin_theta, cos_theta]` 三维特征；`relative_polar` MUST 保持既有世界坐标角语义，`rsu_local_relative_polar` MUST 使用 UE-BS 相对向量减去同一时隙 RSU pose yaw 后的局部角。系统 MUST 拒绝其它未声明 GPS 特征模式。

#### Scenario: 构造默认 GPS-Rel-Polar 特征
- **WHEN** `gps_feature_mode` 为 `relative_polar` 或 GPS 配置未显式设置特征模式
- **THEN** 系统 MUST 基于 UE-BS 世界坐标相对向量输出 `[dist, sin_theta, cos_theta]` 三维特征
- **AND** 系统 MUST 使用 `sin_theta` 和 `cos_theta` 表示角度，避免直接输出有跳变的角度值
- **AND** 输出 MUST 与本变更前的 `relative_polar` 数值兼容

#### Scenario: 构造 MMW RSU 局部相对极坐标特征
- **WHEN** `gps_feature_mode` 为 `rsu_local_relative_polar` 且 UE/BS 输入为包含 RSU pose yaw 的 MMW YAML 序列
- **THEN** 系统 MUST 从每个 `bs_gpsN` YAML 读取 `sensors.rsu_pose.rotation.yaw`
- **AND** 系统 MUST 计算 `theta_local = atan2(ue_y-bs_y, ue_x-bs_x) - yaw_rsu`
- **AND** 输出 MUST 为 `[dist, sin(theta_local), cos(theta_local)]` float32 数组且 shape 为 `[seq_len, 3]`

#### Scenario: 局部模式缺少权威 RSU yaw
- **WHEN** `rsu_local_relative_polar` 输入不是 MMW YAML，或任一 BS YAML 缺少有限的 `sensors.rsu_pose.rotation.yaw`
- **THEN** 系统 MUST 在数据加载阶段失败并报告具体资源路径
- **AND** 系统 MUST 不使用 camera、LiDAR、场景名常量或默认零角替代

#### Scenario: 静态 RSU 窗口内 yaw 不一致
- **WHEN** `rsu_local_relative_polar` 的同一历史窗口内 RSU yaw 超出固定数值容差
- **THEN** 系统 MUST 在数据加载阶段失败并报告相关 BS YAML 路径与 yaw 值
- **AND** 系统 MUST 不对不一致 yaw 求平均或只取最后一帧

#### Scenario: 拒绝非保留 GPS 特征模式
- **WHEN** 用户配置 `gps_feature_mode` 为 `raw`、`utm`、`relative`、`motion`、`motion_smooth` 或其它未声明值
- **THEN** 系统 MUST 拒绝启用该 GPS 配置
- **AND** 错误信息 MUST 列出 `relative_polar` 与 `rsu_local_relative_polar` 的适用范围

### Requirement: GPS 训练集 scaler
系统 MUST 只使用训练集 GPS 特征 fit scaler，并将同一 scaler 应用于验证或测试 split。系统 MUST 不使用验证集或测试集统计量参与 GPS 特征归一化。训练流程 MUST 将 fit 后的 GPS scaler 作为可复用工件保存，并在评估同一 checkpoint 时优先加载训练时保存的 scaler。

#### Scenario: 训练集 fit scaler
- **WHEN** dataloader 构建训练 split 且启用 GPS 归一化
- **THEN** 系统 MUST 使用训练 split 的 GPS 特征 fit scaler
- **AND** 系统 MUST 将 fit 后的 scaler 保存在训练 dataset 实例或可复用对象中

#### Scenario: 训练保存 scaler 工件
- **WHEN** 启用 GPS 归一化的训练流程完成 dataloader 构建
- **THEN** 系统 MUST 将训练集 fit 后的 GPS scaler 保存到当前运行目录的稳定工件路径
- **AND** 训练日志或 registry metadata MUST 记录该 scaler 工件路径

#### Scenario: 测试集复用训练 scaler
- **WHEN** dataloader 构建测试 split 且启用 GPS 归一化
- **THEN** 系统 MUST 使用训练 split已 fit 或从训练工件加载的 scaler 转换测试 GPS 特征
- **AND** 系统 MUST 不在测试 split 上重新 fit scaler

#### Scenario: 评估从 checkpoint metadata 加载 scaler
- **WHEN** 评估入口加载的 checkpoint metadata 或 registry sidecar 记录了 GPS scaler 路径
- **THEN** 系统 MUST 加载该 scaler 并传递给测试 dataset
- **AND** 系统 MUST 不为了 GPS scaler 重新扫描训练 split

#### Scenario: 缺少 scaler 的 GPS 评估
- **WHEN** 评估入口直接构建启用 GPS 归一化的测试 dataset 且没有可用训练 scaler、scaler 文件或 registry metadata
- **THEN** 系统 MUST 抛出清晰错误，提示需要提供训练集 scaler 或使用训练 dataloader 构建流程

### Requirement: GPS batch 字段
启用 GPS 的 Scenario 9 dataset MUST 在样本字典中返回 `gps` 字段。`gps` 字段 MUST 为浮点张量，形状为 `[seq_len, 3]`，并与 `input_beam` 的历史时隙对齐。

#### Scenario: dataset 返回 GPS 张量
- **WHEN** dataset 配置启用 GPS 且读取一个序列样本
- **THEN** 返回样本 MUST 包含 `gps`
- **AND** `gps` MUST 是 `torch.float32` 张量
- **AND** `gps` 的第一维长度 MUST 等于配置的 `seq_len`
- **AND** `gps` 的第二维长度 MUST 为 3

#### Scenario: dataset 不启用 GPS
- **WHEN** dataset 配置未启用 GPS
- **THEN** 返回样本 MAY 不包含 `gps`
- **AND** 训练、验证和评估流程 MUST 保持旧配置兼容

### Requirement: GPS 平滑窗口死配置移除
系统 MUST 不再将 `gps_smooth_window` 作为受支持的 GPS 配置能力暴露。GPS `relative_polar` 特征构造 MUST 不依赖该字段，默认配置、示例配置、公开文档和显式参数管线 MUST 移除该字段。

#### Scenario: 默认配置不暴露 gps_smooth_window
- **WHEN** 用户查看默认 GPS 或包含 GPS 的实验配置
- **THEN** 配置中 MUST 不包含 `gps_smooth_window`
- **AND** README 或实验说明 MUST 不把 `gps_smooth_window` 描述为可用 GPS 特征参数

#### Scenario: GPS 特征构造不接收平滑窗口参数
- **WHEN** 系统构造 `relative_polar` GPS 特征
- **THEN** `build_gps_features` 和 `load_gps_feature_sequence` 的公开调用路径 MUST 不要求或传递 `gps_smooth_window`
- **AND** 输出 GPS 特征 MUST 仍保持 `[seq_len, 3]` 的 `[dist, sin_theta, cos_theta]` 语义

#### Scenario: 历史配置字段不改变特征
- **WHEN** 历史外部配置仍包含 `gps_smooth_window`
- **THEN** 系统 MUST 忽略该遗留字段或抛出包含迁移说明的清晰错误
- **AND** 系统 MUST 不因为该字段改变 GPS `relative_polar` 特征值

### Requirement: MMW RSU yaw 与 scaler provenance
MMW RSU 局部 GPS 模式 MUST 记录 feature mode、yaw source 和训练集 scaler provenance。训练、验证和评估 MUST 使用相同 feature mode；scaler MUST 只由对应 mode 的训练 split 拟合并复用于 validation/test。

#### Scenario: 保存局部 GPS scaler
- **WHEN** 训练使用 `rsu_local_relative_polar` 且启用 GPS normalization
- **THEN** 保存的 scaler metadata MUST 标记 feature mode 为 `rsu_local_relative_polar`
- **AND** yaw source MUST 记录为 `bs_yaml:sensors.rsu_pose.rotation.yaw`
- **AND** scaler 数值工件 MUST 不包含 RSU yaw 或 validation/test 统计量

#### Scenario: 拒绝 world/local scaler 混用
- **WHEN** evaluation 配置使用 `rsu_local_relative_polar` 但加载的 scaler provenance 标记为 `relative_polar`
- **THEN** 系统 MUST fail closed 或要求重新提供匹配 mode 的训练 scaler
- **AND** 系统 MUST 不静默转换或复用不匹配 scaler

### Requirement: MMW pooled 局部 GPS preflight 与配对验证
MMW pooled all-weather 实验使用 `rsu_local_relative_polar` 时 MUST 在启动训练前验证全部 domain 的 BS GPS 列、引用 YAML 和静态 RSU yaw，并 MUST 使用与 world 坐标对照相同的数据、模型和训练评估协议。

#### Scenario: 15-domain 局部 GPS preflight
- **WHEN** all-weather launcher 请求 `rsu_local_relative_polar`
- **THEN** preflight MUST 要求每个 split 具有 `bs_gps1..bs_gps5`
- **AND** preflight MUST 逐引用验证 `sensors.rsu_pose.rotation.yaw` 为有限值且窗口内一致
- **AND** preflight 输出 MUST 记录每个 domain 的 yaw、feature mode、angle frame 和 yaw source
- **AND** 任一 domain 失败时 launcher MUST 不启动 GPU 训练

#### Scenario: world/local matched T2 对照
- **WHEN** 系统比较 `relative_polar` 与 `rsu_local_relative_polar` T2
- **THEN** 两个 resolved config 除 GPS mode、GPS input profile、output identity 和对应 provenance 外 MUST 保持一致
- **AND** 两个 run MUST 使用相同 15-domain inventory、H5/P1 split、seed、训练预算、domain-balanced sampler、missing augmentation 和 fixed-epoch `last.pth` policy
- **AND** missing evaluation MUST 复用相同 sample 与 mask cache identity
