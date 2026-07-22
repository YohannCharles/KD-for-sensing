## ADDED Requirements

### Requirement: PGCD 必须只消费四传感器与 beam-index topology 监督
系统 MUST 只消费 `image/radar/gps/lidar` 四个传感器、最优 beam index 和不依赖信道文件的 64-beam topology。配置或 batch 中出现 channel、CSI、path feature、channel gain target 或 beam-power utility tensor 时 MUST 在训练或评测前失败。

#### Scenario: 启动 PGCD 训练
- **WHEN** resolved config 启用 PGCD
- **THEN** `use_channel`、`use_csi`、`use_path_features` 和 `use_channel_gain_target` MUST 全部为 false
- **AND** dataset MUST 不打开 Channel_Data、CSI、path 或 beam-power 文件

#### Scenario: batch 含禁用 tensor
- **WHEN** PGCD batch 含 channel、CSI、path gain、beam gain vector 或 future beam power tensor
- **THEN** runtime MUST 抛出明确错误而不是静默忽略

### Requirement: 连续退化必须确定、单调且适配实际表示
`SensorDegradationGenerator` MUST 支持 L0 clean、L1 mild、L2 medium、L3 severe 和 L4 missing，并映射到 `0/0.25/0.5/0.75/1`。退化 MUST 在线生成、保持 tensor shape、按实际物理或归一化表示标定，并 MUST 不合成天气。

#### Scenario: 四传感器 seen corruption
- **WHEN** generator 对 image blur/occlusion、LiDAR BEV dropout、Radar detection-map dropout 或 GPS slow drift 施加 L0-L4
- **THEN** L0 MUST 与输入一致，L1-L3 强度 MUST 单调，L4 availability MUST 为 false
- **AND** padding、零占用和物理坐标语义 MUST 保持有效

#### Scenario: unseen corruption
- **WHEN** evaluator 选择 image exposure-noise、LiDAR coordinate jitter、Radar false clutter/coordinate jitter 或 GPS jump/white noise
- **THEN** corruption MUST 由 sample identity 与 variant 确定性生成
- **AND** MUST 不读取标签、天气或信道信息

#### Scenario: stale frame
- **WHEN** temporal corruption 选择 one-step stale
- **THEN** `t>0` MUST 只使用 `t-1`，不得使用未来帧
- **AND** stale mask MUST 记录被替换时间块

#### Scenario: 复制源帧
- **WHEN** source frame ids 表明多个时间块源自同一真实帧
- **THEN** 这些副本 MUST 共享同一 corruption 随机参数

### Requirement: 训练退化采样必须与标签和天气输入解耦
训练 MUST 保留 clean view 并按预注册概率生成最多两个传感器的 corrupted view，severity 与 corruption type MUST 不依赖 beam label。weather MAY 用于近似均衡采样和分层审计，但 MUST 不作为新增 estimator 输入。

#### Scenario: 重放训练样本
- **WHEN** seed、epoch、sample identity 和采样配置相同
- **THEN** corruption 模式、传感器、severity 和随机参数 MUST 相同
- **AND** C0-C7 MUST 使用相同 corruption 随机序列

### Requirement: 质量 target 必须来自 clean-corrupted beam prototype evidence
系统 MUST 复用同一 encoder、projection、prototype bank 与 active topology，输出 clean/corrupted `[B,N,64]` evidence。clean 分支 MUST stop-gradient，且系统 MUST 不创建独立或 EMA teacher。

#### Scenario: 计算 topology drift
- **WHEN** clean 与 corrupted prototype 分布可用
- **THEN** 系统 MUST 记录 expected topology transport 并构造 self-debiased 非负 drift target
- **AND** 相同分布的 drift MUST 近似为零，远距离 beam 漂移 MUST 大于邻近漂移

#### Scenario: 计算 task degradation
- **WHEN** 训练 batch 含最优 beam index
- **THEN** 系统 MUST 由每块 corrupted topology loss 减去 detached clean loss 构造 raw 与 clipped task degradation
- **AND** target 归一化 MUST 只使用当前训练 batch 或训练集统计

### Requirement: 质量估计器不得读取 teacher 或注入元数据
质量估计器 MUST 只读取 corrupted block feature、corrupted prototype logits/统计、modality/time embedding 和 availability。它 MUST 不读取 clean feature/logits、真实 beam 标签、severity、corruption type、weather 或任何 channel/path 信息。

#### Scenario: 推理质量
- **WHEN** 模型仅收到 corrupted deployment view
- **THEN** 模型 MUST 输出 `[B,N]` predicted degradation 和 reliability
- **AND** missing reliability MUST 为零

### Requirement: 可靠性融合必须锚定 learned block prior
PGCD MUST 持有 learned `prior_logits[N]`，并以非负 beta 将 predicted reliability 调制到 prior logits。masked weights MUST 对 missing 为零且在可用块上和为一。

#### Scenario: 动态输出被关闭
- **WHEN** C0 或 D3 令所有可用 reliability 为一
- **THEN** 融合 MUST 精确回退为 availability-normalized learned prior

#### Scenario: 动态输出启用
- **WHEN** predicted degradation 增大
- **THEN** predicted reliability MUST 单调下降
- **AND** `beta_reliability` MUST 由 softplus 参数化为非负

### Requirement: C0-C7 必须是固定可比较矩阵
系统 MUST 提供 C0 corruption+prior、C1 severity、C2 entropy/confidence、C3 prototype regression、C4 prototype ranking、C5 task degradation、C6 combined quality 和 C7 full PGCD，且 MUST 共享 initialization、split、seed、corruption、epoch、effective batch、optimizer、scheduler、BPA/topology 和 checkpoint 规则。

#### Scenario: 生成八任务
- **WHEN** launcher prepare C0-C7
- **THEN** GPU0-7 映射、resolved config、manifest、PID/status 和 claim-ineligible 标记 MUST 完整
- **AND** 任一任务失败 MUST 被记录且不得导致其他任务被静默跳过

### Requirement: 评测必须验证动态真实性与泛化
系统 MUST 评测 clean、seen severity、two-sensor mixed、stale-frame、unseen corruption 和历史 missing，并 MUST 只输出 Top-1/3/5、Within-3、beam-index MAE 等无需信道矩阵的指标。D1 global mean MUST 只由 train 或 train-design portion 统计。

#### Scenario: 执行 D0-D3 替换
- **WHEN** C1-C7 完成评测
- **THEN** 报告 MUST 对主要 corruption family 比较 dynamic、global mean、sensor+severity mean 和 prior-only
- **AND** D0 未超过 D1 时 MUST 不宣称样本级动态质量有效

#### Scenario: 自然天气分析
- **WHEN** 汇总 sunny/rainy/foggy
- **THEN** 系统 MUST 输出分组 prototype/quality/weight/beam 指标
- **AND** 只有位置、轨迹、时间和标签严格匹配时才可进行 paired weather 分析

### Requirement: 快筛必须停在预注册边界
系统 MUST 将本轮标记为 single-seed、inner/development、claim-ineligible，并按 3/4 quick gates 判定 C7。系统 MUST 不自动启动 multi-seed、outer test 或下一轮实验。

#### Scenario: 汇总完成
- **WHEN** C0-C7 结果与诊断已生成
- **THEN** 报告 MUST 如实回答动态增益、质量相关性、单调性、unseen 泛化、梯度冲突和 gate 状态
- **AND** 动态不超过 global mean 时 MUST 判定该动态路线失败
