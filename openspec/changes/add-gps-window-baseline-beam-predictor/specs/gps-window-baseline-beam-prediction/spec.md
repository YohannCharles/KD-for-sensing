## ADDED Requirements

### Requirement: GPS 滑动窗口 baseline 输入边界
系统 MUST 提供一个仅使用 GPS/pose 历史滑动窗口的非神经网络 beam prediction baseline。该 baseline MUST 不构建 torch neural network、不训练参数、不加载 checkpoint，并 MUST 只消费样本预测时刻之前可观测的 GPS/pose、RSU pose、时间戳和允许的 split metadata。

#### Scenario: baseline 不使用神经网络
- **WHEN** 用户运行 GPS window baseline CLI 或配置
- **THEN** 系统 MUST 不调用训练器、不执行 backward、不创建 optimizer
- **AND** run metadata MUST 记录 `uses_neural_network=false`
- **AND** run metadata MUST 记录 `uses_checkpoint=false`

#### Scenario: 预测只使用历史 GPS/pose
- **WHEN** baseline 为 target_test 样本生成预测
- **THEN** 预测输入 MUST 只包含该样本预测时刻之前的 GPS/pose 历史窗口、RSU pose、时间戳和 codebook 配置
- **AND** 系统 MUST NOT 使用 future beam label、future beam_power argmax、target_test label、target_test path/radio/channel fields 或任何 target oracle 字段生成预测
- **AND** metadata MUST 记录实际使用字段列表

#### Scenario: GPS 字段缺失时可诊断失败或回退
- **WHEN** 样本缺少足够 GPS/pose 历史窗口
- **THEN** baseline MUST 按配置选择跳过样本或使用允许的 fallback predictor
- **AND** 输出 artifact MUST 记录缺失原因、受影响样本数和 fallback 使用次数

### Requirement: GPS 几何滤波与 beam score 输出
系统 MUST 支持可配置的 GPS 几何滤波算法族，并 MUST 为每个样本和预测 horizon 输出 beam score/logits，形状与现有 HiST-Beam 评估兼容。

#### Scenario: last geometry 预测
- **WHEN** 用户选择 `algorithm=geometry_last`
- **THEN** baseline MUST 使用最后一个历史 GPS/pose 与 RSU pose 计算相对方位
- **AND** baseline MUST 将相对方位映射到合法 beam id
- **AND** baseline MUST 输出 `[N, H, C]` beam score，其中 `C` 为 beam class 数

#### Scenario: constant velocity 外推
- **WHEN** 用户选择 `algorithm=constant_velocity`
- **THEN** baseline MUST 从 GPS 滑动窗口估计 CAV 平面速度或局部坐标速度
- **AND** baseline MUST 按预测 horizon 外推未来位置
- **AND** baseline MUST 将外推位置对应的相对方位映射为 beam score

#### Scenario: 环形角度平滑
- **WHEN** 用户启用 angle smoothing 或 moving-window smoothing
- **THEN** baseline MUST 使用环形角度语义计算均值、速度或残差
- **AND** baseline MUST NOT 直接对角度度数做会破坏 0/360 边界的普通线性平均

#### Scenario: Top-K 邻域扩展
- **WHEN** baseline 产生单个几何 beam 中心
- **THEN** 系统 MUST 支持按环形 beam 距离扩展邻域 score
- **AND** Top-1、Top-3 和 Top-5 MUST 可通过同一 score tensor 计算

### Requirement: 全场景 GPS baseline 评估
系统 MUST 提供 CLI 或等价入口，用于在所有本地可用 MMW 场景或指定 source-target 矩阵上运行 GPS window baseline，并输出与现有 beam prediction 指标兼容的评估产物。

#### Scenario: plan-only 输出运行矩阵
- **WHEN** 用户运行 GPS window baseline CLI 且未指定执行
- **THEN** 系统 MUST 写出包含场景、split、算法、参数网格、输出目录和 claim scope 的 plan artifact
- **AND** 系统 MUST 不启动训练或评估计算

#### Scenario: execute 输出 metrics
- **WHEN** 用户执行 GPS window baseline run
- **THEN** 系统 MUST 为每个场景或 fold 写出 `metrics.json`
- **AND** metrics MUST 至少包含 Top-1、Top-3、Top-5、coarse accuracy、fine offset accuracy、样本数和算法参数
- **AND** 若 beam power vector 可用，metrics MUST 包含 normalized received power 和 beam power loss dB 或清晰不可用原因

#### Scenario: 所有场景覆盖摘要
- **WHEN** 用户选择 all-scenes profile
- **THEN** 系统 MUST 枚举本地 MMW availability 中所有 ready scenario
- **AND** summary MUST 记录每个 scenario 的样本数、可用 GPS 覆盖率、指标和失败/跳过原因
- **AND** summary MUST 不把单 town/sunny 结果声明为 town-level 或 weather-level 泛化

### Requirement: 参数搜索与逐轮调参记录
系统 MUST 支持 deterministic 参数搜索和迭代调参记录，使用户能根据每次结果调整 GPS 滑动窗口算法并复现实验路径。

#### Scenario: 参数搜索只使用 calibration split
- **WHEN** 用户启用 sweep 或 calibration
- **THEN** 参数排序 MUST 只使用 source split 或 target_adapt labeled support split
- **AND** target_test 指标 MUST NOT 参与参数排序、早停、候选筛选或推荐生成
- **AND** metadata MUST 记录 calibration split、样本数和 `used_target_test_for_calibration=false`

#### Scenario: 记录每轮调参结果
- **WHEN** 一轮 GPS baseline sweep 完成
- **THEN** 系统 MUST 写出 iteration report，包含算法参数、calibration metrics、final eval metrics、误差分桶、预测直方图和 run id
- **AND** report MUST 能追加到同一实验目录的调参历史中

#### Scenario: 推荐下一轮候选
- **WHEN** sweep 结果中存在可比较参数组合
- **THEN** 系统 MUST 输出 deterministic next-candidate summary
- **AND** summary MUST 说明候选来自 calibration 表现、误差分桶或 beam offset 诊断
- **AND** summary MUST 不声称 target_test 上的最优参数可作为无偏主结论

### Requirement: GPS baseline 防泄漏与可审计产物
GPS window baseline MUST 输出足够 metadata 证明其输入边界、split eligibility 和 target oracle 使用情况，并 MUST 支持后续 summary 将其纳入或排除主结论。

#### Scenario: run metadata 记录 oracle 使用
- **WHEN** baseline run 完成
- **THEN** run metadata MUST 包含 `used_target_oracle_fields`
- **AND** 在合法 GPS-only 预测中该字段 MUST 为空列表
- **AND** 若任何禁用字段被读取，run MUST 标记为主结论不合格并记录机器可读 reason

#### Scenario: prediction artifact 可复查
- **WHEN** baseline 写出预测 artifact
- **THEN** artifact MUST 包含 sample id、scenario、split、true beam、top-k predicted beams、score 摘要、GPS coverage status 和 fallback status
- **AND** artifact MUST 不复制大型原始传感器数据或 channel/path tensor

#### Scenario: 与诊断 baseline 同表比较
- **WHEN** summary 读取 GPS window baseline 结果
- **THEN** summary MUST 能同时展示 majority baseline、last-beam baseline、transition fallback 或其不可用原因
- **AND** summary MUST 标明 GPS window baseline 是否超过这些非神经网络参照
