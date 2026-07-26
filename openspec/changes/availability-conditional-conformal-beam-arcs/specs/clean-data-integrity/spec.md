## ADDED Requirements

### Requirement: 标定集必须取自 inner validation 而非 train
Conformal 标定 MUST 只使用 `mmw_full_pool_development_v1` 的 inner validation，并在构建任何阈值前重新验证 protocol 与隔离审计。系统 MUST 不使用 Full-pool train 的分数进行标定，因为冻结 U0 在其训练数据上的分数带乐观偏置，会破坏有限样本覆盖保证。`outer_test_accessed` MUST 保持 false。

#### Scenario: 协议或审计无效
- **WHEN** protocol、审计报告或 checkpoint SHA256 缺失、不匹配，或 outer test 被启用
- **THEN** 工作流 MUST 在读取任何样本前失败

#### Scenario: 尝试用 train 分数标定
- **WHEN** 标定输入包含任一 Full-pool train 样本
- **THEN** 工作流 MUST fail closed

### Requirement: 标定与测试按 `(domain, cav)` 轨迹整块切分
主结果口径 MUST 以 `(domain, cav)` 轨迹为不可分单元切分标定与测试；同一轨迹的帧 MUST 不同时出现在两侧。切分 MUST 由预注册种子确定性生成，且 MUST 按样本数而非轨迹数逼近目标比例。

#### Scenario: 轨迹跨越切分边界
- **WHEN** 任一 `(domain, cav)` 轨迹的帧同时出现在标定与测试侧
- **THEN** 工作流 MUST fail closed

#### Scenario: 可用轨迹不足
- **WHEN** 可用轨迹少于两条，或标定比例不在开区间 (0, 1) 内
- **THEN** 切分 MUST 拒绝执行

### Requirement: frame 级随机切分只作为可交换性对照
系统 MUST 同时报告 frame 级随机切分的结果，并 MUST 将其标记为**对照**而非备选协议——它故意让相邻帧落在两侧以恢复可交换性。任一结果表 MUST 同时给出两种切分；随机切分的覆盖数字 MUST 不被单独引用为主结果。

#### Scenario: 只输出单一切分
- **WHEN** 运行器只产出轨迹切分或只产出随机切分的结果
- **THEN** 该运行 MUST 判定为不完整，MUST 不用于门槛判定

#### Scenario: 两种切分给出不同结论
- **WHEN** 条件覆盖跨度在随机切分下达标而在轨迹切分下不达标
- **THEN** 结论 MUST 按轨迹切分判定，并 MUST 显式记录该差值作为泄漏放大的证据

### Requirement: 唯一可拟合对象只由标定侧轨迹拟合
C4 局部化阈值函数是本筛选中唯一可拟合的状态。它 MUST 只读取标定侧轨迹的分数、mask 与原型标量；测试侧轨迹的分数、标签与协变量 MUST 在拟合期完全不可见。C0--C3 与 C5 MUST 不引入任何可拟合状态。

#### Scenario: 评估期间状态不可变
- **WHEN** 任一 arm 对测试侧轨迹评估
- **THEN** 冻结 U0、prototype、阈值与 C4 阈值函数的全部状态 MUST 保持不变

#### Scenario: 门槛判定后回头调参
- **WHEN** 任一 arm 未通过预注册门槛
- **THEN** 系统 MUST 不调整 alpha、稳健化参数、种子数、切分粒度或分层定义以挽救该 arm
- **AND** MUST 直接产出负结果结论，MUST 不访问 outer test

### Requirement: 区间估计按轨迹重抽而非按帧
所有不确定性区间 MUST 以 `(domain, cav)` 轨迹为重抽单元。按帧重抽 MUST 不用于任何区间或门槛判定，因为同一轨迹内的帧强相关会低估方差。

#### Scenario: 报告任一覆盖率或弧长差值的区间
- **WHEN** 结果表给出跨 arm 的差值区间
- **THEN** 重抽单元 MUST 记录为轨迹，且重抽次数与种子 MUST 在计算前固定并写入产物
