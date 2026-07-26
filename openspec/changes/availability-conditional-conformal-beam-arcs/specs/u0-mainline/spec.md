## ADDED Requirements

### Requirement: 集合值输出不得改变冻结 U0 的任何前向数值
Conformal 波束候选路线 MUST 作为冻结 U0 之上的纯后处理实现。它 MUST 不更新 encoder、projection、fusion、`BeamPrototypeBank`、classifier、router 或归一化统计，MUST 不创建任何指向 U0 的梯度路径，且 MUST 在运行审计中记录 `trained=false`。加载的 checkpoint SHA256 与 U0 结构不符时 MUST 在读取数据前失败。

#### Scenario: 后处理前后的 U0 输出一致
- **WHEN** 同一批样本在同一 mask 下分别经由冻结 U0 直接前向与 conformal 运行器的表征重放计算
- **THEN** 融合 logits MUST 逐样本一致
- **AND** U0 全部参数的 `requires_grad` MUST 为 false，且 optimizer MUST 不被创建

### Requirement: Nonconformity 分数取自原型余弦 logits
分数 MUST 定义为 `1 - softmax(fused_logits)`，其中 `fused_logits` 是 U0 对 `BeamPrototypeBank` 的余弦 logits。系统 MUST 不引入额外置信度头、温度标定或任何新的可拟合打分状态。

#### Scenario: 分数与原型几何绑定
- **WHEN** 运行器计算任一样本的 nonconformity 分数
- **THEN** 该分数 MUST 仅由冻结 prototype 与冻结融合特征决定，不得读取标签或测试侧统计量

### Requirement: 阈值按 15 种可用性模式分层且未见分层回退有限
系统 MUST 支持按 15 种规范 mask 分层的 Mondrian 阈值，每层使用 `ceil((n + 1) * (1 - alpha))` 的有限样本秩而非朴素经验分位。标定集中不存在的分层 MUST 回退到粗一级的**有限**阈值并被逐样本标记，回退比例 MUST 与覆盖率、集合大小一同报告。

#### Scenario: 某分层没有标定数据
- **WHEN** 某个分层在标定侧样本数为零
- **THEN** 该分层的样本 MUST 取得有限回退阈值，MUST 被标记为未见，且回退比例 MUST 出现在结果表中
- **AND** 系统 MUST 不以无穷阈值代替回退

#### Scenario: 秩超过标定样本数
- **WHEN** 某分层的 `ceil((n + 1) * (1 - alpha))` 超过该层标定样本数
- **THEN** 系统 MUST 返回可见的退化答案（保留全部波束），MUST 不静默降级为朴素分位

### Requirement: 环形弧闭包绑定审计拓扑且为集合超集
弧闭包 MUST 使用 `topology_id=ula_dft_phase_cycle_v1`、64 波束 / 64 天线且 15 个 domain metadata 一致的已审计 manifest 定义环形顺序；拓扑校验不通过时 MUST fail closed。输出弧 MUST 是对应 conformal 集合的超集。

#### Scenario: 弧覆盖率不低于集合覆盖率
- **WHEN** 在任一 mask、任一 alpha 下同时评估集合与其弧闭包
- **THEN** 弧覆盖率 MUST 不低于集合覆盖率
- **AND** 弧长 MUST 不小于集合大小，且弧长比 MUST 随结果报告

#### Scenario: 集合跨越码本首尾
- **WHEN** conformal 集合同时包含环形顺序上的首尾波束
- **THEN** 最短弧 MUST 允许回绕，MUST 不退化为覆盖整个码本

### Requirement: 漂移稳健路线与等容量置换负对照
筛选 MUST 包含 C0 边际、C1 mask 分层、C2 leave-one-trajectory-out cross-conformal、C3 轨迹级分布稳健膨胀、C4 原型空间局部化阈值函数与 C5 负对照。C5 MUST 与 C3/C4 具有严格相等的估计器容量，且 MUST 只置换分层标签而不置换分数。

#### Scenario: 负对照保留非条件输入
- **WHEN** C5 在固定样本上评估
- **THEN** nonconformity 分数、mask 与冻结 U0 输出 MUST 与 C3/C4 相同，仅标定侧分层标签来自确定性置换

#### Scenario: 负对照同时通过有效性与条件性门槛
- **WHEN** C5 同时满足 G1 与 G2
- **THEN** 该路线 MUST 判定为增益来自阈值膨胀而非可用性条件化
- **AND** 系统 MUST 直接产出负结果结论，MUST 不调整 alpha、稳健化参数、种子数或切分粒度
