## ADDED Requirements

### Requirement: 系统必须从完整候选窗口重建真实 trajectory groups

系统 MUST 从 15 个 Town3 domain 的 canonical `all_sequences.csv` 读取全部候选窗口并生成来源清单。trajectory unit MUST 优先使用显式 run/episode 元数据；缺少显式元数据时 MUST 通过共享 Radar、BS-GPS、target/dependency frame、完整 CSV row和重叠场景执行资源图的 connected components 重建，且 MUST NOT 仅按 CAV 拆分。

#### Scenario: 多个 CAV 共享 RSU 资源

- **WHEN** 两个 CAV 基础段引用同一个 RSU Radar 或 BS-GPS 资源
- **THEN** 系统 MUST 将它们归入同一个 trajectory group
- **AND** 该 group MUST 作为不可拆分单元进入一个 split

#### Scenario: trajectory 元数据或资源异常

- **WHEN** 窗口跨 group、标签越界、资源缺失或场景执行冲突
- **THEN** 系统 MUST 输出具体异常和处理方式
- **AND** MUST NOT 自动合并无关场景或静默删除大量窗口

### Requirement: 系统必须确定性生成 group-level 80/10/10

系统 MUST 使用 seed 2026 在完整 trajectory group 层生成 train/validation/test。恰好 50 组时 MUST 严格生成 40/5/5；其他组数 MUST 使用最接近 80/10/10 的整数数量并保证 validation/test 至少各一组。系统 MUST 尽量平衡 weather、domain、scenario 和窗口数，但 MUST NOT 拆分 group、随机拆窗口、执行 chronological tail split 或依据模型结果更换 seed。

#### Scenario: group 数不是 50

- **WHEN** 有效 trajectory group 数不是 50
- **THEN** 系统 MUST 报告实际 train/validation/test group 数和确定性分配过程
- **AND** 数量之和 MUST 等于有效 group 总数

#### Scenario: 分层覆盖不可满足

- **WHEN** split group 数不足以覆盖全部 weather 或 scenario 类别
- **THEN** 系统 MUST 记录不可满足的约束
- **AND** MUST 保持 group 完整和固定 seed

### Requirement: 所有 split 必须通过资源级零交集审计

系统 MUST 对 train/validation、train/test、validation/test 两两审计 sample identity、target identity、CSV row、dependency frames、Camera、LiDAR、Radar、GPS、channel 审计资源、trajectory ID、trajectory group ID 和 scenario execution ID。任一关键交集非零时协议构建 MUST 失败且训练 MUST NOT 启动。channel 只可用于泄漏审计，MUST NOT 成为模型输入。

#### Scenario: 共享依赖或传感器资源

- **WHEN** 任意 split 对共享必须隔离的 identity 或资源
- **THEN** split audit MUST 失败
- **AND** loader、optimizer 和 checkpoint MUST NOT 创建

### Requirement: 协议产物必须完整且可复核

系统 MUST 在本地协议目录生成 source inventory、trajectory groups、group audit/anomalies、JSON/YAML manifest、split audit、历史暴露、group/sample ids、SHA256、beam/weather/domain 统计、协议比较和摘要。manifest MUST 记录协议名、版本、seed、源 CSV、trajectory 定义、资源耦合规则、group/window 数、各 split group ids、claim eligibility、legacy protocol 使用与 outer test 访问状态。

#### Scenario: 重建同一协议

- **WHEN** 输入 CSV 内容和 seed 不变
- **THEN** group ids、split sample ids 与 split SHA256 MUST 保持一致
- **AND** manifest MUST 精确绑定全部生成 CSV 的 hash

### Requirement: 历史暴露必须限制论文 claim

系统 MUST 扫描可用历史 manifest，分别统计新 split 样本的历史 train、validation、test/outer 暴露与未暴露状态。任一新 test 样本曾进入历史训练或方法选择时，系统 MUST 设置 `claim_eligible=false`，并 MUST NOT 将该 test 描述为最终无偏论文测试集。

#### Scenario: 新 test 曾被历史开发访问

- **WHEN** 历史身份恢复发现新 test 样本出现在 train 或方法选择 validation
- **THEN** manifest 与摘要 MUST 同时声明 `claim_eligible=false`

### Requirement: test 必须默认封存

普通训练 MUST 只加载 trajectory train 和 validation。test evaluation MUST 要求显式 `--allow-test-evaluation`，默认 MUST 为 false；本变更的 prepare、train、monitor 和 aggregate MUST NOT 执行 test 推理或读取 test 预测结果。

#### Scenario: 普通训练配置包含 test

- **WHEN** 未显式授权的配置或 loader 请求 test CSV
- **THEN** 系统 MUST 在构建 test Dataset 前失败

### Requirement: M0--M4 必须公平共享协议与训练预算

M0 线性 head、M1 普通 prototype、M2 topology prototype、M3 topology prototype 加 random-balanced 单模态训练，以及 M4 availability-balanced topology consistency MUST 使用相同 train/validation group、sample ids、normalization、encoder、temporal processing、fusion、batch size、optimizer、epochs、seed、optimizer steps 和 validation-loss checkpoint selection。所有 normalization MUST 仅从新 train split 拟合。

#### Scenario: 五个方法进入训练

- **WHEN** M0--M4 的 preflight 完成
- **THEN** 五个 resolved config MUST 记录同一 split manifest SHA256
- **AND** 任何 normalization identity MUST 等于 train split identity

#### Scenario: M4 构造 availability-balanced paired views

- **WHEN** M4 处理一个 train batch
- **THEN** encoder MUST 只计算一次并产生 full 与 masked 两个 fusion view
- **AND** masked view 的 1/2/3 可用模态等级及等级内组合 MUST 仅由 train sample identity、epoch 和固定 seed 确定性均衡
- **AND** topology consistency MUST 使用 detached full teacher 与审计 beam topology distance

#### Scenario: M4 执行因果消融

- **WHEN** 比较 M4-a、M4-b、M4-c 与 M4
- **THEN** 四者 MUST 仅依次改变 mask 均衡与 consistency 类型
- **AND** 四者 MUST 保持 paired supervision、单卡 batch size、optimizer steps、seed、split 与 checkpoint selection 相同

#### Scenario: 最小训练 smoke

- **WHEN** 每个方法执行两个 train steps 并保存 checkpoint 与 metrics
- **THEN** loss MUST 为有限值
- **AND** checkpoint MUST 可加载且 metrics MUST 可写入
