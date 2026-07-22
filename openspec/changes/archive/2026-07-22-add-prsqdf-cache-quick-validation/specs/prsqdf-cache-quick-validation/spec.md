## ADDED Requirements

### Requirement: PR-SQDF 必须保持四传感器和无信道监督边界
系统 MUST 只消费 MMW image、radar、gps、lidar、64 类 beam index和已声明的 beam topology，并 MUST 在配置、batch、cache schema 或源码入口出现 channel、CSI、channel gain、beam power、ray-tracing path 或派生字段时 fail closed。

#### Scenario: 缓存前审计输入
- **WHEN** 预计算器加载 C0 config 和一个真实 batch
- **THEN** 它 MUST 验证四模态输入、硬 beam label和 topology provenance
- **AND** 任一信道或 path 字段 MUST 阻止缓存生成

### Requirement: 共享缓存必须冻结 C0 并可复现原推理
系统 MUST 从 C0 validation-best checkpoint提取 pooled pre-prototype feature、prototype feature、block logits、global prior、availability、output/sensor statistics和离线 target，且 MUST 冻结所有 C0 参数。clean view MUST 每个 sample只保存一次，corrupted view MUST 通过稳定 identity引用 clean view。

#### Scenario: cache inference preflight
- **WHEN** 固定 validation 样本完成 float16 cache重构
- **THEN** cache fused logits与原 C0 logits MUST 在声明浮点阈值内一致
- **AND** Top1 MUST 完全一致或只出现报告阈值内的极小差异
- **AND** 超阈值时正式 Q0--Q5训练 MUST 不启动

### Requirement: 分片缓存必须完整、紧凑且确定
预计算器 MUST 在正式运行前以 100 个样本估算单样本字节数和全量占用，并 MUST 将 feature保存为 float16/bfloat16、risk保存为 float32、mask保存为 bool、label/index保存为整数。GPU0--5 shard MUST 按稳定 sample identity互斥划分，并在合并时检查重复、遗漏、shape、dtype、pairing和 corruption determinism。

#### Scenario: 六个 shard 合并
- **WHEN** 六个预计算 worker均退出
- **THEN** merge MUST 证明 sample id无重复且覆盖冻结 manifest
- **AND** 相同 corruption identity MUST 重现相同结果，不同 sample id不得全部相同

### Requirement: 风险 target 必须由 clean-corrupted beam 决策差构造
系统 MUST 分别计算 injected severity、hard CE risk、topology-aware block-loss risk和debiased prototype transport drift。CE/topology risk MUST 截断负增量为零；topology risk MUST 令远距离 beam偏移比相邻偏移承担更大代价；所有 target MUST detach。

#### Scenario: synthetic 风险顺序
- **WHEN** clean logits不变、偏移到相邻 beam和偏移到远距离 beam分别作为 corrupted logits
- **THEN** 不变 view的 CE/topology risk MUST 为零
- **AND** 远距离偏移的 topology risk MUST 大于相邻偏移

### Requirement: 风险归一化只能拟合 train cache
系统 MUST 分别从 train cache拟合 risk quantile clip、median、IQR和normalized q99，并 MUST 由 validation/eval复用。系统 MUST 不按 test corruption、weather或severity重新拟合统计。

#### Scenario: validation 加载风险统计
- **WHEN** Q1--Q5读取 validation/eval cache
- **THEN** 它们 MUST 加载同一个带 train provenance的 normalization文件
- **AND** 缺少或身份不一致时 MUST fail closed

### Requirement: Quality head 不得读取训练期特权字段
推理期 quality head MUST 只读取 corrupted pre-prototype feature、corrupted output statistics、可选 corrupted sensor statistics、modality/time embedding和availability。clean feature/logits、label、severity、corruption type、weather、target risk均 MUST 不进入 forward参数或输入 tensor。

#### Scenario: Q5 forward schema
- **WHEN** Q5执行预测
- **THEN** 输入 schema MUST 只含部署期 corrupted字段
- **AND** 删除全部 target和metadata后 forward结果 MUST 不变

### Requirement: 有界 prior correction 只能降低高风险块权重
系统 MUST 使用 `beta_m=beta_max*sigmoid(raw_beta_m)` 和 `prior_logit-beta_m*clipped_risk` 形成 fusion logits，再执行 masked softmax。beta MUST 非负且不超过 beta_max；missing块权重 MUST 为零；可用权重和 MUST 为一；risk为零时 MUST 精确回退 global prior。

#### Scenario: 风险单调性
- **WHEN** 仅增加一个可用 block的 predicted risk
- **THEN** 该 block权重 MUST 不增加
- **AND** 其他条件相同且所有可用 risk相等时相对 prior MUST 不变
- **AND** NaN/Inf risk MUST 触发错误

### Requirement: Q0--Q5 必须共享固定训练身份
系统 MUST 固定 Q0 prior-only、Q1 outputstats+topology、Q2 preproto+severity、Q3 preproto+CE、Q4 preproto+topology、Q5 preproto+sensorstats+topology+ranking，并 MUST 让 Q1--Q5共享 cache、split、seed、batch order、optimizer、scheduler、epoch、early stopping和validation-best规则。只有 quality adapter/head、bounded beta和极小 normalization参数 MAY 训练。

#### Scenario: 比较五个质量方向
- **WHEN** launcher生成 Q0--Q5 resolved config
- **THEN** config差异 MUST 只来自预注册的input/target/ranking字段
- **AND** semantic backbone参数 MUST 不出现在 optimizer或训练 checkpoint中

#### Scenario: 训练更新预算审计
- **WHEN** Q1--Q5 从共享 cache 启动纠正运行
- **THEN** resolved config MUST 记录 base sample、有效 train view、batch size、steps per epoch和最低 epoch
- **AND** early stopping MUST 在最低 epoch 完成前保持禁用
- **AND** metrics MUST 记录实际 optimizer step和samples seen

### Requirement: 固定评测必须验证动态真实性和泛化
系统 MUST 用同一最佳 checkpoint评测 E0 clean、E1 seen、E2 mixed、E3 stale、E4 unseen、E5 S0--S5 missing和E6 weather分层，并对 Q1--Q5执行 D0 Dynamic、D1 train-fit Global Mean、D2 train-fit Sensor-Severity Mean和D3 Prior Only。报告 MUST 包含任务指标、每传感器/severity/weather、quality相关性/单调性、梯度对齐、效率和success gates。

#### Scenario: 判断样本级动态价值
- **WHEN** comparison报告完成
- **THEN** 它 MUST 显式计算 D0-D1和D0-D3在clean/severe/mixed/unseen/missing上的差值
- **AND** 只有 D0稳定超过D1时才可判断样本级动态质量有价值

### Requirement: 快速验证完成后必须停止
PR-SQDF运行 MUST 标记为 single-seed、inner/development、claim-ineligible，并 MUST 在 Q0--Q5汇总及唯一方向建议后停止。系统 MUST 不自动启动 outer test、multi-seed、下一轮完整训练或修改正式 claim。

#### Scenario: success gates 完成
- **WHEN** Q4/Q5 gates和最终方向已写入 comparison报告
- **THEN** launcher MUST 结束
- **AND** 后续实验只能作为建议记录而不得执行
