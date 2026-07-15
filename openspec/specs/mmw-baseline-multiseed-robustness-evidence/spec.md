# mmw-baseline-multiseed-robustness-evidence Specification

## Purpose
定义 MMW T2 与本地基线的多随机种子公平训练、固定掩码聚合、逐样本严格配对和任务对齐鲁棒性证据契约。
## Requirements
### Requirement: 三方法多随机种子公平训练

系统 MUST 支持在同一 MMW 15-domain、四传感器、domain-balanced sampler、缺失增强、40 epoch和固定 `last.pth` 协议下运行 T2、AMBER-Full与RMBP-MM seeds1-3。seed MUST 控制模型训练、domain sampler和temporal missing随机性，但 MUST NOT 改变数据split或样本inventory。launcher MUST 显式记录 method、seed、GPU、config、log、command和状态，并在目标产物冲突时fail closed。

#### Scenario: GPU0-5并行补齐seed2/3

- **WHEN** 用户为三个方法请求seeds2/3并提供GPU0-5
- **THEN** launcher MUST 生成六个唯一作业并每卡最多启动一个训练进程
- **AND** 每个作业 MUST 写入对应 `<method>/seed<N>`，不得自动重命名为timestamp目录
- **AND** 既有seed1 config、checkpoint和log MUST 不被覆盖

#### Scenario: seed不改变验证样本

- **WHEN** launcher分别生成同一方法的seed1、seed2和seed3配置
- **THEN** 三者 MUST 具有相同15-domain inventory、split路径和固定数据seed
- **AND** experiment、domain sampler和temporal missing的行为seed MUST 分别为1、2和3

### Requirement: 多seed固定mask聚合评估

系统 MUST 对每个完成的method/seed使用对应固定epoch `last.pth`和共享mask cache，输出seed分层的per-domain、per-weather、per-scene、domain-macro、worst-domain与temporal rate指标。0-80% MUST 保持三mask-type type-equal协议；85/90/95% MUST 仅使用可精确表示的modality-frame cell masks并单独报告。

#### Scenario: seeds1-3使用相同mask identity

- **WHEN** evaluator处理三个方法的seeds1-3
- **THEN** 相同rate和mask MUST 具有相同mask digest、cache checksum和domain sample CSV checksum
- **AND** 任一config、checkpoint、domain或mask provenance缺失 MUST 使对应单元unavailable而不是静默聚合

### Requirement: 逐样本任务输出严格配对

系统 MUST 为clean及20/40/60/80%全部固定modality-frame masks保存逐样本target、float32 logits、argmax prediction、稳定sample id、domain、CSV checksum、rate、mask digest、cache checksum和checkpoint checksum。跨方法比较前 MUST 验证sample id、target、domain和mask identity完全一致。

#### Scenario: 样本顺序漂移被拒绝

- **WHEN** 两个方法的NPZ具有相同样本数但sample id、label或CSV checksum不同
- **THEN** summary MUST fail closed并指出首个不匹配字段
- **AND** 系统 MUST 不按数组位置继续生成配对图或差值

### Requirement: 共同clean正确样本保持率

T2与单个baseline的主配对诊断 MUST 使用两者在clean上都预测正确的固定样本集合；三方法同图 MUST 使用三者clean均正确的固定交集。集合 MUST 在所有missing rate和mask保持不变，并报告样本数、全样本覆盖率、各domain覆盖率及空domain。

#### Scenario: missing条件不重选共同样本

- **WHEN** 某共同clean正确样本在Drop80下预测错误
- **THEN** 该样本 MUST 保留在Drop80分母中
- **AND** summary MUST 不因missing结果重新筛选共同集合

### Requirement: 任务对齐鲁棒性证据

summary MUST 同时输出全样本绝对Top1、相对clean保持率、共同clean正确保持率、圆周Exact/Within1/Within3/MAE、同模型真类logit margin变化和归一化clean/missing JS距离。统计 MUST 先按domain和mask聚合，再对mask和15个domain等权；不同head的绝对logit尺度 MUST NOT 直接用于跨模型优劣排序。

#### Scenario: exact保持率与邻近错误排序不一致

- **WHEN** T2的共同样本Exact保持率高于AMBER但圆周MAE更差
- **THEN** 报告 MUST 同时展示两项事实并缩小结论范围
- **AND** 报告 MUST 不写成T2在所有鲁棒性指标全面优于baseline

### Requirement: 三seed结论与baseline范围

系统 MUST 输出逐seed值、三seed mean/std、T2-minus-AMBER和T2-minus-RMBP的配对差值及分组bootstrap 95%区间。T2优于某baseline的稳定性表述 MUST 要求至少2/3 seed主曲线AUC差值为正，且三seed平均clean、0-80 AUC和Drop80不低于该baseline。AMBER和RMBP结果 MUST 分别保留local-adaptation与out-of-paper-scope声明。

#### Scenario: 只通过部分门禁

- **WHEN** T2只在严重缺失或部分seed优于某baseline
- **THEN** decision MUST 标记为partial或unsupported并列出失败门禁
- **AND** 系统 MUST 不通过增加噪声、修改模型或选择性删除domain改变本change结论
