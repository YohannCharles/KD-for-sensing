## ADDED Requirements

### Requirement: 六类时间块 availability mask 可复现且无泄漏
系统 MUST 提供独立的时间块 mask generator，输出布尔 `availability_mask[B,M,T]`，其中 true 表示可用。系统 MUST 支持 full、sparse_easy、single_modality_burst2、single_modality_missing、latest_sync_missing 和 two_modality_recent_async，且 MUST 不硬编码 M=4 或 T=5。缺失块 MUST 在输入、pooling、evidence fusion、Router softmax 和所有归一化分母中被排除。

#### Scenario: 训练 curriculum
- **WHEN** quick-validation 在任一训练 epoch 按 sample identity 生成 mask
- **THEN** 前 10%、中间 20% 和后 70% MUST 分别使用冻结的三阶段概率
- **AND** 四个实验 MUST 使用相同 seed 和采样算法

#### Scenario: 确定性固定评测
- **WHEN** evaluator 对相同 global seed、sample id、mask type 和 variant 生成 mask
- **THEN** 所有实验 MUST 得到逐位相同的 mask
- **AND** S4 MUST 删除每个模态的最新时刻

#### Scenario: 复制帧分组
- **WHEN** generator 收到同一模态内重复的 source frame id
- **THEN** 任一副本被删除时同组副本 MUST 同时删除
- **AND** source identity 不存在时系统 MUST 不伪造分组

### Requirement: 每个时间块共享 64-beam prototype evidence
PCER MUST 复用现有 prototype bank，将融合前 block features 投影到共享 64-beam evidence space。当前 block 和 prototype 维度相同时 MUST 直接使用现有归一化 cosine head，不得复制独立 64 类分类器。

#### Scenario: MMW 五帧四模态 forward
- **WHEN** T2 处理 MMW `[B,5,4,64]` block features
- **THEN** PCER MUST 输出 `[B,20,64]` block evidence logits 和 `[B,20]` availability/weights
- **AND** prototype 参数的梯度策略 MUST 与现有有效 prototype head 一致

### Requirement: 完整视图监督缺失视图
PCER consistency MUST 使用同一 backbone 和 prototype bank 的完整输入输出作为 stop-gradient teacher，并以 temperature KL 监督缺失视图。系统 MUST 不创建 EMA teacher、外部 teacher checkpoint 或原始模态重建。

#### Scenario: full 样本不产生重复一致性损失
- **WHEN** student availability 与 base availability 完全相同
- **THEN** 该样本的 consistency loss MUST 为零
- **AND** 其他缺失样本 MUST 保持有限且可反向传播的 KL

### Requirement: Router 由向量化反事实贡献监督
完整 PCER Router MUST 对每个时间块输出一个 scalar logit，并只在可用块上归一化。target MUST 由 detached static/equal reference fusion 的 leave-one-block-out topology loss 差生成，不得由当前 Router 自监督，也不得为每个 block 重跑 backbone。

#### Scenario: 反事实 target 与 Router 梯度
- **WHEN** batch 至少保留两个时间块
- **THEN** 系统 MUST 从 `[B,N,64]` evidence 一次构造所有 leave-one-out logits
- **AND** target MUST 有限、缺失块质量为零、Router 参数梯度非零

### Requirement: 四组 quick-validation 输出可比较
系统 MUST 在相同 MMW 15-domain split、seed、backbone、batch、epoch、optimizer、scheduler、BPA、训练 mask 和固定测试 mask 下运行 A0--A3，并写出逐场景指标、Router 诊断、resolved config、PID/状态和统一报告。

#### Scenario: 完成四组 GPU 任务
- **WHEN** GPU4--7 通过占用预检并完成或失败
- **THEN** launcher MUST 保留每个任务的独立状态、日志和 checkpoint 路径
- **AND** 单个任务失败 MUST 不静默跳过其他任务

#### Scenario: 统一结果判断
- **WHEN** 四个最佳验证 checkpoint 完成 S0--S5 评测
- **THEN** 报告 MUST 分别给出 full、masked macro、hard average、worst case、逐 mask 和 Router 相关性
- **AND** MUST 按预注册三项条件判断是否值得继续
