## ADDED Requirements

### Requirement: F1 token cache 必须保持冻结基线身份与 parity
系统 MUST 从同一 F1 feature concat MLP validation-best checkpoint 和已验证 feature-fusion cache 派生三 split、六分片 token cache，记录 checkpoint/config/source cache SHA、样本身份、分组元数据、`[20,256]` F1 token、64-beam full logits、modality/time id 与 split provenance。系统 MUST NOT 缓存或消费 channel、CSI、gain、beam power、ray/path、历史 beam 或 label/weather/scene 派生模型输入。

#### Scenario: 合并 token cache
- **WHEN** 六个 shard 完成并执行 merge
- **THEN** 系统 MUST 拒绝重复、遗漏、split 重叠、shape/dtype、token 顺序、source SHA 或禁止字段不一致
- **AND** manifest MUST 标记 single-seed、inner-only、outer test false 与 claim eligible false

#### Scenario: 执行 F1 parity gate
- **WHEN** cache merge 重放 Full 和四个 single-missing 路径
- **THEN** Full Top1 agreement MUST 不低于 0.999，四个 single-missing agreement MUST 分别不低于 0.995
- **AND** 任一门槛失败时 launcher MUST 拒绝训练

### Requirement: Pattern 与 schedule 必须完整且组间均衡
系统 MUST 精确枚举除 Full 和 all-missing 外的 14 个 modality-level pattern，其中 single/double/triple missing 分别为 4/6/4 个；condition availability MUST 按 `image,lidar,radar,gps` 编码，并正确映射到 F1 的 `image,radar,gps,lidar` time-major 20-block mask。训练 MUST 先等概率选择 missing count 1/2/3，再在组内均匀选择 pattern，并为全部方法冻结相同 sample、mask、batch 顺序与 seed；validation MUST 对 14 个 pattern 逐一完整评测。

#### Scenario: 生成固定 schedule
- **WHEN** prepare 阶段生成 train schedule 与 validation manifest
- **THEN** 两者 MUST 使用相同 pattern id/availability/block-mask 映射且不包含 Full 或 all-missing
- **AND** single/double/triple 训练预算 MUST 在预注册容差内相等

### Requirement: Full 必须物理 bypass 且 missing token 不得泄漏
系统 MUST 在 Full 输入进入任何 fallback 模块前直接调用冻结 F1 路径，保持 logits 逐元素一致且 adapter forward count 为零。任意缺失输入 MUST 只修改当前可用 token，缺失位置在 F1 fusion 前保持零；all-missing MUST 明确报错。

#### Scenario: Full 通过 fallback wrapper
- **WHEN** availability 为 `[1,1,1,1]`
- **THEN** 输出 MUST 与冻结 F1 逐元素一致
- **AND** mask encoder、SSF、residual 和 auxiliary adapter forward count MUST 全部为零

#### Scenario: 修改缺失 token 内容
- **WHEN** 测试随机替换 unavailable token 而保持 availability 不变
- **THEN** U0--U5 输出 MUST 在数值容差内不变
- **AND** Only-one-modality MUST 正常预测而 all-missing MUST 在 forward 前失败

### Requirement: U0--U5 必须遵守冻结与结构公平边界
U0 MUST 是唯一 frozen F1 基线；U1 MUST 只选择当前 pattern 的 combination-specific SSF；U2 MUST 以一套共享 mask-conditioned hypernetwork 覆盖 14 个 pattern；U3 MUST 只调用四个 modality-specific contextual residual adapter 中当前可用者；U4/U5 MUST 具有相同 student 与 auxiliary 结构，且 U5 只增加冻结 unimodal teacher KD。F1 token adapter、fusion、output projection、prototype bank、temperature 与全部 teacher MUST 冻结。

#### Scenario: 构建六个方向
- **WHEN** preflight 为 U0--U5 构造 synthetic batch
- **THEN** 每个方向 MUST 输出 `[B,64]` logits，U2/U3 不得按 pattern 建立完整网络，U4/U5 state dict 结构 MUST 相同
- **AND** 只有新增 adapter、auxiliary head 或 teacher temporal head 可训练

### Requirement: Auxiliary 与 teacher 必须保持单模态监督隔离
每个 auxiliary head 和 teacher MUST 只读取对应模态的五个 token并通过冻结 BeamPrototypeBank 输出 logits。auxiliary 与 KD MUST 仅对当前可用模态执行，并先在单样本内对可用模态平均；teacher logits MUST stop-gradient，teacher feature MUST NOT 作为 student 输入。teacher checkpoint MUST 仅由 inner-validation 单模态 loss 选择。

#### Scenario: U5 计算 KD
- **WHEN** 样本仅有部分模态可用
- **THEN** 系统 MUST 只调用对应可用模态的 auxiliary head 和 teacher，并对其 loss 取样本内平均
- **AND** 缺失模态 teacher、其他模态 feature 与 full multimodal teacher MUST 不被读取

### Requirement: 训练选择与诊断必须预注册且 pattern-balanced
U1--U5 MUST 使用同一 cache、sample/mask schedule、seed、batch order、AdamW、scheduler、epoch budget、topology loss 和 BeamPrototypeBank。lambda MUST 只用固定 train batches 校准一次；batch loss MUST 先按 pattern 汇总，checkpoint MUST 仅由 single/double/triple validation loss 宏平均选择，Full、Top1、Worst、Missing LiDAR 与最终诊断 MUST NOT 参与选择。

#### Scenario: 选择 validation-best checkpoint
- **WHEN** 一个 epoch 完成 14-pattern validation
- **THEN** selection loss MUST 等于 single、double、triple 三组宏平均 loss 的平均
- **AND** validation 指标不得触发 lambda、超参数或 schedule 修改

### Requirement: 统一评测必须覆盖全部 pattern、机制与成本
每个可用 checkpoint MUST 在同一 eval evidence 上报告 Full、14 个 pattern、missing-count macro/worst、all-14 macro/worst、modality-absent macro、Top1/3/5、Within-3、MAE、weather、8-sector、error-distance 与成本。U3--U5 MUST 额外评测 normal/mean/zero/shuffle delta、表示变化和模态内容 shuffle；U4/U5 MUST 报告 probe/aux/teacher/fused 转化。

#### Scenario: 汇总候选方向
- **WHEN** U0--U5 评测完成
- **THEN** summary MUST 生成预注册 CSV/JSON/Markdown、逐项 success gates 和唯一推荐方向
- **AND** 辅助 head 改善不得在 fused 输出未改善时被解释为机制通过

### Requirement: GPU 编排与研究结论必须保持 inner-only 停止边界
系统 MUST 在 cache/preflight 和四个 teacher 完成后固定 U0--U5 到 GPU0--5，记录启动前 GPU 状态、PID、resolved config、日志、状态与退出码。单任务失败 MUST NOT 终止其他任务，也 MUST NOT 自动调参或重跑。结果 MUST 保持 single-seed、inner-only、claim-ineligible，完成后 MUST NOT 自动运行 outer test、multi-seed、encoder-level、end-to-end 或下一轮训练。

#### Scenario: 一个方向运行失败
- **WHEN** 任一 U0--U5 子进程非零退出
- **THEN** launcher MUST 等待其他方向并将该任务标记为 failed 与记录原因
- **AND** summary MUST 不伪造缺失结果或自动启动替代任务
