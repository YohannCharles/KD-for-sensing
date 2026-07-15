## ADDED Requirements

### Requirement: S1 temporal-then-modality baseline
系统 MUST 提供显式 opt-in 的轻量 S1 行为，先按 `modality_temporal_mask [B,T,M]` 为每个模态聚合历史表示，再复用 current supervised modality router。该行为 MUST 通过新的 `model.primary.temporal_pooling` 配置表达，MUST 不恢复旧 `temporal_router_type=s1_*` 或 S2-S4 route，且 disabled 时 MUST 保持 current U-Mask forward 和 checkpoint shape。

#### Scenario: Masked mean baseline 可复现
- **WHEN** `temporal_pooling.enabled=true` 且 `type=masked_mean`
- **THEN** 每个模态表示 MUST 等于有效历史 cell 的算术均值
- **AND** 缺失 cell MUST 不贡献分子或分母
- **AND** 聚合结果 MUST 进入现有 supervised router

#### Scenario: 默认配置不改变
- **WHEN** 配置未提供 `temporal_pooling` 或设置 `enabled=false`
- **THEN** 模型 MUST 使用 change 前 current U-Mask forward
- **AND** 模型 MUST 不新增启用路径专属参数到 state dict

### Requirement: Mask statistics 语义
系统 MUST 从历史 `modality_temporal_mask` 计算每模态 coverage、last-age、longest-gap、trailing-gap 和 missing-block count。统计 MUST 归一化、数值有限、只依赖历史 mask，且 MUST 能独立选择是否附加到 current router reliability features。

#### Scenario: 同覆盖率不同 gap 可区分
- **WHEN** 两个 mask 具有相同有效 cell 数但分别为 trailing block missing 和间歇缺失
- **THEN** coverage MAY 相同
- **AND** longest-gap、trailing-gap 或 missing-block count 中至少一个 MUST 不同

#### Scenario: Router stats 可独立消融
- **WHEN** `use_mask_statistics=true` 且 pooling type 为 `masked_mean`
- **THEN** 模态聚合内容 MUST 与 S1 masked mean baseline 相同
- **AND** router input metadata MUST 记录五个新增 feature 名称

### Requirement: 轻量时序聚合对照
系统 MUST 支持无参数 fixed-recency masked mean 和 gap-aware residual pooling。fixed-recency MUST 只改变有效 cell 的时间权重；gap-aware scorer MUST 使用历史内容、每 cell relative age、距前一有效观测间隔和 mask statistics，MUST 对缺失 cell 做 hard mask，并 MUST 以零初始化的 per-modality residual gate 锚定 masked mean。

#### Scenario: Gap residual 初始等价
- **WHEN** gap-aware pooler 刚构建且 residual gate 保持初始值
- **THEN** 其输出 MUST 与同输入的 masked mean 在数值容差内相等
- **AND** backward MUST 能为 scorer 和 residual gate 产生有限梯度

#### Scenario: 时间置换可区分
- **WHEN** 同一组有效特征被放置在不同历史位置且 residual gate 非零
- **THEN** scorer MUST 因 relative age 或 previous-observation gap 产生不同的 cell score
- **AND** 聚合 MUST 不再是对有效时间位置的置换不变函数

#### Scenario: 单 cell 与全空边界
- **WHEN** 某模态只有一个有效历史 cell
- **THEN** 三种 pooling MUST 输出该 cell 的表示
- **AND** 当一个样本所有 modality-time cell 均无效时模型 MUST fail fast 或由上游已登记 fallback 修复，不得产生 NaN

#### Scenario: 参数预算
- **WHEN** 使用默认 `d_model=64` 和默认 gap scorer hidden dim
- **THEN** gap-aware pooling 新增有效参数 MUST 小于 0.03M
- **AND** metadata MUST 记录 pooling type、参数量、residual gate 和 recency 配置

### Requirement: Coverage-aware router shrinkage
系统 MUST 提供显式 opt-in 的 coverage-aware router shrinkage，将 supervised router 权重向当前可用模态的均匀先验收缩。最终权重 MUST 对不可用模态为零、对可用模态和为一；完整 coverage 时收缩率 MUST 为零，单模态可用时最终权重 MUST 为一。

#### Scenario: 稀疏输入触发有界收缩
- **WHEN** coverage 低于完整输入且 shrinkage 启用
- **THEN** `rho` MUST 位于 `[0,rho_max]`
- **AND** 最终权重 MUST 等于 `(1-rho)w+rho*u`
- **AND** diagnostics MUST 记录 coverage、gate entropy、gate margin 和 rho

#### Scenario: Clean guardrail 结构保证
- **WHEN** 所有历史 modality-time cell 均有效
- **THEN** `rho` MUST 精确为零
- **AND** shrinkage 前后 router 权重 MUST 相同

### Requirement: 分阶段实验门禁
S1 lightweight profile MUST 以单组件 seed1 结果作为联合和多 seed 实验的晋级前提。为并行利用资源，workflow MAY 预计算联合 seed1，但只有其组成单项均满足正收益门禁后联合结果才有资格晋级。主要选择指标 MUST 为五档 mean Top1、Drop0-60 mean Top1 和 Drop80 Top1；候选相对同源码 S1 baseline 的 Drop0 下降超过 0.005 时 MUST 不晋级主方法。

#### Scenario: 首轮八卡筛选
- **WHEN** 用户运行 S1 lightweight seed1 profile
- **THEN** launcher MUST 生成 S1、T1、T2、A1、A2、A3、T1+T2 和 J1 八个独立任务
- **AND** 任务 MUST 可映射到 GPU0-7 且每卡最多一个训练进程

#### Scenario: 条件式后续实验
- **WHEN** 单组件或 J1 未通过选择指标与 Drop0 guardrail
- **THEN** workflow MUST 不把对应 J1、J2 或多 seed 结果标记为晋级
- **AND** 本地结果 MUST 保持 experimental，不得自动写入正式 claim
