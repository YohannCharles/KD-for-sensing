## ADDED Requirements

### Requirement: BPA topology descriptor 必须显式且可审计
U-MaskBeamJEPA 的 BPA target MUST 使用显式 `topology_id` 与 mapping descriptor，而不是未标注的 boolean wrap convention。系统 MUST 至少区分 `linear_index_v1`、`cyclic_index_v1` 与 deterministic `permuted_index_v1`；`physical_*` topology 只有在 matching codebook audit descriptor 存在时才能构建。

#### Scenario: 构建 BPA topology counterfactual
- **WHEN** T2 config 启用 BPA 并声明 topology descriptor
- **THEN** loss metadata MUST 返回 topology id、mapping checksum 和 target geometry
- **AND** invalid physical descriptor、non-bijective permutation 或 class-count mismatch MUST 使 config validation 失败

#### Scenario: BPA-disabled control
- **WHEN** matched control 关闭 BPA
- **THEN** config MUST 记录 topology 为 `not_applicable`
- **AND** 不得将 BPA topology metadata 用作该 control 的训练或 router margin 输入

### Requirement: MMW U-Mask availability 必须来自外部训练协议
MMW T2 和 S1 MUST 声明 external missing-mask mode，并把 batch 的 `available_modalities` 作为模型 availability。external mode 下若外部 mask 缺失、为空或 shape 不匹配，runtime MUST fail closed，而不得回退到 `p_missing`。

#### Scenario: external mask 进入 router
- **WHEN** MMW batch 已带合法 `[B,T,M]` mask 与 `[B,M]` availability
- **THEN** U-Mask forward MUST 使用完全相同的 `[B,M]` availability
- **AND** RouterNoPattern 设置 MUST 不引入额外缺失模式权重分支

### Requirement: Router oracle 必须显式处理并列最优模态
U-MaskBeamJEPA 的 Router oracle target MUST 声明 target mode。除 development control `hard_first` 外，任何 tie-aware hard/soft mode MUST 在多个 available modalities 具有相同最小 circular beam distance 时避免固定模态顺序偏置；soft mode MUST 产生只在 available modalities 上归一化的 target distribution，并使用 soft cross-entropy。

#### Scenario: exact-distance tie
- **WHEN** Image、Radar 与 LiDAR 的 unimodal prediction 到真实 beam 的 circular distance 相同且最小
- **THEN** `hard_confidence_tie` MUST 选择真实 beam 概率最高的并列模态
- **AND** `soft_uniform_tie` MUST 向三个并列模态分配相同 target mass
- **AND** `soft_confidence_tie` MUST 按三个模态的真实 beam 概率归一化 target mass

#### Scenario: distance-soft target
- **WHEN** config 选择 distance-soft Router oracle
- **THEN** target MUST 按 circular distance、正温度和 availability 构造归一化分布
- **AND** distance+confidence mode MUST 同时使用 detached unimodal true-beam probability
- **AND** 缺失模态 target mass MUST 为零

### Requirement: Tie-aware Router 筛选不得消费 outer evidence
系统 MUST 以 development-only `mmw_tie_aware_router_screen_v1` 在 frozen inner-train/inner-validation 上比较 Router target modes 与 Uniform fusion。筛选 MUST 固定 seed1、40 epoch、batch64、H4、RouterNoPattern 和训练 mask schedule，并 MUST 将结果标记为 `claim_eligible=false`。

#### Scenario: 八卡并行筛选
- **WHEN** 用户显式授权 GPU0--7 运行 tie-aware Router 筛选
- **THEN** 八个预注册候选 MUST 各绑定一张物理 GPU，且每卡同一时刻最多一个 job
- **AND** config、manifest 与输出 MUST 不引用 outer-evidence CSV
- **AND** canonical T2 MUST 在用户审阅筛选结果前保持不变

#### Scenario: ADBA-first 筛选汇总
- **WHEN** 八候选的 inner fixed-mask evaluation 已完成
- **THEN** summary MUST 复用既有 prediction metrics，以 circular progressive Top-3 ADBA（`delta=5`）为主排序、Top-1 为支线
- **AND** summary MUST 保持相同 15-domain、sample、mask、checkpoint 与 metric-profile identity
- **AND** summary MUST 输出逐域配对差异并保持 `claim_eligible=false`

### Requirement: beam-power utility Router 与 paired monotonic 必须是显式开发分支
U-MaskBeamJEPA MAY 在 `mmw_router_utility_screen_v1` 使用 training-only 64-beam power vector，将每个 available unimodal predicted beam 的 normalized power gain 转成 soft Router target。paired mode MUST 以同一样本 clean/corrupted view 计算受损模态权重，且 monotonic penalty MUST 只在该模态实际 normalized utility 下降超过阈值时启用；future power、clean target weights 与 unimodal utility MUST stop-gradient。

#### Scenario: GPS 噪声导致 utility 下降
- **WHEN** paired GPS corruption 使 GPS predicted-beam normalized gain 相对 clean 下降超过配置阈值
- **THEN** monotonic loss MUST 惩罚 corrupted GPS weight 未相对 clean weight 下降
- **AND** GPS utility 未下降时 MUST 不施加固定方向惩罚
- **AND** Image、Radar、LiDAR MUST 使用同一公式而不得有 GPS 特判

#### Scenario: utility target 只在训练存在
- **WHEN** model 执行 evaluation 或 inference
- **THEN** forward MUST 不要求 future beam-power vector、GPS scaler batch metadata 或 corrupted pair
- **AND** current T2/S1 与未启用 utility 的 baseline forward MUST 保持原 contract

### Requirement: paired Router 的质量信号必须使用连续 expected beam-power utility
`mmw_router_expected_utility_screen_v3` 的 paired branch MUST 以 detached unimodal beam distribution 在完整 future beam-power vector 上的期望 normalized gain定义 clean/corrupted utility，而不得用 argmax-selected beam 作为 monotonic gate。主 Router target 与 paired target MUST 可独立配置；启用 paired branch 不得强制主 target 改为 beam-power utility。paired utility CE MUST 支持 target-entropy informativeness gate，近均匀 target 不得产生 Router CE 梯度。

#### Scenario: argmax beam 不变但预测分布退化
- **WHEN** corruption 后单模态 argmax beam 不变，但概率质量从高功率邻域移向低功率 beam
- **THEN** expected utility MUST 下降并可激活配置的 quality-drop gate
- **AND** utility target、future power 与 clean weight MUST stop-gradient，paired Router logits MUST 获得有限梯度

#### Scenario: paired branch 与 current Router 主监督共存
- **WHEN** main target 为 `soft_confidence_tie` 且 paired branch 启用 expected utility
- **THEN** config validation MUST 接受该组合
- **AND** paired utility temperature、gate epsilon、loss weight 与 corruption seed MUST 写入 resolved config 和 provenance

#### Scenario: early expected-utility target 近似 Uniform
- **WHEN** 四模态 expected-utility soft target 的 entropy 高于配置上限
- **THEN** paired utility CE MUST 对该样本置零并报告 informative ratio
- **AND** clean/corrupted monotonic gate MUST 仍只由受损模态自身 utility drop 决定
