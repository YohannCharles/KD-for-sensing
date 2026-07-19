## ADDED Requirements

### Requirement: MMW strict outer evidence split 必须不可变且隔离历史 test
系统 MUST 从每个 15-domain 的 `h5p1_strict_v2/train_with_radar_with_bs_gps.csv` 建立唯一的 `mmw_twc_outer_v1` 三角色 group-safe post-selection confirmation split。它 MUST 不读取、复制或评估该 protocol 的 historical test CSV，并 MUST 排除 manifest 声明的已消费开发验证 identity；inner-train、inner-validation 与 outer-evidence 的 source/exclusion hash、group assignment、frame/resource/stable-id leakage audit 和 CSV SHA256 MUST 写入 immutable manifest。

#### Scenario: 生成 strict outer split
- **WHEN** 用户请求生成 `mmw_twc_outer_v1`
- **THEN** 每个 domain MUST 产生互斥的 inner-train、inner-validation 和 outer-evidence CSV
- **AND** manifest MUST 记录 group-safe block/guard、split seed、角色样本数和所有输入/输出 hash
- **AND** 任一 frame、resource reference、stable sample identity 或 group overlap MUST 使生成失败
- **AND** manifest MUST 标记该结果为 `post_selection_confirmation_not_historical_blind_test`

#### Scenario: 训练配置读取 strict split
- **WHEN** strict evidence launcher 生成任一 method/seed 的 config
- **THEN** config MUST 仅引用 manifest 中对应 domain 的 inner-train/inner-validation/outer-evidence CSV
- **AND** checkpoint provenance MUST 记录 evidence protocol id 和 split manifest SHA256

### Requirement: fixed-mask cache 必须跨方法与 seed 可复用
系统 MUST 在 strict evidence protocol 下生成内容寻址的 immutable evaluation mask cache，覆盖 Clean、全部非空 whole-modality subset、20/40/60/80% 的 `modality_frame`、`frame_level`、`block` masks，以及预注册的 Drop1/Drop2 × Block40/Block80 联合 masks。cache MUST 记录 canonical masks、seed、digest、checksum 和生成算法版本。

#### Scenario: 生成或重用 cache
- **WHEN** strict evaluator 请求 `mmw_twc_outer_v1` 的 mask cache
- **THEN** 已有同 checksum 的 cache MUST 直接重用
- **AND** 路径相同但 canonical content 不同的 cache MUST 被拒绝
- **AND** whole、temporal 与 joint condition MUST 都有稳定 mask identity

### Requirement: 精确 modality-frame temporal stress 必须作为版本化 evaluation extension
系统 MUST 不修改已发布的 `mmw_twc_outer_v1` cache、训练 config、checkpoint 或 summary，而 MUST 以独立的 `mmw_twc_temporal_token_stress_v3` evaluation extension 复用其冻结的 outer-evidence split 和 `last.pth`。该 extension MUST 只把 5 帧×4 模态的 20 个 modality-frame cell 作为缺失单位，且 MUST 覆盖 20/40/60/80/90/95% 的精确整数 cardinality。

20/40/60/80/90% 每个 rate MUST 使用 100 个固定、seeded masks；95% MUST 使用 20 个唯一 single-cell masks。跨同一 rate 的全部 mask，每一个 modality-frame cell MUST 有相同的保留次数；因此每种模态和每个时间位置的平均丢失率 MUST 与请求 rate 相同。单个 mask 的模态组成 MAY 偏斜。cache、manifest、evaluation row 和 summary MUST 记录 requested/observed rate、20-cell cardinality、per-modality/per-frame retain/drop counts、mask-balance policy、seed、canonical matrix、digest、cache checksum 和 parent protocol SHA256。

#### Scenario: 生成精确且平衡的 90/95% token masks
- **WHEN** 用户请求 `mmw_twc_temporal_token_stress_v3`
- **THEN** 90% rate MUST 每个 mask 保留恰好 2/20 个 modality-frame cells，95% rate MUST 每个 mask 保留恰好 1/20 个 cell
- **AND** 90% 的 100 个 mask MUST 让每一个 cell 保留 10 次；95% 的 20 个 mask MUST 让每一个 cell 保留 1 次
- **AND** 每一种模态在各自 rate 下的 aggregate missing rate MUST 精确为 90% 和 95%
- **AND** 90% 的两个幸存 cell MAY 来自相同模态或相同帧，但跨 panel 的 cell 边际 MUST 精确公平

### Requirement: token-stress panel 必须集合级公平且保留随机组成
在 20/40/60/80/90% rate 下，系统 MUST 从随机 K-of-20 masks 构建固定 panel，并以最少 cell swap 修正为每个 cell 相同保留次数。系统 MUST 不强制单个 mask 的四模态组成相同，并 MUST 输出 per-mask modality composition histogram 以审计随机偏斜。

#### Scenario: 80% token drop 允许少量单模态幸存
- **WHEN** 系统生成 80% rate 的 100 个 masks
- **THEN** panel MAY 包含少量四个幸存 cell 来自同一模态的 mask
- **AND** 每一个 cell MUST 恰好在 20 个 masks 中幸存
- **AND** 任意 cache checksum、parent protocol、outer split 或 condition identity 不一致 MUST 使 evaluator/summary fail closed

#### Scenario: 报告主曲线与 single-cell endpoint
- **WHEN** 完整 4 methods×5 seeds×15 domains 的 token-stress evaluation 完成
- **THEN** summary MUST 报告 Clean、20/40/60/80/90% 的 modality-frame 主曲线和 0--90% AUC
- **AND** 95% MUST 单独报告为 `SingleCell95`/extreme-stress endpoint，且不得纳入 0--90% AUC
- **AND** paired domain×seed bootstrap CI、coverage、mask-balance audit 和全部复现参数 MUST 与数值一并写出

#### Scenario: 跨方法和 seed 固定评估掩码
- **WHEN** evaluator 对不同 method 或 seed 运行同一 evidence cell
- **THEN** 它们 MUST 使用相同 cache checksum、mask digest、domain sample CSV hash 和 metric profile
- **AND** summary MUST 拒绝任何不一致的比较行

### Requirement: strict evidence 的主比较和统计必须完整
系统 MUST 以 `(1,2,3,4,5)` 五个固定 training seed、batch 64、40 epoch、fixed `last.pth` 执行 T2、S1、MaskTrain-CLS、AMBER-Full、RMBP-MM 与 AMR-Net-4M-Adapted。主表 MUST 以 circular progressive Top-3 ADBA（`delta=5`）报告六方法的 15-domain macro Clean、whole Drop1/Drop2/Drop3 平均和 Block80，Top-1 MUST 作为支线保留。所有方法同时输出 weather、scene、worst-domain、Top3/5、within-1/3、temporal AUC 与 paired 95% CI。

#### Scenario: 完整主比较
- **WHEN** 五 seed 的六方法训练与 fixed-mask evaluation 全部完成
- **THEN** summary MUST 仅从 `last.pth` 的 complete 15-domain outer evidence 聚合主表
- **AND** 每一个 reported paired contrast MUST 使用相同 seed、domain、mask、split 和 metric identity
- **AND** CI、seed count 和缺失 cell 状态 MUST 与数值一并写出

#### Scenario: ADBA-first 规则在 confirmation 前冻结
- **WHEN** 开发筛选已经产生过 Top-1 排序后选择 ADBA 作为后续主指标
- **THEN** 既有 outer evidence MUST 不得被追溯改写为 ADBA 主张
- **AND** 新 confirmation manifest MUST 在执行前记录 ADBA 定义、`delta=5`、circular distance mode 和 Top-1 secondary 身份

#### Scenario: 不完整主比较
- **WHEN** 任一方法缺失 required seed、domain、mask 或 provenance field
- **THEN** summary MUST 标记该主比较不可用
- **AND** 不得以剩余 seed、domain 或 mask 计算论文均值

### Requirement: overnight 作业必须可恢复且不越过 GPU 边界
strict evidence launcher MUST 生成具有 source/config/mask/split/provenance identity 的持久 job manifest。默认它只调度 GPU4、GPU5、GPU6、GPU7；只有在用户显式授权且传入 `--allow-gpu0-3` 时，才可调度 GPU0--GPU7。它 MUST 支持从已完成 job 恢复，且不得终止、迁移或修改不属于该 manifest 的进程。

#### Scenario: 默认 GPU4--7 nightly queue
- **WHEN** 用户启动 strict evidence nightly queue
- **THEN** 在未传入授权扩展 flag 时，launcher MUST 只设置 `CUDA_VISIBLE_DEVICES` 为 4--7 中的一个值
- **AND** 每张卡同一时刻 MUST 至多运行一个该 manifest 的训练 job
- **AND** job 完成、失败和未开始状态 MUST 原子写回 manifest

#### Scenario: 用户授权的 GPU0--7 扩展
- **WHEN** 用户明确允许使用 GPU0--3，且 launcher 收到 `--allow-gpu0-3` 与 GPU0--7 的显式列表
- **THEN** launcher MAY 只为本 manifest 的 job 设置 `CUDA_VISIBLE_DEVICES` 为 0--7 中的一个值
- **AND** launcher MUST 保持每张卡同一时刻至多运行一个该 manifest 的训练 job
- **AND** launcher MUST 不终止、迁移或修改其他 manifest 的进程

#### Scenario: 单条失败与 orphan 恢复
- **WHEN** 一个本 manifest 的训练或 evaluation job 失败，而同一 manifest 仍有存活子进程
- **THEN** launcher MUST 记录失败并停止提交新的 job，但 MUST 等待已启动子进程自然结束
- **AND** launcher MUST 不因自身退出向已启动子进程传播终止信号
- **AND** 重启后仅当 orphan training job 在其原始 `run_dir` 有可验证的 `checkpoints/last.pth` 时，launcher MAY 使用 strict `--auto-resume` 续跑
- **AND** failed evaluation MUST 仅在用户显式 retry 请求后回到 planned，且 retry history MUST 保留在 manifest

### Requirement: Oracle Gap Test 必须隔离 checkpoint 差异并保存可复算 trace
系统 MUST 将 `mmw_router_oracle_gap_v1` 限制为同一个 frozen inner-validation checkpoint 的推理期机制诊断。Uniform、Learned Router 和逐样本 Oracle MUST 使用完全相同的四个 unimodal logits；Oracle MUST 按 `future_beam1` normalized gain 选择最佳单模态，而不是按 label distance 选择。图像遮挡、Radar AWGN、LiDAR 稀疏和 GPS 位置噪声 MUST 保持四模态 availability 为真，且 MUST 使用 manifest 冻结的有单位 severity、seed、算法版本和参考文献。

#### Scenario: 同 checkpoint 三分支比较
- **WHEN** evaluator 处理任一 clean 或 corruption condition
- **THEN** Learned MUST 使用 checkpoint 原生 Router，Uniform MUST 等权平均相同 unimodal logits，Oracle MUST 选择实际通信效用最高的 unimodal logits
- **AND** 三分支 MUST 使用相同 sample、target、future beam power 和 metric profile
- **AND** trace MUST 保存四个 unimodal logits、Router 权重、三种 fused logits、Oracle modality 与复算通信指标所需 identity

#### Scenario: 可用但降质的单传感器压力测试
- **WHEN** 任一传感器 severity 从 Clean 增加到 1、2、3
- **THEN** availability 和 modality-temporal mask MUST 保持完整可用
- **AND** GPS MUST 在相对 XY 米空间加噪后重新标准化，Radar MUST 按目标 SNR 加 AWGN，LiDAR MUST 空间一致地丢弃 BEV cells，图像 MUST 按固定面积比例遮挡
- **AND** summary MUST 报告 ADBA、Top-1、normalized gain、rate ratio、normalized gap closure、soft/selection Router regret、mean weight trajectory、Spearman correlation 和逐样本单调下降比例
- **AND** 任何非单调响应或未缩小 Oracle gap 的结果 MUST 原样保留，不得改变 checkpoint、severity 或 domain coverage 后选择性报告

### Requirement: Joint Drop+Corrupt 必须使用固定三状态 cell panel
系统 MUST 将 `mmw_router_joint_stress_v1` 限制为同一个 frozen inner-validation CurrentControl seed1 checkpoint 的推理期诊断。5 帧×4 模态的每个 cell MUST 恰为 clean、drop 或 corrupt：drop MUST 令对应 temporal mask 为 false，corrupt MUST 保持 cell 可用并施加 manifest 声明的模态 S2 corruption，三种状态 MUST 互斥。Joint20/40/60/80 的 screen panel MUST 各有 20 个固定 mask，并在每个 mask 上分别包含 2/4/6/8 个 drop cell 和相同数量的 corrupt cell。

#### Scenario: 三状态 panel 集合级公平
- **WHEN** 系统生成任一 Joint20/40/60/80 screen panel
- **THEN** 每个 mask MUST 具有精确 clean/drop/corrupt cardinality
- **AND** 跨 20 个 mask 的每一个 modality-frame cell MUST 具有相同 drop 次数和相同 corrupt 次数
- **AND** 单个 mask MUST 不强制四模态状态组成相同，cache MUST 记录 state matrix、per-cell/per-modality/per-frame audit、seed、digest 和 checksum

#### Scenario: 同 checkpoint Joint Router 比较
- **WHEN** evaluator 处理任一 joint mask
- **THEN** Learned、availability-normalized Uniform 与通信效用 Oracle MUST 使用相同可用模态的 unimodal logits、sample、target 和 future beam power
- **AND** trace MUST 保存三状态矩阵、availability、temporal mask、unimodal/fused logits、Router 权重和 Oracle modality
- **AND** summary MUST 报告 ADBA、Top-1、normalized gain、rate ratio、Learned-Uniform paired delta/CI、win rate、gap closure 与 Router regret，并保持 `claim_eligible=false`

#### Scenario: 结构修改 fail closed
- **WHEN** 当前窗口级 Router 在 Joint40/60/80 未显示稳定正的 Learned-Uniform 差值
- **THEN** 本协议 MUST 只记录失败模式，不得自动修改 canonical T2
- **AND** 任何 temporal/hierarchical Router MUST 先进入独立 OpenSpec matched screen，且人工 corruption state MUST 不得作为推理输入

### Requirement: Joint Router 必须区分 Uniform Gate 与静态先验反事实
系统 MUST 在预注册 Learned/Uniform/Oracle Gate 之外，以相同 trace 生成 `GlobalCleanPrior` 与 `PerSampleClean` post-hoc control。该 control MUST 不重新训练、不重新前向、不修改 fixed masks，也 MUST NOT 回写预注册 Gate。若 Dynamic 未稳定优于 `GlobalCleanPrior`，系统 MUST 将 corruption-adaptive reliability claim 标为 unsupported，即使 Learned-Uniform Gate 为 PASS。

#### Scenario: Learned 优于 Uniform 但不优于静态先验
- **WHEN** Joint40/60/80 的 Learned-Uniform Gate 通过，而 paired Dynamic-GlobalCleanPrior 差值没有稳定正证据
- **THEN** 系统 MUST 保留原 Gate 结果并单独记录 static-prior falsification
- **AND** 报告 MUST 将当前收益解释为 learned non-uniform fusion，而不是已证实的动态可靠性自适应
