## Why

现有 MMW 结果已完成开发筛选，但历史 outer test 已被用于调参，且多 seed、固定 mask、物理 beam codebook 邻接关系和完整消融尚未在同一不可变协议下闭环。为使 T2 的贡献达到 TWC 可审计标准，需要冻结新的 group-safe 外层证据折叠，并将训练、评估、统计与拓扑反事实统一到同一 provenance。

## What Changes

- 新增冻结后的 MMW post-selection confirmation fold：仅由既有训练样本派生、排除已知开发验证样本，固定其 split manifest、样本身份和 train-fit GPS normalization 边界；历史 `h5p1_strict_v2` test 明确只保留为已消费的开发审计资料，不能称为 blind test。
- 新增版本化、内容寻址的 whole-modality 与时序缺失 mask cache，覆盖 Clean、Drop1--3、20/40/60/80% 及联合缺失，要求所有方法、seed 和域复用完全相同的 mask identity。
- 在不改写既有 `mmw_twc_outer_v1` 结果的前提下，新增 `mmw_twc_temporal_token_stress_v3` evaluation extension：以 5 帧×4 模态的 20 个 modality-frame 单元为唯一缺失单位，精确评估 20/40/60/80/90/95% 丢失。20--90% 各使用 100 个固定、随机化且集合级精确平衡的 mask，允许少量单个 mask 偏向同一模态；95% 使用 20 个 single-cell masks 并与主曲线分开解释。
- 新增 H4 + RouterNoPattern T2 主线，以及 S1、MaskTrain-CLS、AMBER-Full、RMBP-MM、AMR-Net-4M-Adapted 六方法严格多 seed 训练、固定 epoch `last.pth` 评估、domain/weather/worst-domain 汇总和配对置信区间流水线；`T2-PatternWeighted` 仅完成 seed1 开发筛选，因无稳定收益而退役，不进入 strict evidence matrix。
- 将六方法训练输入统一为互斥的公平外部 mask schedule，移除 T2/S1 的第二套 `p_missing` 随机 mask；清理 `outputs/` 中被新协议替代的旧训练、评估、筛选和无效 cache，同时保留 MMW/DeepSense6G 数据派生 cache、物理 codebook audit 与退役筛选的最小审计记录。
- 新增 T2 因果消融矩阵：BPA on/off、物理 topology/linear/random-permuted topology、prototype/classifier、router supervision、T2/S1 temporal consistency、训练掩码范围和 CMA matched control；所有反事实共享同一外层 split、训练预算和 mask cache。
- 新增物理 codebook 拓扑审计，优先从 MMW channel/codebook metadata 恢复 local-array beam 邻接；若证据不足，明确拒绝以标签环形索引声称物理相邻，并将 BPA 结论限制为 label-space topology prior。
- 新增独立 `deepsense6g_twc_secondary_v1` 次数据集证据：将 Scene31--34 合并为一个完整数据集，以 T2、MaskTrain-CLS、AMBER-Full、RMBP-MM、AMR-Net-4M-Adapted 各 3 seeds 运行固定 whole/token mask，不为每个 scene 单独训练 checkpoint，也不与 MMW summary 混合。
- 新增无线功率/速率指标、BPA 与 router 定量机制证据、固定传感器 corruption evaluation extension（默认关闭，仅显式压力测试请求启用），以及参数量/MACs/延迟/吞吐/峰值显存表；不再把 PCA/t-SNE 作为主要机制证明。
- 新增由 manifest 和 run status 自动生成的 `outputs/twc_experiment_status.md`，持续记录 planned/running/complete/failed、配置/seed/GPU/产物和 claim eligibility。
- 新增 development-only 的 tie-aware Router oracle 筛选：只使用冻结的 inner-train/inner-validation，比较 current hard-first、置信度破同分、两种 exact-tie soft target、两种 distance-soft target、distance+confidence soft target 与 Uniform fusion；筛选结果不得进入 outer claim，canonical T2 在确认前保持不变。
- 新增用户显式启用的 `mmw_router_oracle_gap_v1` 机制压力测试：固定同一个 tie-aware inner checkpoint，在保持四模态 availability 为真的前提下，按有文献依据的图像遮挡、Radar AWGN、LiDAR 稀疏和米级 GPS 位置噪声逐级降质，并比较同 checkpoint 的 Uniform、Learned Router 与逐样本通信效用 Oracle；完整保存 logits/权重/融合输出，报告 ADBA、通信指标、Oracle regret 与权重单调性，但不升级为 outer claim。
- 新增 `mmw_router_utility_screen_v1` inner-only seed1 开发筛选：训练期只读加载完整 future beam-power vector，以单模态 predicted-beam normalized gain 监督 Router，并通过不改源码数据的 online clean/corrupted paired forward 加入 quality-gated monotonic loss；八候选固定 GPU0--7、40 epoch、batch64，评估使用独立 corruption seed，用户确认前不提交 seed2--5 或修改 canonical T2。
- 新增 `mmw_router_expected_utility_screen_v3` 修复筛选：冻结 v1 argmax utility 零激活及 v2 早期 expected target 近均匀的负结果，改用单模态预测分布在完整 beam-power vector 上的期望 normalized gain，并以 target entropy fail closed 于无信息 paired CE；主 Router 默认继续使用已验证的 `soft_confidence_tie`，paired utility/monotonic 只训练 Router。提交八卡训练前必须在既有 13-condition inner traces 上逐传感器、逐强度验证连续 utility drop、成熟 target gate coverage，并通过 focused gradient smoke；仍只跑 seed1，且不自动晋升 canonical T2。
- 新增 `mmw_router_joint_stress_v1` inner-only seed1 机制诊断：固定 CurrentControl checkpoint，把 5 帧×4 模态的 20 个 cell 以固定三状态 panel 分为 Clean、Drop 与 S2 Corrupt，在 Joint20/40/60/80 下比较同 checkpoint Learned、availability-normalized Uniform 与通信效用 Oracle；筛选先使用每级 20 个精确 cell/modality/frame 平衡 mask 并由 GPU0--7 分片执行，只有当前窗口级 Router 未通过预注册门槛时才规划 R1/R2 结构筛选。
- 为该诊断增加独立 post-hoc static-prior falsification：从同一 Clean trace 冻结可部署的全局 Router 先验及逐样本 Clean 反事实权重，只复用已保存的 stressed unimodal logits，判断 Learned>Uniform 是否主要来自静态模态先验；该检查不得回写或重新定义预注册 Gate。
- 将后续新冻结证据的主排序指标改为 circular progressive Top-3 ADBA（`delta=5`），Top-1 保留为支线；既有 outer evidence 不得在看过结果后回填为新的 ADBA 主张，tie-aware 筛选只从同一批 inner fixed-mask 预测生成 ADBA-first 汇总。
- 为所有新训练、checkpoint、evaluation row 和 summary 注入 evidence protocol、split、mask、recipe、profile、seed 与 topology provenance，并在身份不一致时 fail closed。
- 修复 strict outer evaluator 与 current `TaskForwardResult` 的标签接口，并使 nightly queue 在单条 evaluation 失败时隔离失败、等待已启动训练收尾；中断训练只可从其同 run 的已发布 `last.pth` 经 strict resume contract 续跑。

## Capabilities

### New Capabilities

- `mmw-twc-rigorous-evidence`: 定义 MMW 新 outer evidence split、固定 mask、严格多 seed 训练/评估、统计和证据身份边界。
- `mmw-codebook-topology-evidence`: 定义 MMW beam label 到 local-array codebook 邻接的可验证审计、反事实配置和结论边界。

### Modified Capabilities

- `mmw-baseline-multiseed-robustness-evidence`: 将四个主比较方法从历史本地验证扩展为新的固定 outer evidence protocol，并要求全 seed/mask/split 完整性；开发筛选失败的 Pattern-weighted CE 不再构成完整性要求。
- `training-evaluation-runtime`: 要求严格 evidence protocol 在训练、checkpoint、fixed-mask evaluation 与 summary 中传播和校验 provenance。
- `u-mask-beam-jepa`: 扩展 BPA 目标的 topology descriptor，使 circular、linear、physical graph 与 deterministic random permutation 反事实可明确区分。
- `t2-baseline-surface`: 增加 MMW 的 MaskTrain-CLS、AMR-Net-4M-Adapted，以及 DeepSense6G 的五方法次验证闭包。
- `canonical-config-resolution`: 增加两个 MMW baseline recipe 和四个 DeepSense6G baseline recipe，保持 clean-clone 可解析。
- `project-architecture`: 允许两个新 baseline 只作为 `modular_sequence` representation core 接入，不恢复 retired AMR whole-model/runner。

## Impact

- 影响 MMW split/cache helper、all-weather launcher/evaluator/summary、Router joint-stress evaluation、T2 BPA loss/config、checkpoint provenance 与相应测试。
- 新生成 split CSV、mask cache、generated config、checkpoint、日志、指标和图表全部写入 `outputs/` 或 `outputs/cache/`，不修改 `dataset/` 或历史 checkpoint；新增 DeepSense baseline canonical recipe 与独立公平 schedule，MMW 已冻结主线 recipe 保持不变。
- 不新增第三方依赖；所有项目 Python 命令、训练与验证均通过 `conda run -n kd_mm_beam` 执行。
