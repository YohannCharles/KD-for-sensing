## Context

MMW 当前主线为 15 个天气×场景域、四模态、5-to-1、40 epoch 和固定 `last.pth`。H4 + RouterNoPattern 仅通过已消费的开发验证被固定为工程主线；旧 `h5p1_strict_v2` test 因此前调参不可再作为盲测。现有 evaluator 已有固定时序 mask 和 provenance 基础，但 split、whole-mask、联合缺失、拓扑状态和跨 seed 统计尚未组成单一不可变 protocol。

本设计新增 `mmw_twc_outer_v1`。它只读取每个域既有 `h5p1_strict_v2/train_with_radar_with_bs_gps.csv`，不读取其 historical test，并排除当前可审计的开发验证样本；按既有 group-safe time block 定义划为 inner-train、inner-validation 和 outer-evidence 三个互斥角色。它是冻结主线后的 post-selection confirmation fold，不是可恢复的历史 blind test。所有派生 CSV、mask cache、config、run、评估和图表均放入 ignored `outputs/`。源数据和历史输出绝不被覆写。

## Goals / Non-Goals

**Goals:**

- 用固定 seed 集合 `(1,2,3,4,5)`、固定 64 batch、40 epoch、`last.pth`、固定 H4/RouterNoPattern provenance 完成 T2、S1、MaskTrain-CLS、AMBER-Full、RMBP-MM、AMR-Net-4M-Adapted 的严格可复现六方法执行矩阵；Pattern-weighted CE 只保留 seed1 开发筛选的最小审计记录，不进入正式矩阵。
- 用一个内容寻址 evidence manifest 冻结 15 域 split、五类 whole-modality 条件、时序 20/40/60/80%、联合缺失及其缓存 SHA256；所有方法和 seed 必须读取同一 cache。
- 让 outer evidence 只用于已冻结配置的报告，inner validation 不得再用于微调主线；所有 checkpoint 和 evaluator 必须记录 protocol/split/mask identity。
- 完成能回答机制问题的配对消融，而非扩大模型：BPA、topology、head、router、T2 temporal consistency、训练 missing coverage 和 CMA replacement。
- 从 dataset metadata、64-beam ULA-DFT 定义及 RSU local frame 恢复可验证的 topology；以 codebook response/center sequence 判定 endpoint 是否邻接，并把结论强度限制在证据可支撑范围内。
- 生成可用于论文的 domain/weather/worst-domain、fixed-mask curve、paired seed interval、topology audit 和复杂度证据。
- 在由 DeepSense6G Scene31--34 合并成的单一完整数据集上，以独立 split/mask/provenance 完成五方法三 seed 次验证，并输出 beam-power-aware 通信指标和定量机制图。

**Non-Goals:**

- 不再将 RouterNoPattern、GPS 坐标校正、codebook topology 或任何 ablation 的 confirmation-fold result 用于新的结构选择，也不将其称为 historical blind test。
- 不修改 `dataset/`、历史 `h5p1_strict_v2` CSV、已有 checkpoint 或 MMW 已冻结主线 YAML；允许为新增 DeepSense 公平协议增加 canonical recipe，但不引入外部预训练模型和第三方依赖。
- 不声称 AMBER-Full/RMBP-MM/AMR-Net-4M-Adapted 是原论文端到端完全等价复现；结果持续标注为四传感器本地适配。
- 不恢复 retired `amr_net` whole-model、旧 AMR runner 或 BEV-Fusion reproduction；AMR 只以 current `modular_sequence` uncertainty core 接入，BEV-Fusion 仅在 Related Work 说明。
- 不把 codebook label 环形索引本身等同于物理角度或全球 GPS 方位。

## Decisions

### 1. 新 outer 证据从旧 train 派生，而不是重用已消费 test

每个域先验证 source CSV 的 hash、`contiguous_segment_id`、window frame ids 和资源引用，并从输入中排除可审计的 H4 design inner-validation identities；然后以单次 group assignment 将剩余 block 划为 64% inner-train、16% inner-validation、20% outer-evidence。block、guard、seed、原始 source hash 与 exclusion hash 写入 manifest，并对任意角色对做 frame/resource/stable-id overlap audit。任何审计失败会拒绝生成 config。

选择该方案是因为旧 test 已经影响设计决策，重用会令 blind-test claim 不成立。由于历史开发已经覆盖原始训练数据的一部分，它只能构成 post-selection confirmation，而非真正未接触的外部盲测；论文必须保留这一边界。重新写 dataset 下 split 会污染本地输入；保存绝对 CSV 路径到 `outputs/cache/` 由现有 MMW loader 支持且保留 source read-only 边界。

### 2. 训练随机性与评估 mask 分开冻结

五个训练 seed 控制 model initialization、sampler、worker 和训练缺失采样；每一方法同一 seed 使用相同 `temporal_missing.seed`，因此训练缺失轨迹跨方法可重现。训练 sampler 先在互斥类别中选择 Clean 20%，Drop1/Drop2/Drop3 各 10%，TokenDrop20/40/60/80/90 各 10%；Drop95 只评估不训练。whole subset 与 modality-frame cell 在可审计 schedule panel 上集合级平衡，单个 mask 仍允许随机偏斜。

T2 和 S1 的 U-Mask extension 直接消费上述外部 `available_modalities`，不得再调用独立 `p_missing` sampler。T2 保留 same-model full-modal superset forward，S1 关闭它；这是模型机制差异，不是第二套输入 mask。评估另用一个与训练 seed 无关的 immutable cache，包含 whole subset、temporal mask、联合 mask、mask digest 和 cache SHA256。缓存以 protocol id + canonical JSON 哈希命名，已有但不相同的文件拒绝复用。

固定每个训练样本的 40 epoch mask 会把 missing augmentation 变成极小的静态训练集，故被拒绝；严格要求的是确定性 pseudo-random training schedule 与全方法共享的固定 evaluation masks。

WholeOnly 反事实仍复用 `balanced_pattern_schedule` runtime，但使用独立的 `mmw_fair_whole_modality_v1` 480-entry panel：Clean、Drop1、Drop2、Drop3 各 120 项，且每个 drop count 内的模态组合精确等频；token20/40/60/80/90 的 condition count 显式记录为 0。其 seed algorithm 与 panel checksum 必须单独记录，任何 modality-frame、frame 或 block token 缺失都不得进入该变体。仅设置旧 stratified sampler 的 rate/type 字段而不改变 active panel 是无效配置，launcher 与测试必须拒绝这种 no-op。

### 3. 配置在 outer 前冻结，outer 行只做报告

主线固定为当前 H4 + RouterNoPattern，S1 只关闭 T2 的 temporal-superset consistency。Pattern-weighted CE 的 seed1 inner-validation fixed-mask 开发筛选未显示稳定收益，并在高缺失条件下降低性能，因此停止补 seed、从 current runtime 删除，只归档配置、指标、provenance 与日志。六方法在同一 split、epoch、batch、domain-balanced sampling 和外部可用模态上训练。MaskTrain-CLS 只使用 availability-normalized mean fusion、普通 classifier CE 和相同外部 mask；AMR-Net-4M-Adapted 使用四模态 Gaussian embedding uncertainty fusion，并明确 `paper_equivalent=false`。训练完成后只加载 `last.pth`，不以 validation 最优 checkpoint 或 outer metric 选择。

执行队列支持显式的 seed-gate 阶段：先只提交所有当前方法的 seed1 并完成固定 outer evaluation，输出可审阅的初步结果；在用户确认前不得自动提交 seed2--5。确认后才恢复同一协议下的 seed2--5，且不改变 config、split、mask 或 checkpoint contract。完整阶段仍按 5×5 任务执行；夜间资源不足以同时完成所有后续反事实时，队列优先保证每个 method/seed 完整，随后按 BPA/topology/head/router/temporal/mask/CMA 的 matched-control order 续跑。未完成任务不会被静默从表格或均值中删除。

### 4. 主报告使用 ADBA，Top-1 作为支线

主指标为 15-domain macro circular progressive Top-3 ADBA（`delta=5`）：Clean、whole Drop1/Drop2/Drop3 平均和 temporal Block80。ADBA 以预测 Top-3 中截至每一 rank 的最小 circular beam distance 计算渐进得分，直接奖励落在真实 beam 邻域内的候选。Top-1 作为支线，同时报告 Top-3/5、within-1/3、linear/cyclic/physical codebook error（仅当 topology audit 允许）、weather macro、worst-domain、temporal AUC 及 complexity。统计以 method×seed 与 domain×seed 配对差异计算 bootstrap 95% CI；主结论要求完整五 seed 集且 paired ADBA CI 不跨零。缺失任一 method/seed/required mask/domain 时 summary fail closed。

该优先级变更发生在 tie-aware inner-development 筛选已经产生 Top-1 结果之后，因此不得追溯改写既有 outer evidence 的主指标身份。只有在候选和 ADBA-first 规则冻结后重新执行的独立 confirmation protocol 才可形成 ADBA 主张；旧 outer 与当前 inner 筛选持续标记为 development evidence。

### 4.1 精确 modality-frame stress 作为独立 evaluation extension

已发布的 `mmw_twc_outer_v1` 仍是其原始 immutable evidence，不能把额外 rate 写入或重新解释为精确 token-drop。新的 `mmw_twc_temporal_token_stress_v3` 只复用其 outer split、冻结 checkpoint 与训练 provenance，并拥有独立的内容寻址 cache、evaluation manifest、输出目录和 summary；它不是新的训练协议，也不改变已训练模型的 recipe identity。先前的 v2 cache 只满足一阶边际平衡，但没有约束单个 mask 的模态组成，因而保留为拒绝使用的审计产物，不用于 claim。

stress cache 只使用 `modality_frame` mask。每个样本窗口固定为 5 帧×4 模态=20 个单元，rate `r` 必须精确丢弃 `20r` 个单元：20/40/60/80/90/95% 对应保留 16/12/8/4/2/1 个单元。20/40/60/80/90% 各使用 100 个固定、seeded masks；集合内每个 modality-frame cell 保留次数精确相同，模态与时间位置平均缺失率精确等于请求 rate，但单个 mask 不约束每模态组成。panel 从均匀 K-of-20 候选开始，以最少 cell swap 修正边际，因此允许自然出现少量单模态幸存。95% 只有 20 个唯一 single-cell 状态，每个 cell 轮流保留一次。

该 panel 是共享于全部方法和 seed 的 fixed randomized balanced evaluation，不是把数据集拆给不同 mask，也不是样本独立的在线随机评估。cache、每个 condition、每一行 evaluation 和 summary 都记录 token 数、保留/丢失数、平衡策略、每模态/每帧计数、seed、canonical matrix、digest 和 checksum。

95% 保留一个单元，语义上是 single-cell/unimodal fallback，而非仍可进行多模态融合的普通缺失率。论文主曲线使用 Clean、20/40/60/80/90%，95% 单独列为 extreme-stress endpoint 和附录表；两者都使用相同固定 mask 与配对统计，但不得将 95% 纳入主 Temporal AUC。

### 5. 物理 topology 采用“先审计、后命名”的策略

审计器读取每域 `Prepared/<scene>/metadata.json` 的 `channel_to_beam`、`ula_dft`、`num_beams`、`tx_antennas`，并从场景 config 读取 RSU transform。它以明确的 ULA codebook construction、local spatial-frequency center 和 sampled beampattern overlap 生成 label adjacency；报告 label 0/63 的 local response relationship、全局 yaw 仅用于 GPS-world 到 local-frame 的解释。

若所有 metadata 和 codebook 检查一致且 endpoint adjacency 通过预注册阈值，BPA 的 `circular` 可称作 codebook-cycle prior；否则它只能称为 index-space cyclic prior，论文物理 claim 改为 `linear` 或明确的 graph descriptor。`random_permuted` 对照只打乱 target-topology mapping，保持 classifier、loss weight、seed、split、batch 完全相同，用于检验收益是否来自 label ordering。

### 6. 消融只改变一个因果对象

按冻结的 H4+RouterNoPattern control，消融为：NoBPA；BPA-circular/linear/physical-or-index-verified/random-permuted；prototype/classifier；router oracle/full、uniform 和 reliability-only；T2/S1；600-entry full structured missing training 与 480-entry balanced whole-modality-only training；BPA 与 NoBPA-CMA matched replacement。CMA 永不与 BPA 叠加。每一条记录 `matched_control`、allowlist diff 与 topology descriptor，summary 在它们不一致时拒绝配对。

### 7. 失败恢复不改变实验身份，也不丢弃活跃训练

strict outer evaluator 只能通过 shared runtime 的 `prepare_task_labels(step.batch, ...)` 取得标签，不能依赖已经移除的 `TaskForwardResult.labels`。nightly queue 发现一个 evaluation 或 training failure 后，先原子记录失败并停止提交新的 work；对本 manifest 仍存活的子进程等待其自然结束，不向其进程组发送终止信号。训练与评估子进程各自创建 session，避免 launcher 退出时连带中断已启动任务。

若 launcher 重启后发现训练 job 的 PID 已死亡且 run 尚未 complete，它只能在同一 `run_dir/checkpoints/last.pth` 存在时以 `--auto-resume` 续跑。resume contract 恢复 model、optimizer、scheduler、RNG、sampler 和 extension state，并拒绝任何 config/split/normalization drift；没有可验证 checkpoint 的 orphan 必须 fail closed，不得通过 trainer 的 unique-run fallback 偷换 run identity。已失败的 evaluation 只有用户显式指定 retry flag 时才回到 planned；失败日志与 manifest retry history 保留，不覆盖为成功。

### 8. DeepSense6G 是独立的次验证，不与 MMW 合并

`deepsense6g_twc_secondary_v1` 将 Scene31--34 的公开 train/test CSV 的只读、内容寻址派生版本分别拼接为一个 pooled train dataset 和一个 pooled test dataset；每个 method/seed 只训练一次并发布一个固定 40 epoch `last.pth`，总训练矩阵为五方法×三 seeds=15 个任务。GPS scaler 必须仅在四场景合并训练集上统一拟合并供合并测试集复用；训练按完整合并样本集采样，不把 scene 当成四个独立模型。派生过程只保留 `future_beam1` 恰有 64 个有限非负功率值的样本，记录原 CSV/派生 CSV hash、原始/有效/剔除计数和被剔除行摘要 hash；不对 NaN 插值，也不读取 `num_pred=1` 之外的 future horizon。方法固定为 T2、MaskTrain-CLS、AMBER-Full、RMBP-MM、AMR-Net-4M-Adapted；S1 与九项 MMW 消融不复制到该数据集。主结果在 pooled test 上一次评估和跨 seed 汇总，scene breakdown 仅可作为同一 checkpoint 的附加诊断；任何 MMW split、normalizer、mask cache 或 summary row 都不得混入。RMBP 的本地 window5 公平适配必须在 fidelity 表中说明，必要时另做 window2 sensitivity，但不让 native-window 行替代公平主行。
DeepSense6G 原始 CSV 中形如 `/unitN/...` 的资源引用按场景 `data_root` 解释为 dataset-relative 路径；loader 必须在读取入口规范化该前导 slash，不能把它当作容器文件系统绝对路径。

### 9. 无线、机制、corruption 与复杂度证据优先复用 checkpoint

MMW evaluator 从 `future_beam1` 读取完整 beam-power vector，输出 predicted-beam normalized gain、gain loss dB 和 SNR 0/10/20 dB 下的 spectral-efficiency ratio/rate loss。机制分析只消费固定 prediction/prototype/router diagnostics，输出物理 beam-error CDF、far-error rate、clean-to-missing drift、prototype neighbor margin 和 router-weight/oracle-quality alignment。`mmw_twc_corruption_v1` 以固定 cache 对 GPS 物理坐标噪声、图像遮挡/模糊、Radar 噪声和 LiDAR 稀疏化做推理期压力测试；它不得修改训练 recipe，且默认不进入 post-hoc 队列。只有用户明确传入 `--run-reliability-stress` 时，launcher 才生成包含 corruption shards 的 manifest 并执行；manifest 必须记录 opt-in 身份，默认 complexity-only 队列不得产生 reliability claim。复杂度表固定硬件、warmup、batch1/batch64 和 AMP policy，记录 params、MACs（若现有 profiler 可用）、latency、throughput、peak memory。

### 10. 状态账本由产物生成而非手工猜测

`outputs/twc_experiment_status.md` 由 manifests、run_status、checkpoint、evaluation provenance 和当前 GPU PID 生成。每行必须显示 protocol、method/variant、scene/domain scope、seed、status、GPU/PID、config/run/eval 路径和 claim eligibility；未知或 stale 状态明确标记，不把文件存在误判为完成。

### 11. Router oracle 修复先走 inner-validation 开发筛选

seed1 outer 诊断显示 current `_router_oracle_loss` 在多个模态具有相同最小 beam 距离时直接 `argmin`，从而按固定模态顺序选择 Image；这与设计图中的 confidence tie-break 不一致，也会把 learned Router 推向单模态。该问题不得直接用已消费 outer evidence 调参。

新增 `mmw_tie_aware_router_screen_v1`，只读取 `mmw_twc_outer_v1` 已冻结的 `inner_train` 与 `inner_validation`，不得读取 `outer_evidence`。筛选固定 seed1、40 epoch、batch64、H4、RouterNoPattern、相同外部训练 mask schedule 和 `last.pth`，仅改变 Router oracle target 或将 learned Router 替换为现有 Uniform control。八个并行候选为：

- `HardFirstControl`：保留现有 hard-first `argmin`，用于复现偏置基线；
- `HardConfidenceTie`：最小距离并列时，以 unimodal 对真实 beam 的概率破同分；
- `SoftUniformTie`：把等概率质量分配给所有最小距离并列模态；
- `SoftConfidenceTie`：只在最小距离并列模态中，按真实 beam 概率归一化 soft target；
- `DistanceSoftT05` 与 `DistanceSoftT10`：按 circular beam distance 的负值和温度构造可用模态 soft target；
- `DistanceConfidenceT10`：distance-soft score 加上真实 beam log-probability；
- `UniformFusion`：现有 availability-normalized uniform fusion，不训练 Router，作为复杂度下界。

所有 tie-aware target 必须只在 available modalities 上归一化；soft target 使用 soft cross-entropy，hard target保持 cross-entropy。筛选输出必须记录 target mode、temperature、tie ratio、target entropy、split/hash、seed、GPU 和 claim-ineligible 状态。canonical T2 在用户审阅筛选结果前不得切换 oracle mode；若后续切换，现有 outer 结果必须降级为 development evidence，并重新规划独立确认边界。

筛选汇总必须复用同一批 fixed-mask prediction rows，以 ADBA 为主排序、Top-1 为支线，并保持 Clean、Drop1、Drop2、Drop3、Block80 的主单元定义不变。不得为了改变排序而重复模型推理或生成新的 mask；汇总必须输出逐域配对差异、覆盖审计和 `claim_eligible=false` provenance。

### 12. Oracle Gap Test 固定同一 checkpoint 并使用通信效用 Oracle

`mmw_router_oracle_gap_v1` 是用户显式授权的 inner-only 机制诊断，固定 `SoftConfidenceTie/seed1/checkpoints/last.pth`、15 个 frozen inner-validation domains、完整四模态 availability 和同一批 sample identities。它不得重新训练，也不得在 Uniform、Learned Router 与 Oracle 之间切换 checkpoint。Learned 使用 checkpoint 原生 supervised-router 权重；Uniform 对同一 checkpoint 的四个 unimodal logits 等权平均；Oracle 按每个样本四个 unimodal argmax beam 在 `future_beam1` 上的 normalized gain 选择效用最高模态，并输出该模态 logits。并列效用只影响 deterministic modality id，不影响 Oracle 通信效用。

压力条件为 Clean 加四类各三级退化，共 13 个 condition。图像采用 Random Erasing/Cutout 风格的 seeded 方形遮挡，面积比例为 10%/25%/40%，在已标准化图像空间填 0；Radar 对 RA/DA map 加按每样本信号标准差标定的 AWGN，目标 SNR 为 20/10/0 dB；LiDAR 对 BEV 的空间 cell 使用跨 channel 共享的 seeded Bernoulli dropout，保留率为 75%/50%/25%；GPS 将 checkpoint scaler 标准化的 `[range, sin(theta), cos(theta)]` 逆变换到相对 XY，加入二维独立零均值 Gaussian 位置噪声 `sigma=1/3/10 m`，再恢复极坐标特征并使用同一 scaler 标准化。四类退化分别受 Random Erasing、雷达 Gaussian thermal-noise/SNR 模型、ModelNet40-C/MSC-Bench density/incomplete-echo corruption 与 GNSS Gaussian observation-noise/urban multipath 文献支持；severity 数值是本 protocol 的预注册 stress grid，不声称精确复刻特定硬件。

每个 condition 必须保存 compressed NPZ trace：sample/domain identity、target、future beam powers、四个 unimodal logits、Learned Router 权重、Learned/Uniform/Oracle fused logits、Oracle modality 和每模态 normalized gain；manifest 同时冻结 checkpoint/config/split/scaler/source hash、corruption seed/参数/算法版本和参考文献。三个分支均报告 circular progressive Top-3 ADBA（主）、Top-1（辅）、normalized gain 和 0/10/20 dB spectral-efficiency ratio。Router oracle regret 主定义为 `best unimodal normalized gain - sum(router_weight * unimodal normalized gain)`，同时报告 argmax-router selection regret；normalized gap closure 定义为 `(Learned-Uniform)/(Oracle-Uniform)`，分母非正时标为不可解释而不得强制归零。

每一受损模态的权重响应必须以相同 sample identities 配对检查 Clean/S1/S2/S3：报告各级 mean weight、相邻级差、Spearman severity correlation 和逐样本 `w_clean >= w_s1 >= w_s2 >= w_s3` 比例。权重没有单调下降，或 Learned 未缩小 Uniform-to-Oracle gap，均是有效负结果；不得通过改 severity、换 checkpoint 或删除 domain 修饰结论。GPU0--7 只用于并行 condition shards，且不得终止其他进程。

### 13. Router utility alignment 与 paired monotonic 只做 inner seed1 开发筛选

Oracle Gap Test 显示 current Router 在 13/13 conditions 的 normalized gain 高于同 checkpoint Uniform，但只关闭 22.8%--45.2% 的 Uniform-to-best-unimodal gap；Image、Radar、LiDAR 平均权重随质量下降，GPS 在 `sigma=10 m` 下 utility 下降但 confidence 与权重反升。`mmw_router_utility_screen_v1` 因此只在 frozen inner-train/inner-validation 上修改 Router 训练，不消费 outer-evidence，也不改写 canonical T2。

MMW dataset source CSV、传感器文件和 beam-power 文件保持只读。只有显式设置 `include_router_utility_targets=true` 的开发 config 才在 `__getitem__` 加载 64 维 future beam power，并随 train-fit GPS scaler 输出 mean/scale；这些字段只用于 training auxiliary loss，推理不需要。对每个 available modality，以 detached unimodal argmax beam 的 `P(pred)/max(P)` 为 utility，并在 available modalities 上以温度 softmax 构造 Router target。该 full power vector 是 training-only privileged supervision，必须在论文和 provenance 中显式披露。

paired variant 复用 T2 已有 full-modal no-grad superset forward 作为 clean view，并以训练 seed `20260719`、`(epoch+step) mod 12` 在 Image/Radar/LiDAR/GPS × severity1/2/3 间轮换，只在线生成一个 corrupted full-modal view；源数据不复制。corrupted encoder/unimodal forward 在 no-grad 下运行，随后仅用 detached reliability features 重新计算可训练 Router logits，避免第三套 encoder graph。paired utility loss作用于 corrupted Router。monotonic loss只在受损模态 utility drop `delta_u > 0.01` 时启用：`relu(w_corrupt - stopgrad(w_clean) + 0.25*delta_u)`，不得在实际 utility 未下降时强迫权重变化。

八个 seed1 候选固定为：`CurrentControl`（SoftConfidenceTie）；`UtilityT01/UtilityT02`（仅主 forward beam-power utility，温度 0.1/0.2）；`PairedT01/PairedT02`（增加 corrupted utility，paired weight 0.1）；`FullMono002/005/010`（温度 0.1、paired weight 0.1、monotonic weight 0.02/0.05/0.1）。全部固定 H4、RouterNoPattern、40 epoch、batch64、seed1 和相同 missing schedule，分别绑定 GPU0--7。训练完成后以独立 evaluation corruption seed `20260718` 运行同一 15-domain Oracle Gap protocol；按 ADBA 主、normalized gain/rate ratio/Oracle regret/四模态单调性机制指标选择。用户确认前 seed2--5 MUST 保持未提交。

### 14. Expected-utility Router 修复只做 inner seed1 fail-closed 筛选

`mmw_router_utility_screen_v1` 是有效负结果：argmax-selected-beam utility 在大多数 clean/corrupted 配对中分段常数，实际 `router_pair_active_ratio=0`、monotonic loss 为 0，且 paired target entropy 接近 `ln(4)`；因此不能据此声称 Router 与通信效用对齐。v2 虽修复为 continuous expected utility，但 batch64 smoke 与 epoch11 中间诊断显示随机初始化和早期模型的 paired target entropy 仍约为 `1.38627`，固定 epoch10 warm-up 不能阻止无信息 CE 把 Router 推向 Uniform；v2 因而提前终止并保留为负诊断。v3 不改 corruption 数据集，也不把 outer evidence 用于选择；它只在 expected target 足够有信息时启用 paired CE。

对单模态 logits `z_m`、beam-power vector `P` 和正温度 `T_beam`，连续 utility 定义为 `u_m=sum_b softmax(stopgrad(z_m)/T_beam)_b * P_b/max_j(P_j)`。`P`、`z_m` 与由其构造的 Router target 全部 stop-gradient，梯度只流入 paired Router logits。clean/corrupted monotonic gate 使用同一 expected utility，默认 `T_beam=0.5`、`delta_u>0.001`；主 forward 默认仍使用 `soft_confidence_tie`，避免把 v1 已失败的 main utility target 强制绑定到 paired branch。允许一个显式 expected-main control，但不得将其与 paired contract 混淆。

正式训练前，preflight 必须从固定 `CurrentControl` inner Oracle-Gap traces 读取相同 sample identities，覆盖 Image/Radar/GPS/LiDAR 各三级退化，记录每格 mean/quantile utility drop 与 `epsilon` active ratio，并验证四种模态都至少在一个 severity 产生非零 active samples；还必须记录成熟 checkpoint 在 target entropy 1.2/1.3 gate 下的 13-condition coverage。focused test 必须证明 argmax 不变时 expected utility 仍会变化、近均匀 target 不产生 paired CE 梯度、信息性 target 使 paired Router 参数获得有限非零梯度。任一条件不满足时 launcher fail closed，不生成训练子进程。

v3 八个 seed1 候选固定为：`CurrentControl`；`ExpectedMainT01`；`PairUngated`；`PairEntropy120`、`PairEntropy130`；`MonoW002`、`MonoW005`、`MonoW010`。全部共享 H4、RouterNoPattern、40 epoch、batch64、相同 external missing schedule、训练 corruption seed `20260719` 与评估 corruption seed `20260718`，分别绑定 GPU0--7。除 `ExpectedMainT01` 外，主 Router target 均为 `soft_confidence_tie`；`PairUngated` 保留 epoch10 固定 warm-up 负对照，其余 paired candidates 从 epoch index 0 构造 pair，但 paired utility CE 只在逐样本 target entropy 不超过 1.2/1.3 时激活。paired candidates 使用 `T_beam=0.5`、utility target temperature 0.1，monotonic 主组使用 `epsilon=0.001` 与 entropy 1.3 gate。评估仍为完整 15-domain Oracle Gap，按 corrupted normalized gain、gap closure、soft oracle regret 与四传感器权重响应联合筛选；seed2--5 和 canonical T2 变更必须等待用户复核。

### 15. Joint Drop+Corrupt 先诊断现有窗口级 Router，再决定是否改结构

`mmw_router_joint_stress_v1` 继续固定 CurrentControl seed1、frozen inner-validation 15 domains、40 epoch `last.pth` 和 `claim_eligible=false`，不训练新 checkpoint。每个 5×4 modality-frame cell 只允许 `clean`、`drop`、`corrupt` 三个互斥状态：drop 令 temporal mask 为 false，corrupt 保持可用并按模态施加既有 S2 算法（Image 25% 遮挡、Radar 10 dB AWGN、LiDAR 50% cell 保留、GPS XY sigma=3 m），clean 不变。Joint20/40/60/80 每个 mask 的 `(clean, drop, corrupt)` 数分别为 `(16,2,2)`、`(12,4,4)`、`(8,6,6)`、`(4,8,8)`。

inner seed1 gate 每级使用 20 个固定、seeded、随机化 Latin-balanced masks；每个 mask 保持精确三状态 cardinality，跨 20-mask panel 每一个 modality-frame cell 的 drop/corrupt 次数分别等于该级每 mask 的 drop/corrupt 数。公平性只约束 panel 边际，不强制单 mask 四模态同质，以保留 Router 所需的样本级质量差异。cache 必须记录 canonical state matrix、state/count audit、seed、generator、digest 和 checksum；所有分支复用同一 sample、target、beam power 和 cache identity。

Learned 使用 checkpoint 原生四个窗口级 Router 权重；Uniform 只在至少一个非-drop cell 的模态间等权聚合相同 unimodal logits；Oracle 只在这些可用模态中按 predicted-beam normalized gain 选最佳单模态。评估保存 availability、5×4 temporal mask、三状态矩阵、unimodal/fused logits、Router 权重和三分支指标，并按 8 个 condition shards 绑定 GPU0--7。summary 以 ADBA、normalized gain、rate ratio、Learned-Uniform paired delta、mask/domain bootstrap CI、win rate、gap closure 和 regret 判断。

该诊断不以复杂度保证正结果，也不直接修改 canonical T2。只有 Learned 在 Joint40/60/80 的 ADBA 或 normalized gain 对 Uniform 无稳定正差，且失败集中于同一模态内部 clean/corrupt 混合时，才新增独立 R1/R2 change：R1 只向窗口级 Router 增加可观测 temporal quality statistics；R2 使用 temporal-quality gate 与现有 modality Router 的 factorized hierarchical weights。不得把人工 corruption state label 作为推理输入，也不得直接引入 flat 20-cell softmax Router。

预注册 Gate 完成后另做不改变 Gate 的 static-prior falsification。`GlobalCleanPrior` 由 15 域 Clean trace 的全部样本 Router 权重求均值并归一化，是部署时可固定使用的静态四模态权重；`PerSampleClean` 使用相同 sample identity 的 Clean Router 权重融合同一 stressed unimodal logits，只作为机制上界反事实。两者与 Dynamic、Uniform 必须复用已保存的 logits、availability、labels 和 beam powers，不重新前向或训练。报告 Joint20/40/60/80 的 ADBA、Top-1、normalized gain、rate ratio，以及 Joint40/60/80 的 paired-domain Dynamic-GlobalCleanPrior bootstrap CI。若 Dynamic 只优于 Uniform、却未稳定优于 GlobalCleanPrior，则预注册 Gate 仍按原规则记为 PASS，但论文不得把该结果解释为 corruption-adaptive reliability routing；只可主张 learned non-uniform fusion，除非后续独立 R1/R2 change 产生外部证据。

2026-07-19 的完整执行覆盖 8/8 shards、81/81 conditions、15 domains 和 1,215 traces。预注册 Gate 为 PASS：Joint40/60/80 的 Learned-Uniform ADBA 差分别为 `+0.0112/+0.0111/+0.0079`，normalized gain 差为 `+0.0224/+0.0253/+0.0245`，合并 paired-domain 95% CI 下界分别为 `+0.0067/+0.0190`。因此本 change 按预注册规则保留当前窗口级 Router，不自动启动 R1/R2。独立 static-prior falsification 同时显示 Dynamic-GlobalCleanPrior 合并 ADBA `-0.0040`（95% CI `[-0.0068,-0.0011]`），normalized gain `-0.0014`（95% CI `[-0.0061,+0.0030]`）；当前可主张 learned non-uniform fusion 优于 Uniform，但 corruption-adaptive reliability claim 不受支持。

### 16. DeepSense6G 先做同 checkpoint Router falsification

在恢复五方法三 seed 的正式 DeepSense6G 队列前，先复用现有 T2 seed1 epoch20 checkpoint 做一次低成本、`claim_eligible=false` 的 falsification。该诊断绑定 `deepsense6g_twc_secondary_v1` 的 Scene31--34 pooled test、统一 train-fit GPS scaler、同一 checkpoint SHA256 和同一组 sample identity；Scene31--32 作为 day breakdown，Scene33--34 作为 night breakdown，但四个 scene 仍是同一 pooled checkpoint 的分片，不能解释为四个独立训练任务。

评估 mask 包含 parent cache 的 Clean 与全部 Drop1/Drop2/Drop3 whole-modality 条件，以及独立固定的 Token20/40/60/80/90 modality-frame panel。每个 token rate 使用 10 个 seeded random K-of-20 masks，并通过最少 cell swap 使 20 个 modality-frame cells 的集合级保留次数完全相同；cache 记录 parent protocol SHA256、seed、canonical matrix、digest、平衡审计和 checksum。诊断不加入人工 corruption，先检验真实 day/night 与自然传感器差异能否产生动态收益。

每个 scene 只前向一次每个固定 condition，保存 sample id、target、future beam power、四个 unimodal logits、availability、Learned Router 权重与 Learned/Uniform/Oracle logits。`GlobalCleanPrior` 从四个 scene 的 Clean Router 权重全局平均后冻结，再离线融合所有 stressed traces；四个分支不得使用不同 checkpoint 或不同 logits。summary 报告 pooled、逐 scene、day/night 的 ADBA、Top-1、normalized gain、rate ratio、Router regret、权重和 Oracle modality frequency。epoch20、训练未完成与 claim-ineligible 身份必须显式保留；只有 Learned 对 GlobalCleanPrior 存在有意义的稳定优势且场景权重变化与 Oracle 模态变化方向一致，才值得恢复正式 40 epoch/multi-seed Router 证据。

## Risks / Trade-offs

- [从 old train 再划 confirmation fold 不能恢复真正 blind test] → 明确报告 64/16/20 角色计数、exclusion source 与 post-selection 边界；不把它伪装成原始官方或未接触 test。
- [GPU 边界与可用资源变化] → 默认只使用 GPU4--7，作业队列在单卡上顺序执行；只有用户显式授权并传入 `--allow-gpu0-3` 时才扩展至 GPU0--7。启动前检查可用显存，绝不终止或迁移其他进程。
- [runtime API 变更或单条 evaluator failure] → 用 `TaskForwardResult` 无 labels 的回归测试锁定 batch-label 路径；queue 只 fail-fast 于新提交，不中断已启动 work，并以显式 retry 与 strict resume contract 恢复。
- [五 seed 全矩阵无法一夜完成] → 作业 manifest 有状态、可恢复；优先完整主比较，余下消融排队，summary 只消费完整 cell。
- [ULA DFT 不足以证明全球物理圆环] → 只在 local-array codebook 层做证据；RSU yaw 不参与 label 更新，不用 GPS 相关性取代 codebook 审计。
- [baseline 与原文输入不一致] → provenance 和论文表格中固定标注 local adaptation、缺失的历史 beam/partial measurement/pretraining，不作 paper-equivalent claim。
- [mask 过度静态导致结论依赖单一模式] → cache 含多个固定、按类型平衡的 masks；报告每 mask 方差与 worst mask，不只报告合并均值。

## Migration Plan

1. 完成 P0：生成并审计 `mmw_twc_outer_v1` split manifest 与 immutable mask cache；运行 config/one-step smoke。
2. 完成 P1：生成 codebook topology audit，冻结其 descriptor；若不满足物理判据，锁定保守 claim wording 和 linear/label-space ablation 分支。
3. 完成 P2--P3：生成严格训练/评估 manifests，默认提交四方法五 seed 队列至 GPU4--7；在用户显式授权时可扩展至 GPU0--7。每个完成 run 自动进入 fixed-mask evaluation shard；单条失败不终止已启动作业，重启时只从 validated `last.pth` 续跑 orphan。
4. 完成 P4：按 matched-control queue 提交消融，并对完成的 full seed cells 输出 paired statistics。
5. 完成 P5：生成 summary、plots、protocol report 和论文数据表；只有 validate/identity/coverage 全通过的结果可进入 claim-facing 文档。

所有新产物均可通过删除 `outputs/mmw_twc_outer_v1` 和 `outputs/cache/mmw_twc_outer_v1` 回滚；源码 rollback 只撤销本 change 文件，不触碰历史结果。

## Open Questions

- 物理 ULA codebook endpoint 的 beampattern 判据将由 P1 用实际 metadata 和保存的 DFT convention 给出；在报告前不得预设其为 cyclic。
- baseline 40-epoch batch-64 是否都能在与 T2 相同显存预算下稳定完成；若某 baseline OOM，必须全方法一起使用新的共同 16 倍数 batch 重跑，而不是为该方法单独降 batch。
