## Context

历史 S1 在 `92012d5` 中先对每个模态做 `[B,T,D]` masked mean，再复用 supervised modality router；它在 Scene31-34 seed1 公平评估上的 Drop0/20/40/60/80 Top1 为 0.535797/0.520497/0.509238/0.490618/0.420179。该提前实现及 S1-S4 wrapper 已被删除，current U-Mask 会忽略 temporal metadata 并对 encoder 序列做普通 mean，因此历史配置不能在当前源码中直接复跑。

冻结 evidence ledger 为：config `outputs/temporal_router_s1_s4_v1/generated_configs/s1_temporalagg_modality_router_seed1.yaml` 的 SHA256 是 `1fb51902c80bc342e7396ea7af5ef575c2fe957d4f0eff6df38dbaf6467a4145`；`best_top1.pth` 的 SHA256 是 `162e9c6744fc367d8a0c647943ee8d0a9b2d80300d3df3b120246d729c2cde5a`；startup summary 记录 actual-module total/trainable params 为 25,568,996、排除两个未参与 downstream forward 的 TinyViT classifier head 后 effective params 为 11,547,074；seed1 run status 记录 40 epoch、best val acc 0.5357968@epoch8、总时长 13,985.873 秒。运行监控显示 batch64 单进程峰值 reserved 显存约 36.6 GiB，因此本 change 固定每张 A40 最多一个训练进程。

数据侧已经提供 `[B,T,M]` mask、零填充、分层 temporal sampler 和固定 eval cache；模型侧已经提供 supervised router、oracle target、prototype head 和 same-model online-full stabilization；H5/P1 workflow 已提供参数化 launcher/eval/summary 与 GPU 调度。本 change 只补齐这些现有 owner 之间缺失的最小语义，不新增模型注册名、训练循环或 wrapper suite。

## Goals / Non-Goals

**Goals:**

- 用新 `model.primary.temporal_pooling` opt-in 配置恢复历史 S1 的可比较语义，并继续拒绝旧 S1-S4 `temporal_router_type`。
- 独立消融 mask statistics、fixed recency、gap-aware residual pooling、temporal superset KD、beam monotonic ranking 和 coverage shrinkage。
- 保证 disabled 行为与 current U-Mask 完全兼容，`masked_mean` baseline 可由 synthetic test 精确刻画，gap pool 在残差门为零时精确退化为该 baseline。
- 复用现有 H5/P1 workflow，在 GPU0-7 上一卡一进程完成 seed1 筛选，并只为通过 guardrail 的候选补 seeds 1/2/3。

**Non-Goals:**

- 不恢复 S2 per-time、S3 two-level、S4 global-cell route 或已删除的 launcher/eval/summary wrapper。
- 不在本 change 实现 temporal curriculum、rate×type CVaR、missing token、外部 teacher、checkpoint teacher、feature L2 或独立 distiller。
- 不用 test split 调参，不把本地 checkpoint、日志、cache、summary 或未审查结果升级为正式 claim。

## Decisions

### 1. 以 U-Mask 内嵌 opt-in 行为表达 S1

使用 `model.primary.temporal_pooling` 字典，`enabled=false` 为默认；启用时只允许 `masked_mean`、`fixed_recency` 和 `gap_aware_residual`。模型显式消费现有 `modality_temporal_mask`、`temporal_mask` 和 `available_modalities`，先输出每模态 `[B,M,D]` 表示，再进入现有 `supervised_router`。这保持 whole-model exception、共享 batch/runtime 和当前 router owner，不新增第二个 MODELS 名称。

旧 `temporal_router_type=s1_temporalagg_modality|s2_pertime_modality|s3_two_level|s4_global` 继续 fail fast。选择新字段而不是恢复旧名称，是为了保持 post-C2 生命周期边界并防止 S2-S4 通过兼容 alias 回流。

### 2. 统一 mask statistics 定义

从 `[B,T,M]` bool mask 计算每模态五个归一化统计：coverage=`valid_count/T`；last-age=`(T-1-last_valid)/max(T-1,1)`；longest-gap=`最长连续缺失长度/T`；trailing-gap=`末尾连续缺失长度/T`；num-blocks=`缺失块数量/max(ceil(T/2),1)`。全缺失模态的 last-age 取 1，统计只描述历史输入，不读取 target/future。

`use_mask_statistics=true` 时将这五维附加到现有八维 router reliability features；关闭时 router shape 和 checkpoint 语义不变。统计 helper 同时供 pooling、router diagnostics 和测试使用，避免两份定义漂移。

### 3. Gap pool 保留 masked mean 残差锚点

对每个有效 cell 使用共享轻量 scorer：内容投影 `W_h h` 加该 cell 的 normalized relative age、距前一有效观测间隔和五维模态 mask statistics 投影 `W_s s`，经 `tanh` 和标量投影得到 masked softmax 权重。输出为 `mean + tanh(eta_m) * sum(alpha * (h-mean))`，每模态 `eta_m` 零初始化。缺失 cell 的权重严格为零，单 cell 模态输出该 cell，所有模态全空仍由已有 temporal fallback 或模型校验拒绝。

fixed recency 使用 `exp(-decay*age)` 的无参数 masked weighted mean，作为“最近观测是否足够”的低成本对照。T=5 不引入 Transformer、多头 attention 或新依赖。

### 4. Teacher 使用原始输入与可观测 superset mask

temporal operator 在显式 `preserve_unmasked_for_superset=true` 时，在零填充前只保存原 tensor 引用和已有有效性 base mask；student batch 仍按 sampled partial mask 零填充，满足当前数据契约。training extension 构造 teacher batch 时恢复这些原 tensor，并使用 base mask 作为 `M+`，student 使用 sampled `M-`，保证 `M- subseteq M+`、同 sample/target、无 future cell。

teacher forward 在 `no_grad` 与临时 eval mode 下运行，结束后恢复 primary model 原训练状态。未启用 superset loss 时不保存原输入、不增加第二次 forward。该机制属于同一 primary model 的在线 stop-gradient consistency，不构建 frozen/checkpoint teacher，也不恢复 distiller registry。

### 5. KD 与 ranking 共享一次 superset forward

confidence-gated KL 使用温度 `T`，每样本权重为 teacher 正确指示乘以 `1-H(p+)/log(C)`；权重 stop-gradient，分母按有效权重归一。beam ranking 使用 64 类 circular distance 的期望风险，并在 superset teacher stop-gradient 的约束下优化 `relu(R(p-) - R(p+) - tolerance)`，即只在 partial student 风险超过 teacher 允许容差时降低 student 风险。附件原式 `relu(R(p+) - R(p-) + margin)` 只适合作为“superset 是否更差”的只读判别；若在 teacher stop-gradient 时直接优化，它会反向增大学生风险，因此本 change 不把该原式作为 loss。二者可独立开启或联合开启，并共享一次 teacher forward；feature KD 默认且本 change profile 中固定为零。

metrics 至少记录 raw/weighted KD、gate mean/active ratio、teacher/student Top1、ranking loss、`student_risk-teacher_risk`、partial excess violation rate 和只读 superset-worse rate。eval 不执行 teacher branch。

### 6. Coverage shrinkage 只改变可用模态权重

在 supervised router masked softmax 后计算可用模态均匀先验 `u`，再以 `w'=(1-rho)w+rho*u` 收缩。`rho` 由平均 coverage、gate entropy 和 top1-top2 gate margin 的小 MLP 给出，并乘 `1-mean_coverage`；因此完整输入 `rho=0`，不可用模态保持 0，单模态输入保持权重 1。该行为显式 opt-in，并输出 rho/coverage/entropy/margin diagnostics。

### 7. 只扩展现有 H5/P1 workflow

launcher 增加 `s1_lightweight` profile，但保留现有默认五方法。profile 的 seed1 首轮固定八个独立任务：S1 masked mean、T2 gated KD、T1 beam rank、A1 mask stats、A2 fixed recency、A3 gap residual、T1+T2、J1 gap+T1+T2。默认 GPU 列表为 0-7、`max_jobs=8`、`per_gpu=1`；每个任务使用独立 output dir 和 log。

首轮使用相同 split、epoch、optimizer、mask sampler 和 deterministic eval mask。为利用八卡，T1+T2 与 J1 可作为预计算组合与单组件并行运行，但只有其组成单项正收益后才有资格晋级；选择顺序为五档 mean Top1、Drop0-60 mean Top1、Drop80 Top1，且 Drop0 相对 S1 降幅不得超过 0.005。只有合资格 J1 通过 guardrail 后才运行 J2；真实运行产物保持 ignored。

### 8. S1 profile 使用实测吞吐资源参数

GPU0-7 一卡一进程的 1-epoch smoke 显示 batch64 时 DataLoader `data_time` 稳态为 0-1ms，但主进程 temporal zero-fill 与 batch 处理受单 CPU 线程限制。固定 3,840 个训练样本的并行基准中，S1/T2 在 1 线程时分别为 23.82/20.04 samples/s；12 线程时为 48.16/37.10 samples/s。16 线程使 S1 回退到 45.53 samples/s；batch72 对 T2 回退到 31.72 samples/s；batch80 虽有小幅吞吐增益，但 reserved 显存约 45.0GiB，不满足 46GiB A40 的运行余量要求。

因此 S1 lightweight profile 默认保持 batch64、`num_workers=4`、`prefetch_factor=2` 和 pin memory，并使用 intra-op 12、inter-op 1 与 persistent workers。默认 H5/P1 profile 继续使用 intra-op 1 和 non-persistent workers。所有值仍可由 launcher 参数显式覆盖；不增加 NUMA 自动探测或新 benchmark wrapper。

## Risks / Trade-offs

- [Risk] current 重实现无法复现历史 S1 数值。→ 先用 synthetic exact-equivalence test 和 seed1 baseline characterization；所有新候选只与同源码、同协议的新 baseline 比较。
- [Risk] 保存原 tensor 引用导致意外内存复制。→ operator 只在 opt-in 时保存引用，不 clone；测试验证 data pointer，并在 batch 结束后随 batch 释放。
- [Risk] teacher eval mode 与 student train mode引入分支差异。→ teacher 仅提供 stop-gradient软约束，记录 teacher/student gap；门控排除错误和高熵 teacher。
- [Risk] 二次 forward 使训练接近两倍耗时。→ T1/T2 共享 forward，首轮每卡一进程；推理无额外 forward。
- [Risk] gap scorer 过拟合 mask pattern。→ 残差门零初始化、参数预算小于 0.03M、A1/A2 对照和 Drop0 guardrail共同约束。
- [Risk] router shrinkage 在中等缺失下稀释正确路由。→ 完整 coverage 硬置零，只在 J1 通过后测试，并通过 `rho_max` 限幅。
- [Risk] 8 个作业同时读取数据造成 I/O 压力。→ 延续每卡一进程、线程上限和现有 cache；不使用每卡双进程。
- [Risk] 12 个 intra-op 线程在较小机器或其它并发负载下争用 CPU。→ 仅作为 S1 profile 默认值，保留 CLI 覆盖；默认 profile 不变，正式 manifest 记录实际线程与 persistent worker 配置。

## Migration Plan

1. 建立 S1 baseline characterization、mask statistics、pooling、superset teacher 和 loss focused tests。
2. 实现新 opt-in 字段；默认 config 和旧 S1-S4 rejection tests 必须继续通过。
3. 扩展现有 H5/P1 launcher/eval/summary，并以 dry-run 验证 GPU0-7、独立路径和八任务矩阵。
4. 运行 1-epoch smoke；失败时删除新 ignored smoke root即可回滚，不触碰历史 outputs。
5. 运行 seed1 筛选和固定 mask eval；按 guardrail 选择后续 seeds/J2，不修改测试集或历史 checkpoint。
6. 源码回滚只需关闭/删除新 opt-in 实现；`enabled=false` 路径没有 checkpoint shape 变化。

## Open Questions

- J2 是否进入多 seed 正式矩阵由 J1 seed1 的 Drop0 guardrail、Drop80 和 router regret 共同决定，不在实现前预设结论。
- claim-grade clustered paired bootstrap 与结果入账在本地 screening 产生稳定 per-sample evidence 后另行决定；本 change 不自动推广 claim。
