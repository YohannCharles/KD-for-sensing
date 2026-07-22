## Context

上一轮 `add-beam-conditioned-feature-fusion-search` 已生成经过身份和 parity 校验的三 split、六分片 frozen cache，并训练得到 F1 feature concat MLP validation-best checkpoint。F1 的 `FeatureTokenAdapter` 把 time-major 的四模态原始 frozen feature 转为 20 个 256 维 token，随后由 concat MLP、output projection 与 `BeamPrototypeBank` 产生 64-beam logits。本轮只允许在缺失输入上修改仍可用的 F1 token，Full 必须物理绕过新增模块。

仓库 canonical token 模态顺序是 `image, radar, gps, lidar`，而研究 condition 向量定义为 `image, lidar, radar, gps`。实现必须显式转换两种索引，不能改写 checkpoint 或 cache 的既有 time-major 语义。

## Goals / Non-Goals

**Goals:**

- 以同一 F1 checkpoint、token cache、split、seed、batch/mask schedule、optimizer 和 checkpoint selection 公平比较 U0--U5。
- 精确枚举并完整评测 4 个 single、6 个 double、4 个 triple missing pattern。
- 通过 Full bypass、missing-token invariance、teacher/aux isolation 和禁止字段检查保持实验边界。
- 产出可恢复、可审计、失败隔离的 GPU0--5 inner-only 快速实验和预注册诊断。

**Non-Goals:**

- 不解冻 encoder、F1 token adapter、fusion MLP、output projection、prototype 或 temperature。
- 不生成缺失 token，不使用 Router、Transformer、MoE、reconstruction、full residual、历史 beam、channel/path/power、label/weather/scene 输入。
- 不运行 outer test、multi-seed、端到端或下一轮实验，不修改 canonical recipe、公共 CLI 或正式 claim。

## Decisions

### 1. 从已验证 cache 确定性派生 F1 token cache

预计算入口读取 feature-fusion cache manifest 和 F1 best checkpoint，校验各自 SHA、F1 variant、split coverage、样本唯一性与禁止字段；以 F1 eval-mode `FeatureTokenAdapter` 生成 float16 `[N,20,256]` token，并用冻结 F1 fusion/head 生成 float32 full logits。merge 阶段重新计算 Full 与四个 single-missing 在线/缓存 parity，未达到 0.999/0.995 Top1 agreement 时拒绝训练。

替代方案是重新运行四个 encoder。它不会增加本轮需要的信息，却扩大运行时间和数据面；只有已有 cache 身份失效时才重建上游 cache。

### 2. 一个冻结 F1 owner 包裹五种最小 adapter

新增非 registry 的 fallback 组件统一接收 F1 tokens、四维 availability 和可选诊断 replacement。U0 直接调用冻结 F1；U1 用 14-pattern 参数 bank；U2 用一个 29 维 mask encoder 和共享 hypernetwork；U3--U5 共用四个 modality-specific contextual residual adapter。U4/U5 具有完全相同的 auxiliary head 结构，U5 只增加冻结单模态 teacher 的 KL 监督。

Full 在构造 mask embedding 或 adapter 前直接返回冻结 F1 logits并保持 forward count 为零；all-missing 在任何预测前报错。missing token 在适配前后均显式置零，任何缺失位置数值变化不得影响输出。

### 3. 固定两级均衡 mask schedule 与 group-balanced selection

训练 schedule 先等概率选择 missing count 1/2/3，再在组内均匀选择具体 pattern，并以 sample identity、epoch、seed 固化为 JSON；所有 U1--U5 复用相同 sample、mask、batch 顺序。validation manifest 对 14 个 pattern 逐一完整评测，checkpoint 仅按 single/double/triple 三组 loss 的宏平均选择。Full 不训练也不参与选择。

每样本先计算 fused、available-modality auxiliary 与 KD loss，再按 batch 中 pattern 先组内平均、后 pattern 平均，避免可用模态数或样本数改变权重。lambda 只使用固定 train batches 校准一次。

### 4. Teacher 与诊断复用同一冻结 prototype

四个 teacher 和匹配 probe 只读取对应模态的 5 个原始 F1 token，以轻量 temporal MLP 投影到冻结 prototype 维度；validation 单模态 loss 选择 checkpoint。U4/U5 auxiliary head 同样只读取对应当前可用模态，并先按样本对可用模态平均。teacher logits stop-gradient，teacher feature不进入 student。

U3--U5 在同一 checkpoint 上评测 normal、train-pattern mean delta、zero delta 与 pattern 内跨 sample shuffle delta；表示、aux/teacher、模态内容 shuffle、weather、sector、error-distance 和效率均由统一 evaluator 从 eval split 生成，不参与调参。

### 5. 编排分成 prepare、teacher、U0--U5 三个显式阶段

launcher 先记录 `nvidia-smi`，完成 cache/preflight 后在 GPU0--3 训练四个 teacher；成功后固定 U0--U5 到 GPU0--5，分别保存 PID、状态、日志、resolved config 与退出码。失败任务不终止 sibling，也不自动重跑或调参；汇总只读取成功且 provenance 一致的运行目录并在唯一推荐方向后停止。

## Risks / Trade-offs

- [Risk] condition 顺序与 F1 token 顺序不同会造成模态串位。→ 用两个命名常量和显式转换函数，并以 14-pattern/block-mask 测试锁定映射。
- [Risk] float16 token cache 引入 logits 差异。→ logits 保持 float32，merge 执行 Full/四 single parity gate，失败即停止。
- [Risk] U1 参数随组合数指数增长。→ 只作为 combination-specific upper-bound control，报告 `2^M-2` 扩展趋势，不作为默认推荐。
- [Risk] U3 可能仅学习 pattern 偏置。→ 强制 mean/zero/shuffle replacement 与表示诊断，机制 gate 不使用辅助指标替代 fused 改善。
- [Risk] frozen F1 fusion 可能无法利用新 token。→ U4/U5 结构相同并比较 aux 到 fused 的转化；若失败，报告冻结 fusion 瓶颈而不继续扩展。
- [Risk] 本地 single-seed 结果不具正式 claim 资格。→ 所有 manifest/report 标记 inner-only、outer-test=false、claim-ineligible，并禁止自动进入下一轮。
