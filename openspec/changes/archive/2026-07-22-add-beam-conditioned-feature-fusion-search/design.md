## Context

C0-static validation-best checkpoint 已由 PGCD/PR-SQDF 本地流程验证，其 frozen cache 保存同一四模态五时间步样本的 pre-prototype feature、prototype feature、block logits、availability、在线 fused logits、固定全局 prior 和 split provenance。该缓存已完成 inner train/validation/development-eval 全覆盖与在线推理 parity，因此本轮不需要再次运行 backbone；需要把其中 clean frozen evidence 转换为更小的 feature-fusion cache，并在转换后重新执行 F0 full/单模态缺失 parity gate。

当前模态顺序由 `MODALITY_ORDER` 固定为 image、radar、gps、lidar，token 采用 time-major `[T,M]` 展平。四个 pre-prototype 宽度不同，缓存以最大宽度 padding 并显式保存实际宽度；adapter 只能读取对应模态的有效切片。

## Goals / Non-Goals

**Goals:**

- 在相同 frozen features、prototype bank、split、seed、mask schedule、训练预算和 checkpoint selection 下比较 F0--F5。
- 验证 feature-level interaction、Transformer、64-query、prototype-query 和 balanced evidence 对 Full 与 Missing LiDAR 的影响。
- 提供严格 mask、query 来源、prototype 冻结、输入敏感性、attention beam-specificity、模态 shuffle、天气/sector 和成本诊断。
- 产出可恢复、可审计、单任务隔离的 GPU0--5 本地筛选结果。

**Non-Goals:**

- 不解冻 encoder、temporal extractor、prototype bank 或原 block head。
- 不加入动态 Router、quality/reconstruction/residual recovery、channel/path/power 输入或历史 beam 输入。
- 不运行 outer test、multi-seed、端到端训练，不修改 canonical recipe 或正式 claim。

## Decisions

### 1. 复用已验证 frozen cache，并以 checkpoint SHA fail closed

预处理入口读取现有 C0 cache manifest，校验 checkpoint/config SHA、claim/outer-test 标记、禁止字段、split 覆盖、样本唯一性和源文件 SHA；然后按六个互斥 shard 写入 `outputs/feature_fusion_quick_search/cache/`。每个 shard 使用无 pickle 的 NPZ，feature 为 float16，block/base logits 与 prior 为 float32，mask 为 bool，标签/索引为紧凑整数。

替代方案是重新执行 C0 backbone。它不能增加本轮信息，却增加约六次重复 I/O/forward 成本；只有源 cache 缺失或身份不匹配时才应重新生成上游 C0 cache，而不是在本流程静默换 checkpoint。

### 2. 独立轻量模型组件，不扩展公共 registry

新增非 registry 的 feature-fusion 组件，统一接受 padded pre-prototype feature、实际宽度、modality/time id、availability 和冻结 prototype。公共 T2 forward、package CLI 与 canonical YAML 不增加分支。

四个 modality adapter 均为 `LayerNorm(D_m) -> Linear(D_m,256)`，再加 modality/time embedding；mask 在 attention 前后均应用，all-missing 直接报错。F1 使用小 MLP；F2 使用两层 pre-norm masked self-attention 与单 fusion token；F3/F4/F5 共用两层 pre-norm cross-attention decoder 和 shared scalar score，唯一 query 差别分别为 learned parameters 与 frozen prototype 的共享投影。

### 3. 参数公平由同宽主干保证并显式报告

F1 hidden 取 192，使其与两层 attention 候选处于同一参数量级。F2--F5 固定 `d_model=256`、4 heads、FFN 512、2 layers、dropout 0.1；F3--F5 同时保留同形 auxiliary projection，使 F4/F5 模型结构完全相同，F5 只改变训练 loss。报告 total/trainable/fusion params 与相对 F2 比例，超过预注册 15% 时标记不公平而不隐藏。

### 4. 训练 mask 与 validation selection 预先冻结

F1--F5 以 sample identity 和固定 seed 在 full、missing image/lidar/radar/gps、既有 S3 之间采样同一结构化 mask；F5 仅增加可用模态 auxiliary evidence loss。topology 与 auxiliary lambda 只可用固定 train batches 校准一次，validation 不参与调参。checkpoint 只按 `0.5 * full validation loss + 0.5 * fixed masked-macro validation loss` 选择。

### 5. 评测只消费 development eval clean evidence

统一评测从同一 clean development eval cache构造 Full、四个 single-missing 和固定 S0--S5 mask，输出 Top1/3/5、Within-3、MAE、weather、8-sector、LiDAR drop 与跨 sample token shuffle。F2--F5 同时输出 attention entropy、missing leakage；F3--F5 输出 query pairwise JS、neighbor/far similarity 和 topology Spearman。attention 只称为 interaction，不命名或解释为 reliability。

### 6. 编排只启动显式六任务并在汇总后停止

GPU 脚本先记录 `nvidia-smi`，固定 F0--F5 到 GPU0--5，分别保存 PID、日志、resolved config 与状态；等待所有任务并保留各自退出码。它不杀进程、不自动改超参数、不自动重跑、不启动下一轮。汇总器只有在 cache parity gate 通过后运行，并根据预注册阈值给出唯一推荐方向。

## Risks / Trade-offs

- [Risk] pre-prototype feature 以不同宽度 padding，错误切片会造成模态串位。→ adapter 按 manifest 宽度切片，并用 time/modality index focused tests 锁定 time-major 顺序。
- [Risk] float16 cache 引入小量 logits 差异。→ block/base logits 转存 float32，F0 parity 同时检查 max/mean error 和 Top1/Top3/Within-3 agreement；低于门槛时禁止训练。
- [Risk] query attention 可能退化成相同 pooling。→ 报告 pairwise JS、beam variance、neighbor/far similarity、输入 shuffle 与 zero-token 测试，不以 attention 可视化替代指标。
- [Risk] frozen-cache 结论不能代表端到端可训练性。→ 所有产物标记 single-seed、inner/development、claim-ineligible，并在本轮结束后停止。
- [Risk] GPU0 已有非零占用。→ launcher 只记录状态并在可用显存不足时让对应任务明确失败，不终止或迁移其他进程。
