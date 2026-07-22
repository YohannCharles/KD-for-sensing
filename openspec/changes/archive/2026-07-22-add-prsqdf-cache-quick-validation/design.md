## Context

C0 `qv_c0_corrupt_global_prior` 已有 validation-best checkpoint、固定 inner split、train-fit GPS normalization、`[B,5,4,64]` block latent/prototype logits 和 `[20]` learned global prior。模型的 `return_intermediates` 可取得 `F_block/F_proto/Z_block`，现有 prototype-collapse 诊断已经验证可用 encoder 最后投影层的 input hook 取得真实 `F_preproj`：image/lidar 为 320 维，radar/gps 为 64 维。`SensorDegradationGenerator` 已支持所需 seen/unseen/missing/stale 退化并以稳定 sample identity 派生随机流。

本轮不是新主模型训练，而是冻结 C0 后对质量监督和输入空间做 claim-ineligible 快筛。所有 Q1--Q5 必须共享同一缓存和训练协议；缓存、checkpoint、日志与报告均位于 ignored `outputs/prsqdf_quick_search/`，不能成为 package import 或 canonical config 的依赖。

## Goals / Non-Goals

**Goals:**

- 一次运行冻结 C0 backbone，生成可审计、分片、无信道字段的 clean/corrupted block cache，并验证 float16 cache fusion 对原模型推理的复现误差。
- 用同一个小型 quality head 骨架比较 output statistics、pre-prototype feature、severity、hard CE risk、topology risk 和 sensor statistics/ranking。
- 保持 C0 global prior，只允许非负有上界 beta 对高预测风险块做减法修正，严格满足 missing 权重为零和可用权重和为一。
- 在 E0--E6、D0--D3、质量相关性、单调性、梯度冲突和效率指标上完成 Q0--Q5 同口径汇总并应用预注册 gates。

**Non-Goals:**

- 不读取 channel、CSI、channel gain、beam power、ray-tracing path 或任何派生输入/质量标签。
- 不解冻或重训 semantic backbone，不修改 prototype loss，不增加 MoE、恢复网络、第三方依赖或 public CLI。
- 不运行 outer test、multi-seed 或下一轮完整训练，不更新 canonical recipe 或正式 claim。

## Decisions

### 1. PR-SQDF 作为独立 cached-quality 组件，不接入默认 U-Mask forward

新增 `PRSQualityHead` 和 `bounded_prior_correction` owner。quality head 接受 padded `F_preproj` 及每模态真实宽度，使用四个小 input adapter、共享 trunk、modality/time embedding 和可选 output/sensor statistics；输出经 Softplus 保证非负。这样 checkpoint 只包含 quality 参数和 beta，C0 state dict 与 canonical forward 完全不变。

备选是在 `U-MaskBeamJEPA` 增加第二个 opt-in Router，但这会再次把缓存实验和 backbone forward 绑定，并复制 PGCD extension 的生命周期，因此不采用。

### 2. clean 数据只保存一次，corrupted view 引用同一 sample identity

每个 split/shard 生成一个 clean `.npz` 和若干 corrupted `.npz`。clean shard 保存 `F_preproj/F_proto/Z_block`、label、weather、scene 和原始 C0 fused logits；corrupted shard保存部署期输入、output/sensor statistics、availability、corruption metadata及离线 target，并以 sample id 与 clean shard关联。global prior 与 topology provenance 在 cache root 单独保存，不在每个 view 重复。

feature 使用 float16、risk/loss 使用 float32、mask 使用 bool、label/index 使用 int16/int32。pre-prototype 统一 pad 到 320 维并保存 `[64,64,320,320]` 的真实宽度，quality adapter 只读取对应有效切片。分片 index 保存 shape/dtype、SHA、sample identity 和 forbidden-field 审计。

### 3. 训练 bank 与评测 bank 使用固定条件清单

train 对每个 sample 由 `hash(seed,sample_id)` 均衡指定一个主要传感器和一个 seen corruption type，并生成 clean/mild/medium/severe 四 view。validation 使用相同固定 bank 且覆盖完整 inner-validation；eval 使用历史 development split而非 outer split，生成 E0 clean、E1 seen 单传感器、E2 mixed、E3 stale、E4 unseen、E5 S0--S5 missing，E6 只按已有 weather metadata 分层。

所有退化由 `global_seed/sample_id/sensor/type/severity/variant` 决定。GPU shard 归属只由 sample identity 的稳定 hash 决定；设置 `CUDA_VISIBLE_DEVICES` 后程序内部始终使用 `cuda:0`。

### 4. 风险 target 与归一化全部离线、train-only

hard CE risk 为 `clamp(CE_corr-CE_clean,0,q_clip)`。topology risk 使用当前已验证 topology soft-label block loss的同一差值；远距离 beam 错误因此比邻近错误承担更大损失。prototype transport 使用现有 debiased topology transport，仅作独立 drift 诊断，不替代 topology risk。

q_clip、median、IQR 和 normalized q99 只从 train cache 的可用、非 missing block拟合，写入 `normalization_stats.json`。validation/eval 和 predicted-risk clip 全部复用该文件；clean feature/logits、label、severity、corruption type、weather 和 target 不进入 quality head forward。

### 5. Q0--Q5 只由固定枚举决定

- Q0：C0 global prior，无训练。
- Q1：output statistics -> topology risk。
- Q2：pre-prototype -> injected severity。
- Q3：pre-prototype -> hard CE risk。
- Q4：pre-prototype -> topology risk。
- Q5：pre-prototype + sensor statistics -> topology risk + within-sample ranking。

Q1--Q5 共享 seed、batch order、AdamW、scheduler、epoch、early stopping、validation-best fused beam loss和一次 train-only loss 量级校准。校准仅用最前 3 个固定 train batch，将 weighted risk/rank loss分别放到 fused beam loss的 20%/10%；不得观察 validation 后重调。

首次运行使用 batch 2048 时，14,400 个有效 train view 每 epoch 仅产生 8 次更新，且五个方向都在 7 epoch 后停止、最佳点均落在 epoch 0；该运行只能视为训练预算诊断，不能据此否证方向。纠正运行固定 batch 256，并在至少完成 10 epoch 后才允许 patience early stopping，使每个可训练方向至少执行约 570 次 optimizer step。旧运行保留在原输出目录，纠正运行写入独立目录并复用相同 cache manifest；两次运行不得混合汇总。

### 6. 有界 prior correction 是唯一融合路径

每模态 `beta_m=beta_max*sigmoid(raw_beta_m)`，初始化约 0.3，默认 `beta_max=2.0`。对 train q99 截断后的风险执行 `prior_logit[m,t]-beta_m*risk[m,t]`，再调用现有 masked block softmax。相同风险只产生公共平移；missing 为零；风险增加时对应权重不增加；risk 非有限立即报错。融合 logits 始终是同一 C0 block logits 的加权和。

### 7. 评测将机制真实性与任务指标分开

D0 使用样本预测，D1 使用 train-fit block mean，D2 使用 train-fit sensor/severity mean且仅标为 oracle-style诊断，D3 令风险为零。报告 Top1、Within-3、MAE 及按 protocol/sensor/severity/weather 分组；质量诊断报告 Pearson、Spearman、per-sample across-block Spearman、MAE、violation、top-risk命中、risk std 和 ranking accuracy。

在 early/middle/late 三个训练位置分别对 predicted risk 求 fused beam loss与quality loss梯度 cosine；不自动添加 PCGrad。最终报告只按预注册五个 gates选择 Q1--Q5 或停止动态融合。

### 8. launcher 只负责独立进程和状态，不管理其他 GPU 进程

预处理 shell 在 GPU0--5 启动六个互斥 shard并等待，然后由 merge 命令执行重复/遗漏、pairing、shape、determinism 和 inference reproduction 检查。训练 shell 先保存 `nvidia-smi`，再将 Q0--Q5 一卡一任务启动，分别保存 PID、日志、resolved config和状态；一个退出码不影响其他进程，脚本不发送 kill 信号。

## Risks / Trade-offs

- [pre-prototype 维度因模态不同] -> 统一 pad 并由 adapter 按保存宽度切片，禁止将 padding 当特征。
- [完整 eval condition bank 预处理耗时] -> 六卡按 sample identity 分片；只运行一次 backbone，Q1--Q5 复用结果；报告实际 wall time和估算加速比。
- [float16 cache 改变边界样本 Top1] -> 在正式训练前对固定 validation 样本比较原始和 cache fused logits/Top1；超出阈值立即停止。
- [sensor statistics 识别 seen corruption 而非任务风险] -> 单独报告 unseen correlation、Q4 对 Q5 与 Dynamic 对 Global Mean，不因 Q5容量更大而默认晋级。
- [已有 PGCD 任务仍占用其他 GPU] -> 只检查并使用用户指定的 GPU0--5，不终止 GPU6/7 或其他用户进程。

## Migration Plan

1. 新增组件、预计算/训练评测程序、launcher和 synthetic tests，不修改 canonical T2 recipe。
2. 运行 100-sample estimate、preflight 和 C0 cache reproduction；失败时停止且保留报告。
3. 在 GPU0--5 完成全量 cache，再并行运行 Q0--Q5 和统一汇总。
4. 保留所有结果为本地 claim-ineligible artifact；代码可通过删除 opt-in分析入口回滚，C0 和默认模型不受影响。

## Open Questions

- 无。历史 development split仅用于本轮 inner/development 评测，明确不视为 outer test 或正式 evidence。
