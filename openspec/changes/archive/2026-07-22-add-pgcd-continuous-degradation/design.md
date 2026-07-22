## Context

MMW current loader 固定使用 `image/radar/gps/lidar`，历史窗口 `T=5`，经 encoder 后形成 `[B,5,4,64]` latent。PCER active change 已复用同一 `BeamPrototypeBank` 形成 `[B,20,64]` block evidence，并已有 block availability、masked softmax、validation-best checkpoint 和 fixed-mask 评测基础；PGCD 应在该 surface 上增加连续退化与质量监督，不复制 encoder、prototype bank 或训练循环。

实际输入是 ImageNet 归一化 RGB `[B,5,3,224,224]`、RA/DA radar maps（模型前拼成 `[B,5,2,128,64]`）、train-fit 标准化的 BS-centric `relative_polar` GPS `[B,5,3]`、点云在线栅格化得到的 LiDAR BEV `[B,5,3,224,224]`。prepared CSV 历史上含 channel/path/power 路径列，但 dataset current sample 只加载四传感器文件和最优 beam index；PGCD 必须关闭 power utility target，并 fail closed 拒绝信道 tensor 或信道配置。

本轮仅为 seed1 inner/development 快筛。C0-C7 共享数据划分、初始化算法、训练预算、corruption 随机流、optimizer、scheduler、BPA/topology 与 validation-best 规则；测试和自然天气结果不得反向调参，也不得升级为正式 claim。

## Goals / Non-Goals

**Goals:**

- 提供 L0-L4、按 sample identity 可复现的四传感器在线连续退化，并让 missing 成为同一谱的 availability endpoint。
- 从 clean/corrupted 共享 prototype evidence 构造 topology drift、task degradation 与 combined target，监督只读取最优 beam index 和声明的 64-beam topology。
- 用轻量 block quality estimator 在 learned block prior 上调制动态 reliability，并通过 D0-D3 替换验证真实样本级价值。
- 完成 C0-C7 配置、preflight、seen/unseen/weather/missing 评测与诊断输出；任一失败必须显式记录。

**Non-Goals:**

- 不读取 channel、CSI、ray-tracing path、beam-power/channel-gain vector，不以其构造输入、target、loss 或指标。
- 不合成 rain、fog 或 snow，不改变原始文件、MMW split、canonical T2 默认行为或 public CLI。
- 不新增 EMA/外部 teacher、MoE、第三方依赖、multi-seed、outer test 或下一轮实验。
- 不使用 weather、corruption type、severity 或 clean evidence 作为推理期 quality estimator 输入。

## Decisions

### 1. 在线退化集中在独立 batch 变换器

新增 `SensorDegradationGenerator`，直接处理 collated 四传感器张量并输出新 batch、`[B,T,M]` availability 和仅供 loss/审计使用的 metadata。种子由 `global_seed/sample_id/sensor/corruption/severity/variant` 做 SHA256 派生，训练 epoch/step 只参与训练采样；eval 对相同 identity 完全可复现。

Image 在归一化空间支持 Gaussian blur 和随机 1-3 矩形遮挡，并通过 ImageNet 反归一化实现 exposure/noise。LiDAR 当前为 BEV，因此 point dropout 以共享空间 cell mask 同时清零三通道，coordinate jitter 以 ROI 分辨率换算空间位移。Radar 当前为 RA/DA maps，因此 detection dropout 以共享 bin mask 同时作用两张图，coordinate jitter/false clutter 只在非 padding 合法范围内生成。GPS 在 train-fit scaler 反变换后的 `relative_polar` 物理表示中转为相对 XY，施加时间相关 drift/jump/noise 后再转回并标准化。

one-step stale 只令 `t>0` 使用 `t-1`；`t=0` 保持原值。若提供 source frame ids，随机参数按真实 source frame 分组后广播；当前 MMW 审计若无窗口内复制帧，则记录该分支未激活。

### 2. PGCD 复用 PCER block surface

在 `UMaskBeamJEPA` 增加 opt-in `pgcd` 配置与一个共享 `PrototypeGuidedDegradationRouter`。默认未声明时不实例化参数，canonical T2 forward/state dict 不变。启用时模型复用 `[B,T,M,D]` latent 和同一 prototype bank，质量估计器只消费 corrupted block feature、prototype probability 的 confidence/entropy/margin、modality/time embedding 和 availability。

Router 保留 trainable `prior_logits[N]`；C0 仅使用 masked prior。C1-C7 输出非负 `predicted_degradation`，令 `reliability=exp(-softplus(predicted_degradation))`，并用 `softplus(raw_beta)` 调制 `prior + beta*log(reliability+eps)`。missing reliability/weight 严格为零，可用权重和严格为一。

### 3. clean teacher 在 extension 内按批次生成

训练 extension 先复制未退化 batch、建立全可用 clean mask，并以共享模型 `eval()+no_grad()` 取得 clean block evidence；随后在线退化原 batch，主 forward 只处理 corrupted view并保留梯度。clean evidence 一律 detach，不建立额外模型或 checkpoint teacher。C0 仍生成 corruption，但跳过 clean teacher 以节省计算；C1 可只保留 severity target。

### 4. 同时保留 raw transport 与零自漂移 target

按 active topology positions 构造归一化 `D_beam[64,64]`，不加载 beam pattern、power 或 channel 文件。附件给定的 `E(p_clean,p_corr)=einsum(...)` 作为 `raw_topology_transport` 记录。由于该量对非 one-hot 的相同分布通常不为零，回归 target 使用去偏 energy form：

```text
drift = E(clean,corr) - 0.5 E(clean,clean) - 0.5 E(corr,corr)
```

再截断到非负；这保留 topology 距离并满足 clean-vs-self 为零。task degradation 使用每个 block 的 topology soft-label CE 差，保留 raw 值并对 target 截断。归一化只使用当前训练 batch 的可用 block；validation/test 不更新统计。missing target 设为 1，但不参与连续回归。

### 5. 单一模式枚举表达 C0-C7

`loss.u_mask_beam_jepa.pgcd.variant` 只允许 `c0` 到 `c7`，由固定映射决定 target 与 loss：C0 prior-only；C1 severity regression；C2 无显式质量监督；C3 topology regression；C4 topology regression+ranking；C5 task regression；C6 combined regression+ranking；C7 再加 degradation-aware consistency。这样避免任意布尔组合产生不可比较方法。

ranking 在同一 batch/sample 的 clean-corrupt block pair及不同 severity block pair上，只有 detached target 差超过阈值才激活。consistency 直接使用 topology drift，并乘 `exp(-target)`，不会强制 severe 与 clean 完全一致。质量和 beam loss 对 predicted degradation 的梯度 cosine 在可求导时记录；同时记录 quality/router/backbone gradient norm。

### 6. launcher 只生成受控本地运行面

tracked helper 从 `mmw_twc_outer_v1` manifest 读取 frozen inner train/validation 和 historical development test，生成八份 ignored resolved config、manifest、PID/status 和 `scripts/run_pgcd_quick_search_gpu4_5.sh`。配置固定 20 epochs、batch 32、bf16、best validation loss 和相同 effective batch。按用户运行时指示，GPU4 串行执行 C0/C2/C4/C6，GPU5 串行执行 C1/C3/C5/C7，两卡并行且每卡同一时刻只有一个任务。`--prepare` 与 `--preflight` 不启动训练；只有显式 `--launch` 才启动任务，单任务失败不终止队列后续任务。

固定评估复用 validation-best checkpoint 记录且通过 train-fit provenance 校验的 GPS scaler，避免每个 C0-C7 worker 重复扫描训练集。评估仅将 test batch 提高到 256 以利用空闲显存，并以 inference mode 执行 D1-D3 cached reroute，避免纯评估路径建立反向图；D1/D2 train-fit replacement statistics 仍使用训练 batch 32 和固定 8 batches，因此不改变评估协议口径。

评测按 sample identity 生成 E0-E5，D1 global mean 只由 train subset 汇总，D2 sensor+severity mean 只作 oracle-style 诊断，D3 reliability=1。天气默认只分组；只有 condition 之外的 town/scenario/trajectory/frame/label 全部严格一致才允许 paired analysis。评估调度可复用已完成且 condition inventory 完整的实验结果，并将剩余实验分配到用户明确释放的 GPU0-5；不得重复已完成实验或改变 checkpoint/condition identity。

## Risks / Trade-offs

- [双 view 增加显存和计算] -> clean 分支 `no_grad` 且不保留 activation；若仍超限，八个任务统一减小 batch 并增加相同 accumulation。
- [BEV/map dropout 不是原始 detection dropout] -> 在实现记录中明确 representation-adapted 语义，不伪称原始点级操作；参数由实际输入范围与 ROI 分辨率确定。
- [同 batch 归一化噪声较大] -> 快筛优先满足无验证泄漏；方向通过后才考虑冻结 train-only running statistics。
- [自然天气样本看似同轨迹但非严格对应] -> 不构造伪 clean-weather teacher，缺少全键匹配时只输出分组统计。
- [八任务仍不超过 global mean] -> 如实判定动态路线失败，不追加 Router 容量或选择性汇报。

## Migration Plan

1. 保持 canonical T2 未声明 `pgcd`，先完成 config fail-closed、degradation 与数学单测。
2. 接入 opt-in Router 和 training extension，完成 synthetic forward/backward、双 view 和泄漏测试。
3. 生成 C0-C7 resolved configs，运行 1-2 batch preflight 与相关项目验证。
4. 显式启动 seed1 八任务并在完成后评测、汇总；不修改 canonical recipe 或正式 claim。
5. 任一阶段可移除 opt-in PGCD config/run，默认 T2 行为与历史 checkpoint 不受影响。
