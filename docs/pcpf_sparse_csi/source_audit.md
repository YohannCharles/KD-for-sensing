# PCPF-T 稀疏 CSI 源码与数据审计

审计日期：2026-07-30。sparse-CSI 正式开发路线现固定绑定 `mmw_trajectory_disjoint_v1`：12 个 train trajectory group、37,510 个 train window；2 个 validation trajectory group、6,365 个 validation window。1 个 test trajectory group、2,985 个 window 继续封存。此前 `mmw_clean_inner_development_v1` 的 3,600/900 审计与 checkpoint 仅保留为迁移背景；outer test、outer evidence 和 future channel 均未读取或构建，本轮全部产物固定 `claim_ineligible=true`。

## 结论与训练门禁

trajectory protocol 与 split audit 门禁通过：train/validation/test 在 trajectory group、sample/target identity、dependency frame、camera/radar/lidar/GPS/channel resource 和完整 CSV row 上均为零交集；manifest fingerprint 为 `d5650621e134c8cbc1bc7ce90684fcb6db41c6c5a5cc9f993b93bfb771b62b90`。正式 PCPF loader 必须直接读取 manifest 中的 train/validation domain CSV，禁止把独立 TSPC record 按 role 二次 join，也禁止构建 test loader。

五模态实现只从当前 trajectory CSV 行的 `csi1..csi5` 历史引用确定性生成稀疏观测。运行时继续逐行校验：所有 channel 文件存在；五个 channel 文件名与 `history_frame_ids_json` 逐项一致；历史 frame 连续递增且最后历史帧严格早于第一个 future frame。训练前必须扫描 37,510/6,365 样本补齐内容寻址 cache，并写出未访问 test 的 cache manifest；任何 identity/hash 不一致仍必须在创建 optimizer 前失败。

旧 clean-inner train 与 trajectory train/validation 分别重叠 2,880/494 条，旧 clean-inner validation 与 trajectory train/validation 分别重叠 716/128 条。因此现有 clean-inner 四模态 Stage 1 checkpoint 已见过 494 条新 validation 样本，不能用于 trajectory 五模态初始化。trajectory sparse-CSI Stage 1 固定 fresh start，不提供四到五模态 checkpoint 迁移。

迁移后的 cache scan 已通过：37,510 个 train window 覆盖 37,662 个唯一历史 channel，6,365 个 validation window 覆盖 6,393 个唯一历史 channel，二者交集为 0；共享内容寻址 cache 从 13,574 增至 44,901 个条目，新增 31,327 个。清单位于本地 ignored 产物 `outputs/pcpf_sparse_csi_router_v1/cache/trajectory_cache_manifest.json`，固定记录 `outer_test_accessed=false`。trajectory 五模态真实单 batch Stage 1/2/3 CUDA smoke 全部 finite，31-subset panel 生效，Stage 1 初始化为空，峰值显存约 1.0 GiB。

## Mother CSI 与时序语义

- TSPC recovery records 中 `candidate_history` 的真实 shape 为 `[N,5,32,16]`，dtype 为 `torch.complex64`；train/validation 分别为 37,510/6,365 个样本，所有实部和虚部均有限。
- `candidate_history[:,i]` 由同一样本第 `i` 个历史 `csi` path 的 `a: complex64 [1,1,16,1,64,L,1]` 和 `tau: float32 [1,1,1,L]` 生成。32 是固定 Tx/Rx probe pattern 数，16 是固定母频率位置数；它不是未来 channel，也不是 beam-power 特征。
- 当前 trajectory PCPF 输入帧为 `t-4..t`，标签 `future_beam_label1` 为 `t+1`。`resolve_input_channel_refs` 要求五个 history frame 连续递增、每个 channel 文件 frame id 与 history id 相同，并要求 `last_input_frame < target_frame`。
- Prepared CSV 含 `future_csi1`，但 `create_samples` 不把该列保留到 dataset row；`MMWDataset(include_channel_history_refs=true)` 只解析 `csi1..csi5`。新 CSI owner 不得接收 `future_csi*`、`future_beam*`、当前/未来 beam power 或历史 beam index。

## 固定 2x2 Pilot 协议

第一轮完全采用 TSPC-V2 的 nested selection：从 32 个 mother pattern 固定取索引 `[0,1]`，从 16 个母频点用 `round(linspace(0,15,2))` 固定取索引 `[0,15]`，五个历史时刻均使用同一选择。对应频率位置为 `[-61440000.0, 61320000.0] Hz`；每帧 4 个复数 RE、每窗口 20 个复数 RE，实际比例为 `4/(32*16)=0.0078125`，即 `0.78125%`。

这里的 pattern index 不是直接天线 index。pattern `0/1` 分别选择 codebook 中同号的 Tx/Rx probe pair；每个 Tx probe 是覆盖 64 个发射阵元的固定权重向量，每个 Rx probe 是覆盖 16 个接收阵元的固定权重向量。因此空间选择记录为 `tx_pattern_indices=[0,1]`、`rx_pattern_indices=[0,1]`、`direct_antenna_indices=null`，并由 codebook logical/file SHA256 固定其完整阵元权重。

选择描述的 canonical JSON 为：

```json
{"frequency_indices":[0,15],"history_indices":[0,1,2,3,4],"mother_shape":[5,32,16],"pattern_indices":[0,1],"schema_version":1,"selection_rule":"tspc_v2_nested_prefix_v1"}
```

其 SHA256 为 `87bad2292ba3d22cac413e71d9303f2dd229ed64fe39eeb4df6272f42e6bca28`。固定 probe codebook 的逻辑 hash 为 `eec8709fbabff5bf2530a7b3789d3d23ad59165c1eead8ebad44af995ec763fa`，文件 SHA256 为 `efa88aa18483664b53fe9506f15aa6c9bbf276e2042fb4664ee56d04e806779e`。pattern、频率和 codebook 均在训练前固定，不读取 label 或 validation 指标。

本轮禁止调用 TSPC 的 `_noisy_observations`/`_noisy_pilots`：不注入 AWGN、pilot dropout 或其他人工 corruption。输入是上述 path-domain simulator 的确定性历史观测。

## SparsePilotEncoder 与 SNR

`SparsePilotEncoder` 接收 complex `[B,M,K]`、`pattern_ids [B,M]`、frequency position、bool pilot mask 和可选 `snr_db`，从 real、imag、log magnitude、validity 与 frequency 生成 `csi_feature`，同时输出 valid ratio、log RMS、quality confidence 和 `snr_available`。它没有丢弃虚部。

Recovery record、prepared CSV 和 channel NPZ 中均没有真实 SNR 字段。旧 TSPC 配置中的 `train_snr_db`/`validation_snr_db` 是人工噪声实验控制，不是真实观测。本轮必须让 SNR 显式可选；缺失时输出 `snr_available=false`，不得随机生成或把固定常数描述为实测 SNR。第一轮 quality confidence 只记录诊断，不进入四项 risk 或解析权重。

## PCPF 本地五模态边界

五模态支持不修改全局 `MODALITY_ORDER`。PCPF 内局部定义 sensing 顺序 `image,radar,gps,lidar` 和 opt-in 顺序 `image,radar,gps,lidar,csi`；canonical dataset 仍加载四个 sensing modality，CSI 作为只读历史 sidecar；mask/evaluator 仅在 `use_sparse_csi=true` 时扩展为五模态。`use_sparse_csi=false` 不实例化 CSI module。

## Stage 预算与 checkpoint

以下三个现有 seed1 validation-best checkpoint 均使用 clean-inner、batch 32、Adam、weight decay `1e-4`、gradient clip 5、无 scheduler，并按 validation loss 最小选择；它们只用于历史背景，不得加载到 trajectory 路线：

| Stage | Epoch | LR | Best epoch / validation loss | Checkpoint SHA256 |
|---|---:|---:|---:|---|
| Stage1 expert | 40 | `5e-4` | 40 / 6.24175835 | `fb0f791d35c09b836428b81998ba0096817d2fc4c836af0c773bee4f2d5a8355` |
| Stage2 risk | 20 | `5e-4` | 20 / 0.72446114 | `65fbd39a5df4e5537169e29dd88e6ce73d9e9cd9770d1ca04af2e3acb7bb35fd` |
| Stage3 fusion | 10 | `1e-4` | 10 / 1.88972914 | `61c3039f3a659e69da2533cfd6ba32ad0a7bf7f3430f51419e73ebc7cbb86db5` |

trajectory 五模态 Stage1 采用相同 40 epoch seed1 budget，但全部模型参数 fresh start，并使用 trajectory train-only GPS scaler（effective sample count 37,510，sample identity hash 与 split audit 一致）。Stage2/3 只接受新五模态前一阶段、同一 trajectory protocol identity 的 validation-best checkpoint。

## 数据流与冻结预期

历史 channel refs 经固定 codebook/2x2 selection 得到 `complex [B,5,2,2]`，每帧进入同一 `SparsePilotEncoder`，再经 Linear/LayerNorm 投影为 `[B,5,64]`。它与四 sensing sequence stack 为 `[B,5,5,64]`，逐模态进入唯一 `SharedTemporalTransformer`，并通过同一 `ProbabilityEmbeddingHead`、唯一 `BeamPrototypeBank`、四项 risk 和解析融合。CSI 全缺失时 frame、CLS、probability、risk 和 weight 必须显式为零。

Stage1 训练五个 encoder/input path、共享 Temporal Transformer 和 prototype bank；Stage2 只训练共享 probability head 与 risk coefficient/bias；Stage3 默认只训练五个 temperature 与全局 tau（以及配置明确允许的现有低维参数）。旧 Direct Router 仅为 control，CSI 不建立独立最终 classifier，也不直接叠加到 fused logits。
