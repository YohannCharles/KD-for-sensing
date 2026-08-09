# PCPF-T 稀疏 CSI 源码与数据审计

审计更新日期：2026-08-02。sparse-CSI 开发路线现固定绑定 `mmw_id_stratified_block_v1` seed 0 manifest v2：train/validation/test 为 350/75/75 blocks、27,666/5,931/6,003 windows，三个 split 均覆盖全部 5 个场景和 16 条 trajectory。开发 loader 与 sparse-CSI bundle 只包含 train/validation，test 保持封存；全部开发产物继续固定 `claim_ineligible=true`。

## 结论与训练门禁

ID-block protocol 与 split audit 门禁通过：trajectory overlap 是设计目标；block、base frame、天气副本、窗口引用原始 frame 的跨 split 重叠均为 0，窗口不跨 block。manifest fingerprint 为 `e630414f70d6260f14b5b8bd5b4f586eceb3fa627c37d7ed49c09967df3e30de`，manifest SHA256 为 `856b9ca4627e71e0f311eb5908eb4ea62cf1f91b5ef8b772bd9e863c3caeb833`。正式 PCPF loader 默认只能读取 manifest 中的 train/validation domain CSV。

五模态实现只从当前 block CSV 行的 `csi1..csi5` 历史引用确定性生成稀疏观测。运行时继续逐行校验：所有 channel 文件存在；五个 channel 文件名与 `history_frame_ids_json` 逐项一致；历史 frame 连续递增且最后历史帧严格早于第一个 future frame。训练前必须扫描 27,666/5,931 个 train/validation 样本并写出完整八项 split identity 与 `test_evaluated=false` 的 cache manifest；任何 identity/hash、coverage 或 packed-cache SHA256 不一致都必须在创建 optimizer 前失败。

所有旧 clean-inner、资源连通 80/10/10 与 `mmw_trajectory_disjoint_v1` cache/checkpoint 都属于已退休历史输入，不能用于当前 trajectory 初始化或 paired comparison。trajectory sparse-CSI Stage 1 固定 fresh start，不提供四到五模态 checkpoint 迁移。

当前 cache scan 已通过：27,666 个 train window 覆盖 31,866 个唯一历史 channel，5,931 个 validation window 覆盖 6,831 个唯一历史 channel，二者交集为 0。原始内容寻址 cache 保留 47,052 项且无需新增；新 split-specific packed bundle 含 38,697 项，SHA256 为 `12a0650a8b55f59c9fe20c8fa1f3e8df83b0f107571e13addd58fb2508a85a09`。本地 ignored 清单 `outputs/pcpf_sparse_csi_router_v1/cache/trajectory_cache_manifest.json` 记录 schema 4、完整 block identity 和 `test_evaluated=false`。这些缓存完整性事实不构成模型效果 claim。

## Mother CSI 与时序语义

- 当前 seed 0 train/validation cache scan 分别覆盖 27,666/5,931 个样本；默认 selection 从五帧历史 channel 生成 `[5,2,2]` complex sparse observation，C2 独立 selection 生成 `[5,4,2]`。
- `candidate_history[:,i]` 由同一样本第 `i` 个历史 `csi` path 的 `a: complex64 [1,1,16,1,64,L,1]` 和 `tau: float32 [1,1,1,L]` 生成。32 是固定 Tx/Rx probe pattern 数，16 是固定母频率位置数；它不是未来 channel，也不是 beam-power 特征。
- 当前 trajectory PCPF 输入帧为 `t-4..t`，标签 `future_beam_label1` 为 `t+1`。`resolve_input_channel_refs` 要求五个 history frame 连续递增、每个 channel 文件 frame id 与 history id 相同，并要求 `last_input_frame < target_frame`。
- Prepared CSV 含 `future_csi1`，但 `create_samples` 不把该列保留到 dataset row；`MMWDataset(include_channel_history_refs=true)` 只解析 `csi1..csi5`。新 CSI owner 不得接收 `future_csi*`、`future_beam*`、当前/未来 beam power 或历史 beam index。

## 固定 2x2 Pilot 基线与 C2 4x2 筛选

第一轮完全采用 TSPC-V2 的 nested selection：从 32 个 mother pattern 固定取索引 `[0,1]`，从 16 个母频点用 `round(linspace(0,15,2))` 固定取索引 `[0,15]`，五个历史时刻均使用同一选择。对应频率位置为 `[-61440000.0, 61320000.0] Hz`；每帧 4 个复数 RE、每窗口 20 个复数 RE，实际比例为 `4/(32*16)=0.0078125`，即 `0.78125%`。

这里的 pattern index 不是直接天线 index。pattern `0/1` 分别选择 codebook 中同号的 Tx/Rx probe pair；每个 Tx probe 是覆盖 64 个发射阵元的固定权重向量，每个 Rx probe 是覆盖 16 个接收阵元的固定权重向量。因此空间选择记录为 `tx_pattern_indices=[0,1]`、`rx_pattern_indices=[0,1]`、`direct_antenna_indices=null`，并由 codebook logical/file SHA256 固定其完整阵元权重。

选择描述的 canonical JSON 为：

```json
{"frequency_indices":[0,15],"history_indices":[0,1,2,3,4],"mother_shape":[5,32,16],"pattern_indices":[0,1],"schema_version":1,"selection_rule":"tspc_v2_nested_prefix_v1"}
```

其 SHA256 为 `87bad2292ba3d22cac413e71d9303f2dd229ed64fe39eeb4df6272f42e6bca28`。固定 probe codebook 的逻辑 hash 为 `eec8709fbabff5bf2530a7b3789d3d23ad59165c1eead8ebad44af995ec763fa`，文件 SHA256 为 `efa88aa18483664b53fe9506f15aa6c9bbf276e2042fb4664ee56d04e806779e`。pattern、频率和 codebook 均在训练前固定，不读取 label 或 validation 指标。

本轮禁止调用 TSPC 的 `_noisy_observations`/`_noisy_pilots`：不注入 AWGN、pilot dropout 或其他人工 corruption。输入是上述 path-domain simulator 的确定性历史观测。

C1 validation-best 相对同协议零层 C0 已提高 Top-1/Top-3/Top-5/ADBA，但逐 epoch 曲线同时显示明显过拟合。后续 C2 因此只增加一个预注册开销档：保持一层 token Transformer、CSI-only mask、loss、split seed 0、train seed 1、40 epoch、batch 64、8 workers和 validation-loss checkpoint selection 不变，把 pattern 固定扩为 `[0,1,2,3]`，频率仍为 `[0,15]`。C2 输入为 `[5,4,2]` complex，每帧 8 RE、每窗口 40 RE，占 mother grid 的 `8/(32*16)=1.5625%`；selection SHA256 为 `2d035d64f6b9ac408532040b3ff09151a8831361d81c83b1b77e218e4344a4f4`。

2x2 与 4x2 是 sidecar 唯一允许的 selection。二者复用同一 codebook、两个物理频率与原内容寻址 `[32,2]` mother cache，但 4x2 必须生成独立的 `[N,4,2]` packed bundle/cache manifest；selection hash 或 shape 不匹配时失败，不能把已有 2x2 bundle 重标。C2 仍只读取 train/validation 历史 channel，test 保持封存，结果保持 `claim_ineligible=true`。

## SparsePilotEncoder 与 SNR

`SparsePilotEncoder` 接收 complex `[B,M,K]`、`pattern_ids [B,M]`、frequency position、bool pilot mask 和可选 `snr_db`，从 real、imag、log magnitude、validity 与 frequency 生成 `csi_feature`，同时输出 valid ratio、log RMS、quality confidence 和 `snr_available`。它没有丢弃虚部。

Recovery record、prepared CSV 和 channel NPZ 中均没有真实 SNR 字段。旧 TSPC 配置中的 `train_snr_db`/`validation_snr_db` 是人工噪声实验控制，不是真实观测。本轮必须让 SNR 显式可选；缺失时输出 `snr_available=false`，不得随机生成或把固定常数描述为实测 SNR。第一轮 quality confidence 只记录诊断，不进入四项 risk 或解析权重。

## PCPF 本地五模态边界

五模态支持不修改全局 `MODALITY_ORDER`。PCPF 内局部定义 sensing 顺序 `image,radar,gps,lidar` 和 opt-in 顺序 `image,radar,gps,lidar,csi`；canonical dataset 仍加载四个 sensing modality，CSI 作为只读历史 sidecar；mask/evaluator 仅在 `use_sparse_csi=true` 时扩展为五模态。`use_sparse_csi=false` 不实例化 CSI module。

## Stage 预算与 checkpoint

以下三个已退休 seed1 validation-best checkpoint 来自旧 clean-inner、batch 32、Adam、weight decay `1e-4`、gradient clip 5、无 scheduler，并按 validation loss 最小选择；它们只用于历史背景，不得加载到当前 trajectory 路线：

| Stage | Epoch | LR | Best epoch / validation loss | Checkpoint SHA256 |
|---|---:|---:|---:|---|
| Stage1 expert | 40 | `5e-4` | 40 / 6.24175835 | `fb0f791d35c09b836428b81998ba0096817d2fc4c836af0c773bee4f2d5a8355` |
| Stage2 risk | 20 | `5e-4` | 20 / 0.72446114 | `65fbd39a5df4e5537169e29dd88e6ce73d9e9cd9770d1ca04af2e3acb7bb35fd` |
| Stage3 fusion | 10 | `1e-4` | 10 / 1.88972914 | `61c3039f3a659e69da2533cfd6ba32ad0a7bf7f3430f51419e73ebc7cbb86db5` |

当前 ID-block 五模态 Stage1 全部模型参数必须 fresh start，并使用 seed 0 train-only GPS scaler（effective sample count 27,666，sample identity hash 与 split audit 一致）。Stage2/3 只接受新五模态前一阶段、同一 block protocol identity、split seed 和 train seed 的 validation-best checkpoint。

## 数据流与冻结预期

历史 channel refs 经固定 codebook 与预注册 selection 得到 `complex [B,5,M,2]`（默认 `M=2`，仅 C2 为 `M=4`），每帧进入同一 `SparsePilotEncoder`，再经 Linear/LayerNorm 投影为 `[B,5,64]`。它与四 sensing sequence stack 为 `[B,5,5,64]`，逐模态进入唯一 `SharedTemporalTransformer`，并通过同一 `ProbabilityEmbeddingHead`、唯一 `BeamPrototypeBank`、四项 risk 和解析融合。CSI 全缺失时 frame、CLS、probability、risk 和 weight 必须显式为零。

Stage1 训练五个 encoder/input path、共享 Temporal Transformer 和 prototype bank；Stage2 只训练共享 probability head 与 risk coefficient/bias；Stage3 默认只训练五个 temperature 与全局 tau。Direct Router/CUAF 已退出 active surface；CSI 不建立独立最终 classifier，也不直接叠加到 fused logits。
