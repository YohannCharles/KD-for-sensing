# TSPC-V2 源码审计

审计日期：2026-07-28。范围只包括 `mmw_trajectory_disjoint_v1` 的 development train/validation；没有构建 outer-test loader、读取 outer-test record 或生成 outer-test pilot。V2 必须保持现有 TSPC P0--P5/L0--L5 和已发布结果不变。

| 模块 | 当前文件 | 类/函数 | 输入输出 Shape | 当前作用 | V2 复用或修改 |
|---|---|---|---|---|---|
| Canonical model factory | `src/kd_sensing/engine/optim.py`, `src/kd_sensing/registries.py` | `build_model`、`MODELS` | canonical YAML -> registered model | 服务保留的 public U0/baseline workflow | V2 是 active OpenSpec 下的 local research workflow，由 `tools/run_tspc_v2.py` 直接构造新 model，不注册 public/canonical recipe，避免扩张现有 CLI surface |
| M4 四模态编码与融合 | `src/kd_sensing/baselines/mmw_trajectory.py` | `TrajectoryBaselineModel.encode/forward_tokens` | 四个 encoder 各 `[B,5,64]`；stack `[B,5,4,64]`；`fused_features [B,64]`、`logits [B,64]` | 冻结 M4 使用 time/modality embedding、availability 置零和 `1280->1024->512->64` MLP | 复用 encoder token cache、M4 Full probability、真实 bank；V2 主感知路径不再把 1280 展平 |
| M4 原始 fusion 实现 | `src/kd_sensing/baselines/full_pool_candidate12.py` | `Candidate12Model` | token `[B,5,4,64]`、fusion input 1280 | 提供 M4 encoder、positional embedding和冻结 head 的来源 | 仅作为 A0/Full teacher 参照；不得在 V2 主路径中复用 Flatten MLP |
| M4 checkpoint 加载 | `tools/run_tspc_final_ablations.py` | `_load_m4` | checkpoint -> frozen `TrajectoryBaselineModel` | 校验 ABTC method、protocol fingerprint、bank `[64,64]` | V2 runner 沿用同一 checkpoint identity 并冻结全部 M4 参数 |
| M4 feature cache | `tools/run_csi_anchored_completion.py` | `_extract_feature_role/validate_cache_record` | `token_sequence [N,5,4,64]`、`p_full [N,64]` | 从真实 M4 encoder 提取、验证稳定 ID 和无 future channel | V2 只读复用；cache 与 recovery record 的 sample ID 必须逐元素一致 |
| Beam Prototype Bank | `src/kd_sensing/losses/beam_prototype_alignment.py` | `BeamPrototypeBank` | feature `[B,64]` -> evidence `[B,64]`；prototypes `[64,64]` | L2 normalize 后 cosine / `temperature=0.1` | 主 A2 与 residual query 复用同一冻结 bank；A3/A4 为显式控制，A1 无 prototype |
| 已验证 Prototype Loss | `src/kd_sensing/losses/beam_prototype_alignment.py` | `prototype_alignment_loss` | `fused_features [B,64]`、labels -> scalar | circular Gaussian topology alignment | V2 sensing branch 通过可配置权重复用，不新建不兼容标签目标 |
| CSI mother cache 和采样 | `tools/run_tspc_final_ablations.py` | `select_candidate_history/_noisy_observations` | mother `[N,5,32,16] complex` -> selected `[B,T,M,K] complex` | 嵌套选择、历史 AWGN/dropout，验证 noise seed 固定 | V2 C2 选择 `2x2 x 5`，4 RE/frame、20 RE/window；C1 只保留最后帧 `5x4` 并置前四帧无效 |
| Probe codebook/simulator | `src/kd_sensing/channel/probe_codebook.py`, `src/kd_sensing/channel/sparse_pilot_simulator.py` | `generate_probe_codebook`、simulator | QPSK Tx/Rx probe；complex scalar RE | 产生和审计 32x16 mother observation | 不重拟合 codebook，不读取完整 CSI 或 future channel |
| Sparse CSI frame encoder | `src/kd_sensing/models/sparse_pilot_encoder.py` | `SparsePilotEncoder` | complex `[B,M,K]` -> `csi_feature [B,128]`；bool `[B,M,K]` | real/imag/log-mag/valid/frequency token feature、pattern embedding、masked pooling | V2 复用并启用可选 frequency/time/validity index embedding；invalid token 保持 key mask |
| CSI temporal encoder | `src/kd_sensing/models/temporal_radio_encoders.py` | `TemporalRadioEncoder` | `[B,5,128] -> [B,128]` | T0--T5 的公平时序控制，LSTM 为最佳已测实现 | V2 新 encoder 复用同样 2-layer LSTM 超参数，但保留全部 `[B,5,128]` 输出给 residual |
| 旧 CSI evidence/fusion | `src/kd_sensing/models/tspc_ablation_heads.py` | `SparseRadioAblationModel/fuse_expert_probabilities` | radio `[B,128]` -> evidence `[B,64]`；fixed fusion | P0--P5/L0--L5 及 Full/CSI-off fallback | B0 保持该不变基线；V2 主路径不得调用其 64 类 CSI head 或 0.5 fusion |
| 现有 CSI completion | `src/kd_sensing/models/csi_anchored_completion.py` | `SparsePilotRadioEncoder/CSIAnchoredPrototypeCompletion` | radio `[B,128]`、slot completion | 旧 B4--B8D 路线通过重建 M4 slot 再走 M4 fusion | 不复用其 slot completion，以免把 V2 混成“重建模态”；只参考 cache/Full 分流策略 |
| 缺失 mask | `src/kd_sensing/baselines/mmw_trajectory.py`, `tools/run_mmw_trajectory_baselines.py`, `src/kd_sensing/data/temporal_missing.py` | `availability_balanced_assignment/ALL_PATTERNS` | physical `[B,4]`，可扩为 temporal `[B,5,4]` | 定义 14 个非 Full mask 与训练 mask schedule | V2 固定 modality 顺序 `[image,lidar,radar,gps]`，支持 14 mask 并保留所有槽位 |
| 训练/评估/checkpoint | `tools/run_tspc_final_ablations.py` | `preflight/train/_evaluate_outputs/_save_checkpoint` | records、checkpoint、metrics | hash/protocol fail-closed、train-only 更新、validation read-only | V2 采用独立 local runner 和 output root；checkpoint 记录 git/seed/split/M4 hash/noise seed |
| 三 seed 与报告 | `tools/run_tspc_final_ablations.py` | `select_candidates/summarize` | per-seed JSON -> CSV/report | 现有 P0/P1/T3 的公平结论 | V2 提供独立 smoke/ablation/three-seed/report 命令；不自动启动长期训练 |

## 已确认的数据契约

- `records/train.pt` 与 `records/validation.pt` 含 `candidate_history [N,5,32,16] complex`、`labels_future [N]`、`future_beam_power [N,64]`、各旧 mask 的 `z_* [N,64]` 与 `p0_* [N,64]`。`p0_*` 是 probability，不能当作 logits/evidence 直接相加。
- M4 feature cache 含 `token_sequence [N,5,4,64]` 与 `p_full [N,64]`，不含 channel/path tensor；它是 V2 分层感知输入和 Full hard bypass 的唯一合法冻结感知来源。
- `candidate_history` 是历史 mother observation；V2 选择的 `2x2` 为每帧 4 complex RE，五帧共 20 RE。它不先扫描 full CSI 再压缩。
- 当前 `SparsePilotEncoder` 从 complex tensor 读取 real/imag；V2 保持该表示，所有 pilot/availability mask 使用 `torch.bool`。

## 关键边界与兼容策略

- M4 `forward_tokens` 的 Full output 是 logits；feature cache 的 `p_full` 是 `softmax(logits)`。V2 Full 分支直接复制 `p_full`，以保证 probability max difference 和 argmax mismatch 均为零，不把 `log(p_full)` 伪称为原始 M4 logits。
- Cache 不含原始 Full logits，因此 residual-regression 配置使用逐样本居中的 `log(p_full)` 作为 canonical evidence，并同时居中 sensing/residual；该表示只保留可辨识的类间 evidence 差，必须在 metadata 中记录，不能声称恢复了原始 M4 logits。
- 现有 records 不含每时刻 modality token，因此 V2 依赖已有、hash-validated completion feature cache；若 cache 缺失或身份/hash 不一致，V2 preflight fail closed，而不是从 validation 或 outer test 重建。
- 共享真实 bank 在已完成的 P0/P1 消融中不是精度最优配置；V2 只可称其为语义/参数约束，必须报告 independent 与 random-frozen 控制。
- `outer_test_enabled` 必须为 false，V2 runner 只接受 `train`/`validation` role，并拒绝任意包含 `future_channel`/`future_csi` 的 record key。
- Stage C 为 sensing 与 CSI/compensator 分组优化；主配置将 sensing learning rate 固定为其他 V2 路径的 `0.1`，M4 encoder 与共享 bank 始终冻结。
