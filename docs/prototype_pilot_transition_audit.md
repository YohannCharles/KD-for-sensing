# Prototype-Conditioned Sparse Pilot Transition 审计

## 审计范围与结论

审计日期为 2026-07-27。范围包括 current MMW U0 训练/评估路径、prepared sequence contract、本地 15 个 Town03 domain、47,100 个 path-level channel NPZ、现有 prototype/BPA 与 beam topology 实现，以及生成 prepared beam label 的历史源码。

结论：稀疏 pilot 可以从现有 sequence row 无歧义绑定到最后输入帧，但 current dataset 尚未导出 channel reference。新增代码必须只导出不透明路径引用，不得把 `a`、`tau` 或完整 CSI 放进 batch/model。数据文件没有 `Nc`、子载波间隔或 frequency-index metadata；现有 label 生成也不使用频率轴，因此 `1024/120 kHz` 只能作为显式实验配置，`frequency_index_mode=auto` 在无外部证据时必须失败。

## 当前运行路径

- 训练入口：`kd_sensing.cli.train:main`，由 `Trainer`、`BatchStepRunner` 和 training extension 组织 forward/loss/optimizer。
- 配置：`configs/mmw/_base.yaml` 提供 MMW 数据、时序与 15-mask 评估默认值，`configs/mmw/u0.yaml` 提供 current U0 模型和 BPA/superset loss。MMW 实际运行还必须由 launcher 注入 audited clean/full-pool protocol。
- Dataset：`MMWDataset` 调用 `create_samples` 读取 prepared CSV；PyTorch 默认 mapping collate 负责 batch 聚合，没有项目自定义 `collate_fn`。
- Forward：`prepare_fusion_inputs` 生成四模态 tensor 和 temporal availability；`UMaskBeamJEPA.forward` 编码、masked-mean、supervised routing，并返回 logits、fused feature、modality feature、mask 和只读 prototype state。
- Evaluator：`kd_sensing.cli.evaluate:main` 使用 engine evaluator/evaluation pass；15 个非空 modality mask 由现有 missing-pattern contract 枚举。

## 四模态与时间协议

Prepared frame manifest 以 `(condition, town, sensor_scenario, agent, frame_id)` 对齐 camera、LiDAR、GPS、RSU radar、channel path 和 beam-power artifact。`build_sequence_rows` 只在连续 frame segment 内构造窗口，并在同一 row 保存：

- 历史感知：`camera1..T`、`lidar1..T`、`gps1..T`、派生 radar/BS-GPS；
- 历史 channel：`csi1..T`；
- 未来标签：`future_beam_label1..H` 和 `future_beam1..H`；
- 未来 channel：`future_csi1..H`，仅供标签/通信增益诊断，禁止作为模型输入；
- 身份：`history_frame_ids_json`、`future_frame_ids_json`、sample/target/trajectory ids。

Canonical U0 使用 `history_window=5`、`prediction_window=1`。因此输入为 `t-4..t`，`target_beam[:,0]` 是 `t+1` 的 beam label，不是当前时刻标签。Sparse-pilot 默认协议固定为：

```text
pilot_time_mode = last_input
pilot frame      = t = csi5
target frame     = t+1 = future_csi1/future_beam_label1
required order   = pilot_frame <= last_input_frame < target_frame
```

当前 `MMWDataset` 没有保留 `csi*` row columns，也没有读取 channel 文件，故 current 四模态 baseline 不存在 channel 泄漏。新增 `channel_ref` 时的主要风险是误取 `future_csi1`；实现必须在读取 NPZ 前按 frame id 和列族 fail closed。

## Prototype bank 与 assignment

- `BeamPrototypeBank` 数量为 64，与 beam class 一一对应；不是一个 beam 多 prototype。
- prototype 为可学习的 `[64,d_model]` 参数，logit 是归一化 feature 与归一化 prototype 的 cosine similarity 除以 temperature。
- `describe()` 输出 softmax assignment、最近 prototype id、距离、margin、entropy 和 restoration residual；soft assignment 是无标签的，hard id 为最近 prototype，即 cosine 最大项。
- U0 主输出是各可用模态 prototype logits 经 supervised router 加权后的分布；同时返回 fused feature 的只读 prototype state。稀疏 pilot selector 应使用无标签 hard prototype id，不得使用 target beam。

## Beam topology 与损失

- `make_soft_beam_labels` 和 BPA 复用 `beam_topology_positions`。支持 `linear_index_v1`、`cyclic_index_v1`、`permuted_index_v1` 和 `ula_dft_phase_cycle_v1`。
- Canonical U0 配置为 `prototype_target_circular=true`，未注入更强证据时解析为 `cyclic_index_v1`，距离为 `min(|i-j|, 64-|i-j|)`。
- Full-pool workflow 已有经 15-domain 回放绑定的 `ula_dft_phase_cycle_v1` topology manifest。正式 transition 应加载该 audited topology；仅有 `circular=true` 只能用于单元测试/诊断，不能冒充物理 codebook 证据。
- 当前 U0 的 topology-aware 项是 circular soft-label BPA；通用基础 task loss 仍是 beam CE。新 final topology loss 必须调用同一 topology position/distance 口径。

## Channel 文件审计

全量打开本地 47,100 个 `*_paths.npz`，结果如下：

| 项目 | 结果 |
| --- | --- |
| 可读文件 | 47,100 / 47,100 |
| 字段 | `a,tau,theta_t,phi_t,theta_r,phi_r,glob_theta_t,glob_phi_t,glob_theta_r,glob_phi_r` |
| `a` | `[1,1,16,1,64,L,1]`, `complex64` |
| `tau` | `[1,1,1,L]`, `float32` |
| 实际 `Nr` | 16 |
| 实际 `Nt` | 64 |
| 路径数 `L` | 1--9 |
| shape/key 错误 | 0 |

`Nt/Nr` 来自 `a.shape`，未使用目录名推断。`a` 的 path axis 与 `tau` 最后一维逐文件一致。

## Channel 与 label 约定

Current source 已不保留 channel→beam-power 生成模块，但 prepared manifest 标记 `beam_label_source=beam_power_argmax`。历史生成源码使用 64-column normalized ULA-DFT Tx codebook，将 `a` 的 Nt 轴移到首维后把其余 Rx/path 维展平，计算各 beam 投影功率的均值，再取 argmax。该实现：

- 使用 `a` 和 64-beam Tx DFT codebook；
- 不使用 `tau`；
- 不构造 OFDM frequency response；
- 不定义 zero-based/centered 子载波索引；
- 不提供 `Nc` 或子载波间隔。

因此 sparse-pilot 的 path-domain direct calculation 可以复用 `a/tau` shape 和既有 Tx beam拓扑，但不能声称复用了 label 的频率轴。`num_subcarriers=1024`、`subcarrier_spacing_hz=120000` 是待记录的实验假设，不是本地数据实测值。`frequency_index_mode=auto` 必须要求外部 metadata 或一致性证据；当前数据上应显式选择 `centered`/`zero_based` 并把选择写入 resolved config、cache key 和报告。

## 修改边界

计划修改或新增：

- `src/kd_sensing/channel/`：codebook、path-domain simulator、cache；
- `src/kd_sensing/data/samples.py`、`src/kd_sensing/data/datasets/mmw.py`：只读最后历史 channel reference 与时间身份；
- `src/kd_sensing/models/`：selector、SparsePilotEncoder、PrototypeTransition；
- `src/kd_sensing/baselines/` 与 `tools/`：受 clean/full-pool protocol 约束的本地训练、评估、cache 与 ablation workflow；
- `tests/`：物理约束、数值一致性、时间泄漏、train-only lookup、fallback 与 cache invalidation；
- `outputs/sparse_pilot_transition/`、`outputs/cache/`：生成 codebook、cache、lookup、diagnostics 和 summary，保持本地产物边界。

不会新增 public CLI、canonical MMW recipe、完整 CSI tensor、在线 selector 搜索或目标帧 channel 输入。

## 短程诊断与剩余风险

- 以 Full-pool train/validation 各前 100 个样本、2 epoch、单 seed 完成 Stage A smoke；15 个 sensing mask 均评估 CSI off/on，outer test 未访问。
- 10 dB 下 C0 与 C5 Top-1 均为 0.18；C5 Top-3/Top-5 分别为 0.47/0.66，Fix/Harm 均为 0。learned lookup 未优于 fixed/random，因此两个预注册升级门槛均未通过，不启动长训练。
- CSI-off 在 15 个 mask 上的 `fallback_max_abs_error` 为 0；全 dropout 时平均 alpha 为 0，-10 dB 时平均 alpha 约为 0.022。
- 主预算仅覆盖 M=4、Kp=8（4 次空间 sounding、32 个 pilot RE）；尚未执行 M/Kp 网格、多 seed、完整 validation 或 outer-test 评估。
- `1024/120 kHz/centered` 仍是显式实验假设，不能宣称由 MMW channel metadata 验证。
- `conda run -n kd_mm_beam pytest -q` 为 360 passed；新增文件定向 Ruff、CLI/config、compile 和 OpenSpec all-strict 均通过。`make verify-quick` 的 architecture/OpenSpec 子项通过，但全仓 lint 被未修改的 `tools/run_router_observability.py` 中既有 `_amp` 未使用导入阻断。

## Dense-to-sparse 后续诊断

在首轮 T4x8 未提升 Top-1 后，使用独立产物根运行 `D32x16 -> D16x16 -> S8x8 -> T4x8`，每阶段 2 epoch，train/validation 各 100 样本。四阶段共享冻结 U0、CSI 模块、selector 与 optimizer；Kp=8 是同一 16 点母网格的偶数位置子集。

| 阶段 | Sounding | Pilot RE | Top-1 | Top-3 | Fix | Harm | alpha |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D32x16 | 32 | 512 | 0.17 | 0.45 | 0.00 | 0.0556 | 0.2441 |
| D16x16 | 16 | 256 | 0.17 | 0.44 | 0.00 | 0.0556 | 0.1729 |
| S8x8 | 8 | 64 | 0.17 | 0.43 | 0.00 | 0.0556 | 0.1100 |
| T4x8 | 4 | 32 | 0.18 | 0.43 | 0.00 | 0.0000 | 0.0557 |

C0 Top-1/Top-3 为 0.18/0.43。较完整 pilot 提高了 Top-3，但没有提高 Top-1，且 dense 阶段出现 Harm；最终稀疏阶段退回 C0 决策。该结果不支持“当前失败主要由 pilot 数量太少造成”的即时解释，但 dense 阶段只有 2 epoch，不能排除更长的 dense-only 学习需要。当前仍不得启动正式长训练、multi-seed 或 outer test。

四阶段的 `global_route_ratio` 均为 0，Fix 均为 0；D32x16 的 alpha 已达到 0.2441，却没有形成正确纠错。这说明 richer pilot 已被 reliability 分支采纳并改变预测，但当前 transition/route 没有把它转化为正确的 prototype 迁移。后续若继续，应先做 matched-update 的 dense-only/target-only 对照和 route supervision 诊断，而不是继续增加 sounding。dense-to-sparse 后的全量回归为 364 passed、10 个既有 warning；OpenSpec all-strict、CLI/config/architecture、compile 与本次文件 Ruff 均通过。

## Matched-update pilot budget 对照

为消除 curriculum 各预算停留时间不等的混淆，在 GPU0--3 以同一 seed、train/validation 各 100 样本、总计 8 epoch 和同一 C0/C5 目标并行运行四路对照。四路读取同一 train-only codebook，输出根彼此独立；返回码均为 0，outer test 均未访问。

| Arm | M | Kp | Pilot RE | C0 Top-1 | C5 Top-1 | Top-3 | Top-5 | Fix | Harm | alpha | Global route |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense32x16 | 32 | 16 | 512 | 0.18 | 0.18 | 0.43 | 0.66 | 0.00 | 0.00 | 0.0668 | 0.00 |
| spatial4x16 | 4 | 16 | 64 | 0.18 | 0.18 | 0.43 | 0.66 | 0.00 | 0.00 | 0.0574 | 0.00 |
| target4x8 | 4 | 8 | 32 | 0.18 | 0.18 | 0.43 | 0.66 | 0.00 | 0.00 | 0.0574 | 0.00 |
| curriculum | 4 | 8 | 32 | 0.18 | 0.18 | 0.43 | 0.66 | 0.00 | 0.00 | 0.0557 | 0.00 |

Dense32x16 把 pilot RE 从 32 增到 512，但 Top-1/3/5、Fix/Harm 均未变化；4x16 与 4x8 也完全一致，curriculum 相对 4x8-only 没有收益。因此在当前等更新量诊断内，不支持“pilot 数量太少是主要瓶颈”。四路 `global_route_ratio=0`，alpha 仅随预算小幅变化，表明 CSI 可靠度支路读取到了预算差异，但最终决策没有采用 global transition。下一步应检查 route target、route loss/阈值及 transition logits 的梯度和幅度，不应继续单纯增加 pilot。

Matched-update 后全量回归为 367 passed、10 个既有 warning；本功能聚焦测试 28 passed，CLI/config 11 passed，architecture 7 passed，compile 151 个文件，OpenSpec all-strict 6/6 通过。

## 样本与轮次扩展诊断

为检验“样本数和训练轮次太少”，按用户授权在 GPU0--3 并行运行四路单 seed development diagnosis。每路使用 15-domain 确定性均衡且域内无重复的 2,000 个 train、1,000 个 inner validation，batch 8、40 epoch、10,000 optimizer steps；四路共享 U0、seed、codebook、loss 和样本索引。D32x16、4x16、T4x8 分别训练 40 epoch，curriculum 按四阶段各训练 10 epoch。四路返回码均为 0，未访问 outer test，也未启动 multi-seed 或 Stage B。

| Arm | Pilot RE | C0 Top-1 | C5 Top-1 | Fix | Harm | Local | Global | Transition | alpha |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D32x16 | 512 | 0.259 | 0.259 | 0 | 0 | 0.048 | 0.020 | 0.047 | 1.12e-6 |
| 4x16 | 64 | 0.259 | 0.259 | 0 | 0 | 0.037 | 0.030 | 0.041 | 7.07e-7 |
| T4x8 | 32 | 0.259 | 0.259 | 0 | 0 | 0.037 | 0.022 | 0.040 | 7.51e-7 |
| Curriculum | 32 | 0.259 | 0.259 | 0 | 0 | 0.054 | 0.023 | 0.056 | 7.55e-7 |

四路总损失均从约 1.11 降至 1.00，梯度非零，说明训练循环和参数更新实际发生；但从第 5 epoch 到第 40 epoch，validation C5 Top-1 始终与 C0 相同。训练子集 C0/C5 Top-1 均为 0.8595、route target 阳性率约 0.083，validation 则分别为 0.259 和 0.454，显示明显的 train/validation 泛化落差。validation 上 local/global/transition 分支 Top-1 均低于 0.06，reliability gate 因而把 alpha 压到 1e-6 量级，最终 argmax 完全不变。

短程 100-sample validation 的绝对 Top-1 为 0.18，本轮 1,000-sample validation 为 0.259；由于 C0 与 C5 同步变化，这只是不同验证样本集合的基线差异，不能归因于扩样或训练轮次。可归因的 C5-C0 净增益在两种规模下均为 0。因此，本轮证据不支持“当前失败主要由样本数或训练轮次太少造成”，也再次排除 pilot 密度是主要瓶颈；更直接的问题是 CSI transition 分支没有学到可泛化的 future-beam 决策。下一步若继续，应优先检查 train/validation 的 transition target 分布、样本难度分层、监督定义与表示泛化，而不是继续盲目增加 epoch。

## 严重模态缺失 CSI 兜底

用户明确目标后，重新核对发现旧 runner 虽然给 train 样本循环分配了 15 个 mask，但每个样本固定只见一个均匀 mask，transition 不读取显式模态可用度，local/global 分支都依赖退化 sensing feature，CSI 分支没有直接分类监督，逐 epoch validation 和主表又只看 Full。该实现不能把 sparse CSI 明确训练成严重缺失兜底。

修正版保持 U0 冻结，训练 mask 按可用感知模态数固定分配为 `1/2/3/4 = 50%/35%/10%/5%`，在同一 cardinality 内均衡并用固定 seed 打乱。2,000 个 train 样本的精确计数为 `1000/700/200/100`，四路 schedule SHA256 均为 `347baa42b54985a8903235beb8e7ea2ff9a225209e048b29896787b2f92d8abc`。PrototypeTransition 新增只读取 SparsePilotEncoder feature 的 CSI-only prototype distribution、直接 CE/topology loss、sensing availability-aware reliability gate 和 gate target；CSI unavailable/全 dropout 仍逐元素回退到 `p0`。逐 epoch validation 改为 100% single-only，最终继续评估全部 15 masks 的 CSI off/on。

首轮在 GPU0--3 使用相同 2,000/1,000、batch 8、40 epoch、10,000 steps 运行 D32x16、4x16、4x8 和 curriculum；为落实“先完整再逐渐稀疏”，随后在相同模型/loss/索引/schedule 下补齐 D16x16、S8x16、S16x8、S8x8 四个独立 arm。八路返回码均为 0，outer test 和 future channel 均未访问，trainable-only checkpoint 不含 U0。

| Arm | Pilot RE | CSI-only Top-1 | Single Delta | Single Worst Delta | All-14 Delta | Full Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D32x16 | 512 | 0.206 | +0.00150 | 0 | +0.00043 | 0 |
| D16x16 | 256 | 0.107 | 0 | 0 | +0.00014 | 0 |
| S8x16 | 128 | 0.013 | 0 | 0 | -0.00007 | -0.001 |
| S16x8 | 128 | 0.080 | +0.00025 | 0 | +0.00014 | 0 |
| S8x8 | 64 | 0.029 | 0 | 0 | +0.00014 | -0.001 |
| S4x16 | 64 | 0.012 | 0 | 0 | 0 | -0.001 |
| T4x8 | 32 | 0.010 | 0 | 0 | -0.00007 | -0.001 |
| Curriculum T4x8 | 32 | 0.016 | 0 | 0 | -0.00007 | -0.001 |

D32x16 的 single-only curve 在 epoch 20 首次从 0.168 提高到 0.173，epoch 30 达到 0.176，但固定 epoch 40 为 0.173；没有用 validation 选择 checkpoint。全 15-mask 下，D32x16 在 image-only 和 GPS-only 分别净提高 0.001/0.005，Single Macro 由 0.1725 到 0.1740，Full 不变，但 radar-only worst-case 仍为 0.011。该结果只说明高预算 CSI 存在很弱的兜底信号，不能称为稳健兜底。

预算梯度主要由空间 sounding 数决定：同为 128 RE，S16x8 的 CSI-only Top-1 为 0.080，而 S8x16 仅 0.013；16-pattern 两路保持 Full，不超过 8 patterns 的多数路线在 Full 伤害 1/1,000。目标 T4x8 的 CSI-only Top-1 只有 0.010，Single Macro/Worst 均无增益且 Full 从 0.259 降至 0.258。因此当前 4 sounding/32 RE sparse CSI 未达到严重模态缺失兜底目标；增加缺失训练、直接 CSI 监督和 40 epoch 仍不足。下一步若继续，不应再增加 epoch，而应先改进低 sounding 下的可辨识 pilot/encoder 或引入与当前帧 channel 对齐的物理辅助监督，并为 gate 提供完整 4-bit mask/CSI utility 证据；这些属于新设计，不能从本轮结果自动启动。
