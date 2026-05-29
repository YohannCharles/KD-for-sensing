## Context

当前 HiST-Beam 实现已经有可运行闭环，但仍是快速验证系统：

- `HistBeamFusionNet` 使用 image/radar/gps 默认模态、CLS-token Transformer fusion、flat head、shared/private branch、coarse/fine head 和 zero-init private adapter。
- 当前 loss 覆盖 hierarchical CE、flat auxiliary CE、orthogonality、shared scene confusion、private scene preservation，但没有 angular smoothing、几何一致性或显式 RSU-CAV 相对几何。
- 当前 prototype 产物按 coarse group 保存 shared/private 均值；adaptation 实际使用 `shared_representation` 对齐 `shared_prototypes`，未使用 private prototype 或 fine mapping 语义。
- 当前 `apply_hist_beam_adaptation_strategy` 对 adapter 变体只训练 `private_adapter` 与 `fine_head`，这是合理的轻量更新边界，但不足以解释可迁移知识与 scene-private 知识如何分工。
- DeepSense6G quick validation 的 `loso_summary.json` 与 `quick_validation_conclusion.json` 显示：完成的 `v5_adapter_proto` 与 `v4_adapter` 指标完全相同；scene31/33 的 budget=10 还因 beam power label NaN/Inf 失败。因此当前结果只能作为“需要重设方法”的证据，不能作为论文主张。
- MMW sunny zip 已下载到 `dataset/_downloads/MMW/sunny`，dry-run 已确认传感器 zip 约 9.2G、channel zip 约 405M，输出位置解析为 `dataset/MMW/sunny/Prepared/Town10_skybridge_seed24`。本轮已用当前脚本生成 prepared 产物：3600 个有效 frame、3570 个 sequence window、train/test 为 2856/714，覆盖 `cav_1`、`cav_2`、`cav_3`。但 metadata 同时暴露了一个必须修复的问题：sensor 场景名是 `Town10_skybridge_seed24`，channel 场景目录是 `Town10/Town10_skybridge`，当前 matcher 未正确解析 channel 下的 CAV agent，出现 `cav_1` frame 匹配到 `channel/.../cav_3/...` 的情况。因此当前 prepared 产物只能作为结构审计，正式训练前必须修复 channel agent 匹配并强制重建。

本设计目标是把 HiST-Beam 从“结构上叫 shared/private”调整为“可观测几何与角度语义驱动的跨场景自适应”，同时避免论文过度声明。

## Goals / Non-Goals

**Goals:**

- 建立清晰的 transferable knowledge 与 scene-private knowledge 定义，并把每一项映射到可实现字段或可审计 proxy。
- 为 Multimodal-Wireless 设计可增量构建的 prepared manifest、geometry feature、scenario/town/weather split 和 target adaptation protocol。
- 扩展 HiST-Beam 模型与 loss，使 coarse angular semantics、angular continuity、relative geometry 和 cross-modal geometry consistency 成为可训练、可诊断的约束。
- 修正 prototype alignment，使 `v5` 类变体必须有与 `v4` 可区分的 loss 输入、coverage、confidence 和梯度路径。
- 保留 DeepSense6G 31-34 LOSO 作为历史回归和消融基线，同时把新论文设定优先放到 Multimodal-Wireless。

**Non-Goals:**

- 不自动下载、移动、删除或提交任何真实数据、cache、checkpoint 或训练输出。
- 不把不可观测的真实 scatterer、occluder 或 town semantics 直接声明为已监督建模；只能使用 LiDAR/bbox/radar/depth/channel 等 proxy。
- 不承诺单个 sunny 场景能支撑跨场景结论；单场景阶段只做数据准备、loader smoke、within-scenario sanity 和方法接口验证。
- 不在本变更中推翻现有 DeepSense6G 配置入口；旧 quick validation 应继续可运行，作为回归检查。

## Decisions

### 1. 知识定义必须绑定可观测字段

**Decision:** 将论文概念分为“可直接实现”和“proxy 实现”两类，并在 metadata、metrics 和论文描述中保持同一口径。

可直接实现：

- coarse angular/beam semantics：由 64-beam codebook、beam power argmax、coarse sector 标签、V2I channel path 派生。
- angular neighborhood continuity：由 beam index/codebook 邻接图或 beam steering angle 距离构造 soft target。
- RSU-CAV relative geometry：由 CAV/RSU pose、GPS/IMU、heading、frame transform 派生 range、azimuth、elevation、relative velocity 和 local-frame 坐标。
- cross-modal geometric consistency：由 RGB/depth/LiDAR/radar point cloud/bbox 与 pose 投影关系、centroid/range 一致性和 channel-derived bearing 近似构造。

只能作为 proxy：

- town/scene layout：使用 scenario/town/weather id、RSU pose、静态帧统计和局部点云 occupancy 摘要，不声明真实 HD map semantics。
- local scatterer/occluder：使用 LiDAR occupancy、bbox、depth discontinuity、radar point cloud density、LoS/path count 或 channel energy spread proxy。
- fine beam mapping within coarse sector：使用 coarse sector 条件下的 private representation 和 fine head，不声明跨场景完全共享。

**Alternatives considered:** 继续使用 shared/private branch 名义拆分。拒绝原因是它不能解释 v4/v5 指标一致，也不能利用 MMW 的几何字段。

### 2. MMW 数据协议先 manifest，再训练

**Decision:** MMW 准备流程必须先生成 frame-level manifest，再从 manifest 派生 sequence CSV 和 split。manifest 是跨场景适配的事实来源。

manifest 至少记录：

- condition、town、sensor scenario、channel scenario、agent、frame id、sample id。
- CAV 路径：LiDAR pcd、四路 RGB camera、depth camera、GPS/IMU yaml、bbox、radar point cloud。
- RSU 路径：RSU yaml/pose、RSU LiDAR/camera/depth/radar 可用性。
- V2I channel paths 文件、派生 beam power 文件、beam label、coarse sector。
- CAV/RSU relative pose 和局部坐标 frame metadata。
- modality availability 与 skip reason。

当前 sunny 数据可先处理为 `prepared_status: single_scene_ready`；其它三组数据下载中时记录 `pending`，不得生成虚假的 LOSO fold。

**Alternatives considered:** 直接复用 DeepSense6G sequence CSV 作为唯一数据入口。拒绝原因是 CSV 不足以稳定表达 RSU-CAV 对齐、多模态可用性和 scenario/town/weather split。

### 3. Geometry-aware shared encoder 与 scene-private branch 分工

**Decision:** 在现有 `HistBeamFusionNet` 基础上增加 geometry-aware 输入路径，而不是重写全部 fusion 模型。

结构建议：

- 每个模态 encoder 输出 token：image/depth/LiDAR/radar/GPS/IMU/channel-derived features。
- 新增 geometry token：`relative_range`、`relative_azimuth`、`relative_elevation`、CAV/RSU heading 差、速度、local-frame 坐标、可用性 mask。
- shared encoder 输出 `c_geo`，服务 coarse angular semantics 和 geometry-invariant representation；coarse head 只读取 shared representation。
- scene-private branch 输出 `s_scene`，接收 scene layout proxy、RSU pose/local frame proxy、occluder proxy；fine head 读取 `concat(c_geo, adapter(s_scene, coarse_context))`。
- fine mapping adapter 改为 coarse-sector aware adapter，可选择按 coarse group 使用小型 gate 或 embedding 条件化 adapter。

**Alternatives considered:** 新建完全独立的 MMW 模型。拒绝原因是会丢失当前 HiST-Beam 回归能力，也会让现有 adapter/fine-tune 对比不可复用。

### 4. Loss 组合强调角度与几何，而不是堆名字

**Decision:** 新总 loss 采用显式开关，默认 MMW 方法配置启用几何与角度项，DeepSenseG 回归配置可保持旧行为。

建议形式：

```text
L = L_hier
  + lambda_flat * L_flat
  + lambda_ang * L_angular_smoothing
  + lambda_geom * L_multimodal_geometry
  + lambda_orth * L_orth
  + lambda_scene_c * L_scene_confusion
  + lambda_scene_s * L_scene_private
  + lambda_proto * L_private_proto
  + lambda_ent * L_target_entropy
```

关键约束：

- `L_angular_smoothing` 使用 beam/codebook 邻接图构造 soft target。若 codebook 是线性 ULA，默认使用非循环邻接；只有配置声明 circular codebook 时才使用 circular distance。
- `L_multimodal_geometry` 只能使用当前样本可观测的几何关系；缺失 depth/radar/bbox 时按 mask 跳过并记录 coverage。
- `L_private_proto` 必须对齐 private/adapter representation，而不是 shared coarse representation；必须按 coarse sector、prototype count 和 confidence threshold 过滤。
- `v5` 若 prototype coverage 为 0 或 loss 权重为 0，summary 必须显示为 no-op 或 unavailable，不能只给出 coverage=1.0。

**Alternatives considered:** 保留 entropy + shared prototype consistency。拒绝原因是当前 quick validation 已显示它没有和 adapter-only 形成可观测差异。

### 5. Target adaptation protocol 按数据可用性分级

**Decision:** MMW target adaptation 分为三个阶段，避免单场景阶段过度解读。

- Stage A：single-scene sunny smoke。只验证 zip 处理、manifest、loader、forward/loss、within-scenario split 和小样本 adapter 是否可运行。不得报告跨场景提升结论。
- Stage B：scenario-LOSO。至少两个 scenario 可用后，leave-one-scenario-out；source 为其它 scenario，target 分为 target_adapt/target_test。
- Stage C：town/weather split。多个 town 或 weather 条件可用后，支持 leave-one-town-out、leave-one-condition-out 和混合 source domain。

target protocol：

- `target_adapt` 默认 20%，`target_test` 默认 80%，二者按 sample id / sequence segment 无交集。
- budgets 支持 `0, 5, 10, 20, 50`，优先按 coarse sector 与 relative azimuth bin 分层采样。
- adapter 变体冻结 source encoders、geometry shared encoder 和 coarse head；训练 fine adapter、允许的 fine head 层、private prototype bank 和可选 LayerNorm affine。
- full fine-tune baseline 使用相同 target labeled subset，但仍不得读取 `target_test`。

**Alternatives considered:** 等四个场景全部下载完再开始。拒绝原因是当前可以先把数据准备和接口风险消掉，但必须显式标记单场景状态。

### 6. 评估指标必须支撑论文设定

**Decision:** 除 Top-K 外，新增与设定直接相关的指标。

- beam：Top-1/3/5、coarse accuracy、fine offset accuracy、mean angular error、within-sector accuracy。
- channel power：若有 beam power vector，报告 normalized received power、power loss dB；无 power 时明确 unavailable。
- adaptation：trainable ratio、adapt time、prototype coverage/confidence、prototype loss mean、geometry loss coverage。
- geometry：relative azimuth bin accuracy、range bucket accuracy、cross-modal consistency coverage。
- split：source/target scenario/town/weather、target_adapt/test 样本数、coarse/azimuth 分布。

**Alternatives considered:** 只沿用 Top-K。拒绝原因是无法证明 coarse semantics、角度连续性和几何一致性真的参与了方法。

## Risks / Trade-offs

- **[Risk] MMW 场景下载不完整导致无法做 LOSO。** → 以 data availability manifest 标记 `single_scene_ready`、`pending`、`ready_for_loso`，单场景只跑 smoke/sanity。
- **[Risk] sensor scenario 与 channel scenario 命名不完全一致。** → 准备流程记录 sensor/channel scenario alias，并按 frame id + agent 进行可审计匹配；模糊匹配必须写入 metadata。
- **[Risk] 当前 sunny prepared 产物已暴露 channel agent 错配。** → 将 channel agent 解析修复列为首个任务；修复后使用 `--force` 重建 `dataset/MMW/sunny/Prepared/Town10_skybridge_seed24`，并新增断言检查 frame manifest 中 `agent` 与 `channel_path` 的 CAV agent 一致。
- **[Risk] 几何字段格式不稳定。** → 先实现 schema audit 和字段适配层，缺失字段按 mask 跳过，不让 loss 读取不存在的模态。
- **[Risk] proxy 被误写成真实语义。** → 在 spec、metadata 和报告中固定 direct/proxy 标签，论文写作只能引用同一口径。
- **[Risk] 新 loss 增加训练不稳定。** → 所有新增 loss 默认有 warmup、coverage 诊断和独立开关；先做 single-scene smoke，再扩大到 LOSO。
- **[Risk] prototype 再次 no-op。** → 强制记录 prototype loss、coverage、confidence、used sample count、trainable prototype 参数和 v4/v5 对比；coverage 为 0 或未生效时结论必须标记 inconclusive。

## Migration Plan

1. 保留现有 DeepSense6G HiST-Beam 配置和 quick validation 输出格式，新增 MMW 专用配置与 dataset descriptor。
2. 扩展 MMW preparation：从 `_downloads/MMW/<condition>` 或显式 zip 路径生成 `dataset/MMW/<condition>/Prepared/<scenario>`，写入 enriched manifest。
3. 增加 geometry feature builder 和 MMW batch profile；先用 sunny single-scene smoke 验证 dataset、forward 和 loss。
4. 在 HiST-Beam 中增加 geometry-aware shared/private outputs、新 loss 与 diagnostics；默认旧配置关闭新增项。
5. 扩展 LOSO planner/executor：支持 MMW scenario/town/weather fold；当可用场景不足时只生成 smoke plan。
6. 新场景下载完成后运行 incremental prepare，再运行 scenario-LOSO 和 target adaptation matrix。

回滚策略：新增配置和模型开关保持 opt-in；若 MMW geometry path 不稳定，可关闭 geometry loss/prototype loss，继续运行旧 HiST-Beam 回归。

## Open Questions

- MMW 其余三个下载中的场景是同一 Town10 下的不同 scenario，还是不同 weather/town 条件？这会决定默认主实验是 scenario-LOSO、town-LOSO 还是 condition-LOSO。
- Channel `_paths.npz` 中可稳定使用的字段是否包含显式 AoD/AoA、path gain 和 LoS/path count？需要在 prepared metadata 中审计后固定。
- RGB/depth/bbox/radar point cloud 在每个 CAV/RSU frame 的命名是否完全一致？若不同，需要在 manifest builder 增加 alias 表。
