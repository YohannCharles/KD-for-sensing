## Context

本地 `dataset/MMW` 已有 sunny、rainy、foggy 三个 condition，每个 condition 包含 5 个 Town03 scenario 的 frame manifest、H5/P1 sequence CSV、image/LiDAR cache 和 sample LMDB。每个 condition 的五场景 train/test 窗口数分别为 `442/110`、`532/132`、`398/100`、`322/80`、`398/100`。当前存在四个阻断：

1. 15 个 split 的结构性 frame/window/guard-band 诊断已生成，但 `future_label_sequence_reuse_ratio>0` 被直接当作 strict failure；对 P1 来说该量只是 beam 类别重复率，不等价于样本或轨迹泄漏。
2. rainy/foggy 使用 `metadata_h5p1.json` 与 `sanity_report_h5p1.json`，旧 availability writer 未识别后缀，因而权威 readiness 仍为 pending。
3. manifest 可定位 RSU radar JSON，但 15 个 scenario 尚无 `derived/radar_maps` 和带 radar 列 split CSV，不满足 sensor-assisted 四输入契约。
4. data factory 可拼接同 condition 的多个 scene，但不能由显式 condition/scenario domain 列表构建 15-domain pooled dataset，也没有 domain-balanced sampler。

用户已停止 DeepSense S1/T2 seeds2/3，并保留 GPU4-7 的 LG/CLS seed1。GPU0-3 用于 MMW 四方法 seed1。MMW 本轮只使用 image、GPS、LiDAR、radar sensing inputs；CSI、channel、beam power 和历史 beam 均不进入模型输入。

## Goals / Non-Goals

**Goals:**

- 让 15 个 MMW H5/P1 domain 通过可解释、可机器读取的 preparation preflight。
- 用显式 domain config 构建 pooled train/validation dataset，并按 domain 等权采样。
- 在相同 sensing input、split、缺失增强、训练预算和 checkpoint policy 下比较 S1、T2、AMBER-Full 与 RMBP-MM。
- 输出 per-domain、per-weather、per-scene、macro 与 worst-domain 的 clean/whole-modality/temporal-missing 指标和 reliability diagnostics。
- 先验证 T2 与现有 reliability fusion，不提前增加天气标签输入或天气专用网络。

**Non-Goals:**

- 不将 CSI/channel/mmWave/beam-power/path/radio label 作为 sensor-assisted sensing input。
- 不恢复 Hist/HiST、Raymobtime 或其它 retired 路由。
- 不用 DeepSense 多 seed 或尚未完成的 PPT 作为 MMW 启动条件。
- seed1 validation screening 不升级为正式 test claim；最终 claim 需要独立 group-safe test 与 seeds1-3。

## Decisions

### 1. future-label reuse 只作诊断，不单独否决 P1 split

`future_label_sequence_reuse_ratio` 继续写入 metadata，用来描述标签模式重复；strict eligibility 只由 train/test frame overlap、完整或近完整 window overlap、adjacent-window cross-split、guard-band violation 和空 split 等结构性条件决定。替代方案是改用 P6 降低标签序列重复，但这会改变当前 T2 的 next-beam P1 任务，不能解决错误的泄漏定义。

### 2. 复用公开 preparation/radar utility

不在 dataset loader 中静默生成 radar。先用 `mmw_sequence_splits_from_manifest` 重写独立 strict tag，再用 `mmw_radar_maps` 为 15 个 scenario 生成 RA/DA maps、materialized CSV 和 metadata；preflight 必须检查 H5/P1 后缀 metadata/sanity、split eligibility、radar columns 和文件存在性。原始 H5/P1 与 zip 不覆盖、不删除。

### 3. 显式 domains 列表而不是拼接跨天气 CSV

配置使用 `data.dataset.domains`，每项至少包含 `id`、`condition`、`scene`、`data_root`、`train_csv_name` 和 `test_csv_name`。data factory 对每项应用同一基础 dataset config 后构建 leaf dataset，再形成带 domain provenance 的 `ConcatDataset`。不创建跨 condition 的绝对路径巨型 CSV，避免丢失 MMW family metadata。

### 4. 训练按 domain 等权，评估按 domain 宏平均

训练 sampler 对每个样本赋权 `1 / domain_sample_count`，每 epoch 抽样数等于 pooled train 总样本数；天气和场景样本量不影响期望 domain 权重。评估不采样，逐 domain 记录后先做 domain macro，再计算 weather/scene macro 和 worst-domain。替代方案是普通 shuffle，但会让 Tjunction 等较大 domain 获得更高权重。

### 5. seed1 screening 使用固定 epoch 的 last checkpoint

当前 prepared split 只有 train/test。第一阶段把 frozen test CSV 作为 validation screening，四方法使用相同固定 epoch、关闭 early stopping，并统一评估 `last.pth`，不得根据 validation 指标选择 best checkpoint。该结果只标记 local validation evidence。通过运行与评估 smoke 后，再为最终多 seed claim 增加独立 group-safe validation/test 协议。

### 6. 不新增天气模块

S1/T2 保持 image、radar、GPS、LiDAR 编码与 reliability router；T2 只增加训练期 confidence-gated temporal superset KL。天气不作为 router 输入。首轮额外记录每个 weather/domain 的 modality gate、entropy、calibration 和 missing delta；只有可复现的过度自信或 domain 负迁移才触发后继模型 change。

### 7. baseline 以“本地适配”而不是“论文等价复现”报告

AMBER-Full 与 RMBP-MM 可以复用当前 registry core，但必须记录原论文输入、训练阶段、时序能力、损失和评估协议与本地实现的差异。AMBER 原文是带历史 beam 的五模态时序模型；本实验为四 sensing modality、无历史 beam、较小 transformer 和近似辅助损失，因此只能标记为 `amber_full_local_adaptation`。RMBP-MM 原文是单时刻五模态模型，包含不可缺失的 partial beam、单模态预训练和基于真值 beam label 的离线相似样本补全；当前实现只有四传感器 channel-attention fusion，因此只能标记为 `rmbp_mm_channel_attention_local`。

RMBP-MM 原文没有 temporal aggregation，不能把 H5/P1 最后一帧 logits 上的 temporal dropout 曲线称为论文等价结果。该曲线只保留为 out-of-paper-scope diagnostic，并必须同时报告末帧是否可用；whole-modality 结果也需保留本地适配 caveat。

### 8. temporal missing 使用几何覆盖而不是每比例三张掩码

0% 缺失只评估一个 clean mask。20/40/60/80% 对 `modality_frame` 使用固定数量的互异随机掩码，对 `frame_level` 与 `block` 枚举 `seq_len=5` 下全部互异几何；所有方法共享同一 v2 cache。summary 先对每个 mask 做 15-domain macro，再在 mask type 内平均，并报告 mask 间标准差、最差值、末帧不可用计数和实际缺失率；跨 type 总均值采用 type-equal average，避免可枚举几何数量不同导致隐式加权。

评估在运行前必须校验 cache 的 rate、shape、type、互异数量和完整支持覆盖，旧的三掩码 cache 不得静默复用。RMBP-MM 的 diagnostic 需要按 `last_frame_available` 分层，避免把最后一帧读取策略造成的崩溃误解为随机缺失率的稳定退化。

### 9. 85/90/95% 作为 modality-frame 极端稀疏扩展

H5/P1 输入共有 `5 time steps × 4 modalities = 20` 个 modality-time cells，因此 85%、90%、95% 可以精确表示为丢失 17、18、19 个 cells，分别只保留 3、2、1 个 cells。该扩展只使用 `modality_frame` mask，并与现有 80% `modality_frame` 结果比较。

`frame_level` 与 `block` 在 5 帧窗口下只能产生 20% 的离散步长；85%/90% 会重复 80% 几何，95% 会退化为全缺失后再由 fallback 修补，不能作为对应 rate 的公平实现。因此极端稀疏结果 MUST 单独报告，不进入原 0–80% 三类型 type-equal 主曲线。

### 10. 融合表征诊断使用方法内 clean PCA 与原空间配对指标

T2、AMBER-Full、RMBP-MM 都提取实际送入 beam head 的融合表征：T2 使用二维 `output_features`，两个 modular baseline 使用最后预测时刻的 `output_features[:, -1, :]`。三者当前均为 64 维，但独立训练模型的潜在坐标轴可任意旋转，因此不得把三种方法的原始特征直接叠在同一 PCA 坐标系后比较绝对位置。

每种方法先对 clean 融合表征做 L2 normalization，再只用该方法 clean 样本拟合确定性 PCA；20/40/60/80% 使用共享 v2 cache 中全部 `modality_frame` 固定 mask，并投影到同一方法的 clean PCA 基底。图片展示同一样本在 clean 与各 rate 的 mask-mean 表征之间的位移；定量结论在原始 64 维归一化空间计算 paired cosine distance、最近 clean beam centroid 的圆周 beam 距离、clean-to-missing centroid assignment 保持率和预测 beam 圆周偏移，并按 mask 报告 mean/std/worst、按 domain 报告分层结果。

PCA 仅用于展示，不以二维距离代替高维指标。T2 与两个不同架构 baseline 的比较可以支持 prototype-aligned T2 与更稳定表征相关，但不能单独归因于 Beam Prototype Alignment Loss；因果 claim 仍要求同一 T2 架构、相同训练协议下关闭 alignment loss 的消融。该诊断继续标记为 seed1 local validation，AMBER/RMBP 保留 local-adaptation scope。

### 11. 高维循环拓扑使用原空间证据与固定无监督流形投影

普通 PCA 最大化全局方差，不保证保留一维闭合流形；因此 prototype 或 class centroid 在二维 PCA 中不形成圆，不能据此否定原始 64 维循环邻接。主证据直接使用 L2-normalized 64 维向量的 cosine Gram matrix、cosine similarity 随 circular beam distance 的衰减、原空间 nearest-neighbor Topo@1/Topo@3，以及缺失前后 nearest-clean-centroid assignment 的 signed circular shift 分布。

为提供直观二维图，prototype 使用只由原始 cosine distance 构建的 2-NN Isomap；每个节点在一维闭合流形上固定使用两个局部邻居。三方法先用 validation 真值标签构建 label-conditioned clean class centroid，再分别使用相同的 3-NN Isomap；`k=3` 是让三方法图全部连通的最小共同邻居数，不按方法调参。prototype Isomap 坐标不得消费 beam label；centroid Isomap 在 centroid 构建后也只消费 centroid distance，但图注必须披露 centroid 本身由标签聚合。beam index 继续用于着色、anchor 标注和允许全局旋转/镜像的 phase consistency、angular MAE 量化。附带 Laplacian Eigenmap 或 k 敏感性只作稳健性检查，不新增 sklearn/UMAP 依赖。

T2 prototype 图与三方法 clean centroid 图承担不同问题：前者展示 learned prototype bank 的高维结构，后者才允许跨方法比较。missing 图固定使用各方法自己的 leave-one-out clean centroid bank，在原始 64 维空间对 clean/missing 表征做 assignment，再先按 domain 汇总 `-32..31` signed circular offset、后对 15 domain 等权平均；所有方法和 rate 使用同一色标与同一 mask/sample inventory。该图可以证明高维邻接与缺失偏移现象，但在 matched T2 no-alignment checkpoint 缺失时仍不得把差异因果归于 Beam Prototype Alignment Loss。

## Risks / Trade-offs

- [Radar 预处理约 45k 帧并增加数 GB 产物] → 串行按 condition 生成，先 dry-run/单场景 smoke，产物留在 `dataset/MMW/*/Prepared`。
- [GPU4-7 训练仍占用共享 CPU/I/O] → radar 预处理限制为单进程，GPU0-3 训练只在 preflight 完成后启动。
- [sample LMDB 不含新 radar] → 四方法关闭旧 sample LMDB 或重建带 radar 的 cache，禁止混用无 radar cache。
- [固定 epoch 对不同 baseline 未必各自最优] → seed1 只用于流程与方向筛选；正式阶段使用独立 validation，但四方法保持相同 selection contract。
- [P1 标签重复诊断不再否决 split] → metadata 继续报告 reuse ratio，并用 frame/window/adjacency/guard-band 四类结构诊断作为真正泄漏门禁。
- [domain-balanced replacement sampling 改变每 epoch 样本重复率] → metadata 记录 sampler、权重和样本数；同时保留普通 macro evaluation，禁止把 sampler 影响隐藏在模型结论中。
- [结构化 mask 的有限支持大小不同] → frame/block 枚举全支持，跨类型使用 type-equal average，并单独输出每类 mask 数与方差。
- [论文 baseline 与本地四传感器协议不等价] → provenance 和 summary 固定输出 reproduction scope；RMBP temporal 只作诊断，不进入论文等价主比较。
- [极端缺失率无法由整帧几何精确表示] → 85/90/95% 只运行 modality-frame exact-cell masks，并在报告中明确剩余 cell 数。

## Migration Plan

1. 修正 split eligibility 并用 synthetic test 证明 P1 类别重复不等于 frame leakage。
2. 生成新 strict tag，保留现有 `h5p1/` 不动；核对 15 个 metadata。
3. 串行生成 radar maps 与 materialized split CSV，执行单样本四模态 shape smoke。
4. 实现 domains/ConcatDataset/domain-balanced sampler 与 provenance tests。
5. 生成四方法配置并先 dry-run、每方法 1 batch smoke，再在 GPU0-3 运行 seed1。
6. 固定缺失 cache 评估并输出 validation summary；任何 preflight/eligibility 失败都停止训练。

回滚只需停用新 domain config 和本地 launcher；原 split、原始数据、旧 cache 和现有模型配置均不被覆盖。
