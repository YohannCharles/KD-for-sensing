## Context

arXiv:2604.05668《A BEV-Fusion Based Framework for Sequential Multi-Modal Beam Prediction in mmWave Systems》提出将 camera、LiDAR、radar 和 GPS 映射到统一 BEV 空间后做时序 beam prediction。论文主实验聚焦 DeepSense6G scenarios 32、33、34，使用 5 步历史观测、64 类 beam、单步未来标签，报告 S32/S33/S34 DBA 为 86.60%/86.27%/86.70%，overall DBA 为 86.52%。论文还给出关键配置：128×128 BEV grid、ResNet-34 camera backbone、3 层 camera-to-BEV attention、4 heads、temporal transformer 4 层 4 heads、`d_model=256`、AdamW `lr=1e-4`、`weight_decay=1e-2`、focal loss `gamma=2` 和 class-frequency alpha。

仓库当前已经具备 DeepSense6G scene 31-34 选择、image/radar/GPS/LiDAR/mmWave 模态契约、LiDAR BEV cache、ResNet-18 RGB image encoder、canonical fusion、focal loss、2604-style Image+GPS/JEPA 对照复核和统一训练/评估 engine。缺口在于：还没有论文自身的 BEV-space fusion 模型、GPS spatial mask + global MLP 双路径、paper-style ablation 配置和一份能清楚区分“本地复现 split”与“论文未公开 exact split/seed”的报告。

## Goals / Non-Goals

**Goals:**

- 实现可由 `MODELS.build()` 构建的 `bev_fusion_2604` 主模型，并通过现有 `kd-sensing-train`、`kd-sensing-evaluate` 和配置加载流程运行。
- 复用现有 DeepSense6G dataset、LiDAR BEV cache、radar RA/DA、image RGB profile、GPS 读取、focal loss、checkpoint、metrics 和输出目录规则。
- 提供 paper-aligned full config、low-memory config、quick smoke config 和 ablation config，默认目标为 S32/S33/S34 的 5 帧输入、`future_beam1`、64 beam、linear DBA。
- 输出可审计报告：每 scene DBA/Top-K、macro/weighted overall、论文目标值、差距、本地 split/seed、样本数、metric profile、参数量、可选 latency 和 caveat。
- 为真实数据不可用或显存不足时提供 mock/synthetic forward 与小 BEV 尺寸 smoke，验证 shape、loss、metadata 和 report 生成，不冒充真实指标。

**Non-Goals:**

- 不恢复 KD/distillation、HiST-Beam、CRAF、MARF、Raymobtime 或其它已退役路线。
- 不将论文结果写成官方复现成功，除非真实 DeepSense6G 数据、split、seed 和训练记录全部可审计。
- 不新增绕过 `src/kd_sensing` 包结构的长期训练脚本；如需辅助命令，应作为包内 CLI 或当前 console script 的配置化用法。
- 不提交真实数据、LiDAR/cache、训练日志、checkpoint、TensorBoard 或评估输出。
- 不要求现有 canonical fusion 默认模型改为 BEV-Fusion；本模型只作为 2604 复现实验配置族启用。

## Decisions

### Decision 1: BEV-Fusion 作为独立注册模型，而不是改造现有 CLS-token fusion

新增 `src/kd_sensing/models/bev_fusion_2604.py`，注册名为 `bev_fusion_2604`。该模型接收现有 batch preparation 传入的 `image_batch`、`radar_batch`、`gps_batch`、`lidar_batch` 和可选 `gps_bev_xy_batch`，返回 dict，至少包含 `logits`、`input_features`、`output_features`、`bev_features`、`modalities` 和关键 diagnostics。

理由：论文核心是 spatial BEV fusion，不是把各模态压成 1D token 后融合。独立模型能保留现有 `cls_token_transformer_fusion` 作为对照，也避免修改 canonical fusion 默认语义。

备选方案是把 BEV branch 塞进 `CLSTokenTransformerFusionNet`。该方案复用更多代码，但会让一个类同时承担 1D token fusion 与 BEV spatial fusion，难以维护 ablation 和论文报告。

### Decision 2: 主配置贴合论文，smoke 配置缩小 BEV

paper/full 配置默认使用 `bev_size=[128,128]`、`d_model=256`、camera-to-BEV attention 3 层 4 heads、temporal transformer 4 层 4 heads。为了本地开发和 CI，新增 smoke/low-memory 配置允许 `bev_size=[16,16]` 或 `[32,32]`、较小 `d_model`、较少 attention/transformer 层，并在 metadata 中标记 `paper_approximation=true`。

理由：128×128 BEV query 对 camera tokens 的 cross-attention 计算和显存压力明显高于当前 Image+GPS baseline。保留 full 配置是复现目标，smoke 配置是工程验证手段。

备选方案是只实现小 BEV 近似。该方案更容易跑通，但会从设计上失去论文可比性。

### Decision 3: Camera-to-BEV 使用可配置 torchvision backbone，paper 默认 ResNet-34

BEV 模型内部新增 camera backbone builder，paper 配置使用 ResNet-34 并输出 2D feature tokens；smoke 可用轻量 CNN 或 ResNet-18。camera-to-BEV 使用 learnable BEV query `[H*W,d_model]` 对 image feature tokens cross-attention，输出 `[B,T,d_model,H,W]`。

理由：现有 `ResNet18ImageEncoder` 输出帧级 embedding，适合 1D fusion，但论文需要保留 spatial feature map 供 BEV query attention。将 2D backbone 限定在新模型内部，不改变 `resnet18-image-encoder` 的默认契约。

备选方案是复用 `ResNet18ImageEncoder` 的 pooled embedding 再 broadcast 到 BEV。该方案会丢失视觉空间结构，只能作为 1D/broadcast ablation，不应作为主模型。

### Decision 4: LiDAR/Radar/GPS 均映射为同一 BEV tensor

LiDAR 分支复用 dataset 已构造的 LiDAR BEV `[B,T,C,H,W]`，通过轻量 CNN/1x1 projection 转为 `[B,T,d_model,H_bev,W_bev]`；尺寸不一致时用显式 interpolation 并记录原始尺寸。Radar 分支接收现有 radar RA/DA batch，通过 CNN 投影到 BEV grid。GPS spatial pathway 使用 `gps_bev_xy_batch` 或等价未标准化相对 XY 坐标生成 one-hot/soft Gaussian mask，再经 CNN/MLP 投影到 BEV。

理由：现有数据层已能懒加载 LiDAR BEV、radar map 和 GPS feature；模型层只负责把它们对齐到统一 grid。GPS spatial mask 需要未被 StandardScaler 改写的局部坐标，因此实现时应为 BEV 模型增加可选 `gps_bev_xy` batch 字段或在 dataset metadata 中提供可逆坐标来源。

备选方案是在预处理阶段生成所有模态的统一 BEV cache。该方案可提升吞吐，但会扩大数据产物和 cache 管理复杂度；第一阶段保留在线模型投影，必要时再增加 cache。

### Decision 5: GPS dual-path 按论文语义实现，但允许坐标源可配置

GPS spatial pathway 参与每个时间步的 BEV fusion；global pathway 将全精度 GPS 序列输入 MLP/Transformer 得到 `h_gps`，在 temporal BEV fusion 后通过 `z_aug = z_final + tanh(s) * W_gps(h_gps)` 注入，`s` 为可学习标量。配置支持 `gps_spatial_only`、`gps_global_only` 和 `gps_dual_path` 三种 ablation。

理由：论文报告 GPS mask 单路径与 global MLP 单路径均不如双路径。把它做成显式可配置组件，能复现 Table V 类 ablation，也能诊断坐标量化损失。

备选方案是只使用现有 `gps_batch` MLP embedding。该方案实现简单，但不是论文的 dual-path GPS。

### Decision 6: Spatial fusion 先 concat+conv，后续再考虑 cross-modal BEV attention

主模型将 camera/LiDAR/radar/GPS spatial tensors 按 channel concat，然后用 configurable `BEVFusionBlock` 做归一化、卷积和残差融合，得到逐时隙 BEV feature。若某个 ablation 移除模态，模型必须只跳过该分支并记录 effective modalities。

理由：论文强调 BEV 空间融合，但没有开放代码。concat+conv 是最可测试、最容易和 ablation 对齐的第一版；后续可在同一 block 后面增加 cross-modal attention 作为扩展。

备选方案是实现复杂的多模态 BEV transformer。该方案更接近自动驾驶 BEVFusion 文献，但更难在论文细节未公开时判断是否提升复现可信度。

### Decision 7: Temporal transformer 消费 pooled BEV sequence 并输出单 horizon logits

每个时隙的 fused BEV 先通过 spatial pooling 或 attention pooling 得到 `[B,T,d_model]`，加入 time embedding 后进入 temporal transformer。输出最终时隙或 CLS token，经 beam classifier 得到 `[B,1,64]` logits。single-frame 和 mean-pooling temporal 作为 ablation core。

理由：现有 engine 通过 `select_prediction_slots` 支持 `[B,H,C]` logits；单 horizon 与论文 `future_beam1` 对齐，同时不影响未来扩展多 horizon。

备选方案是对每个历史时隙都输出 logits 再取最后一个。该方案兼容 engine，但会引入额外监督语义，第一阶段不采用。

### Decision 8: 训练优先使用现有 focal loss 与 class-balanced metadata

配置使用现有 `focal_loss`，`gamma=2`；若需要论文式 class-frequency alpha，应复用或扩展 class-balanced weight helper，在训练 metadata 中记录 label histogram、alpha/weight 策略和 fit split。优化器使用 AdamW `lr=1e-4`、`weight_decay=1e-2`，并显式记录 augmentation。

理由：仓库已注册 focal loss 和 class-balanced helper，不需要新增 loss 体系。论文中的 horizontal flip 需要同步 beam index reversal；若当前数据增强不支持安全 beam reversal，full 配置先关闭 flip 或将其标为 blocked/experimental，避免错误 label。

备选方案是直接使用 cross entropy。该方案可作为 ablation，但不应作为 paper 主配置。

### Decision 9: 报告区分 paper target、local exact、local comparable 与 smoke

新增 report helper 或约定训练/评估输出字段，记录：`paper_target`、`local_metric`、`metric_profile`、`split_protocol`、`sample_count`、`scene_breakdown`、`macro_dba`、`weighted_overall_dba`、`topk`、`params_m`、可选 `latency_ms`、`paper_exact_split_available` 和 `mock_data`。

理由：论文未给出 exact split index/seed 且代码权重未开放。报告必须避免把本地 stratified split 或 smoke 结果写成论文 exact reproduction。

备选方案是只在 README 写文字说明。结构化 metadata 更适合后续比较和自动化报告。

## Risks / Trade-offs

- [Risk] 论文未公开 exact split、seed、代码和权重，本地结果无法证明完全复现。→ Mitigation：报告显式记录 split/seed/sample count，并用 `paper_exact_split_available=false` 标记可比性限制。
- [Risk] 128×128 camera-to-BEV attention 显存和耗时高，普通开发环境难以运行。→ Mitigation：提供 smoke/low-memory 配置、AMP/TF32 可选项和 BEV size metadata；full 配置仍保留论文目标参数。
- [Risk] GPS BEV mask 需要未标准化局部坐标，现有 `gps_batch` 可能已标准化。→ Mitigation：新增可选 `gps_bev_xy` 字段或在 dataset 构建时保留 raw relative XY；缺失时主配置清晰失败，global-only ablation 可继续运行。
- [Risk] radar RA/DA 到 BEV 的几何映射缺少论文细节。→ Mitigation：第一阶段采用可审计 CNN projection，并在报告中记录 radar BEV mapping profile；后续若论文代码开放可替换。
- [Risk] horizontal flip + beam index reversal 若实现错误会污染标签。→ Mitigation：默认只启用 photometric augmentation；beam-aware flip 必须有单元测试覆盖 index reversal 后才能进入 paper full 配置。
- [Risk] 新模型导入 torchvision/大依赖可能破坏轻量导入。→ Mitigation：只在默认组件导入或模型构建时导入重依赖；保留 `kd_sensing.config` 和路径工具轻量导入边界。

## Migration Plan

1. 增加 spec、模型模块、registry 导出和默认组件导入；先用 synthetic tensors 完成 forward/loss smoke。
2. 增加 2604 配置族和 config load tests，确认 full/smoke/ablation 均能解析为 `experiment.task: fusion` 与 `model.primary.type: bev_fusion_2604`。
3. 增加可选 `gps_bev_xy` 数据字段或等价 coordinate provider，并补 dataset/batch tests。
4. 运行小 BEV smoke 训练或单 batch forward；真实数据到位后运行 S32/S33/S34 local split 训练和评估。
5. 更新实验矩阵文档和报告模板，写明推荐命令、cache 准备、输出目录和 caveat。

回滚策略：新增内容集中在 `bev_fusion_2604` 模型、2604 配置族、测试和文档；删除这些新增入口即可回到现有 Image+GPS/JEPA 对照路线，不需要迁移历史 checkpoint 或数据。

## Open Questions

- 是否能从作者后续 release 获得 exact split、seed、代码或权重；若获得，应新增 `paper_exact` 配置并在报告中和 `local_comparable` 分开。
- DeepSense6G 本地 scenes 32-34 的 radar/LiDAR 字段是否全部完整；若缺失，需要确定 full 模型阻塞还是自动转为 ablation。
- GPS spatial mask 的 ROI/grid bounds 应优先使用论文默认、scene-specific bounds，还是从训练 split 自动估计；自动估计需确保 validation/test 不参与 fit。
- 是否需要复现论文中的 H100 latency/FLOPs 表；若本地无 H100，应只记录本机 latency 并标注硬件。
