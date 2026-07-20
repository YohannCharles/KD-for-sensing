## Context

当前 `configs/mmw/t2.yaml` 继承 `_base.yaml` 的 Adam/`1e-4`/无 scheduler 配方，`s1.yaml` 又继承 T2。历史 H4 的开发候选同时使用 AdamW、`3e-4` weight decay 和单个 40-epoch cosine cycle，但 `launch_mmw_all_weather_matrix.py` 会无条件恢复 H0 的 optimizer/weight decay 并删除 optimizer 字段，因此仅写 YAML 不会得到 H4 的实际运行。

并行的 `validate-t2-mmw-bpa-cma-ablation` 已生成并运行 H0 配置，必须保持其 method、loss、checkpoint 和产物不变。`tune-t2-mmw-hyperparameters` 的 H0--H5 也属于已消费 outer test 的 development evidence，不能被本 change 重写或升级。新的设计筛选仅使用 outer-train 派生的 group-safe inner validation 作选择，并将全部输出标记为 development-only。

## Goals / Non-Goals

**Goals:**

- 将 H4 固化为 canonical T2/S1 mainline launcher 的默认 U-Mask 训练配方，并为 legacy H0 protocol 提供显式、可验证的 selector。
- 让 MMW launcher 不再静默改写 optimizer、weight decay 或 scheduler，并将 profile/结构候选指纹写入运行 provenance。
- 在 H4、四模态、5-to-1 时序、15-domain、40 epoch、`last.pth` 和共享 inner split 下，分阶段筛选有限的 T2 容量、router、GPS、BPA、CMA 和时序/融合候选。
- 对新增的可靠性融合与 temporal attention 实现 mask-aware 行为，保留默认 supervised-router/masked-mean 的可复现输出。

**Non-Goals:**

- 不改正在收口的 H0 BPA/CMA 正式消融配置或其已有产物。
- 不用 outer test、早停或最佳 checkpoint 选择候选；不把单 seed 筛选写入论文主表或 claim registry。
- 不新增 GPS 两帧窗口、物理坐标数据扰动、外部 pretrained encoder、完整 AMBER Class-Former 或无假设支撑的新损失。
- 不执行所有参数的笛卡尔积，也不按 GPU/候选使用不同 batch 或训练预算。

## Decisions

### 1. H4 采用 canonical training profile + router architecture selector

`t2.yaml` 保持作为 tracked architecture/base recipe，避免其继承关系污染 S1、legacy H0 screening 和未完成的 BPA/CMA ablation。`launch_mmw_all_weather_matrix.py` 新增两个独立 selector：新的 T2/S1 mainline launcher 显式选择 `umask_h4_v1` 和 `umask_router_nopattern_v1`；`legacy_h0_v1` 与 `umask_router_pattern_v1` 仅用于既有 H0 screening 与 BPA/CMA ablation。profile 只允许 U-Mask methods；AMBER-Full/RMBP-MM 始终保留自身 recipe。

builder 的顺序为：加载 tracked recipe、选择并物化 training profile 与 router architecture profile、覆盖 seed/batch/epoch/output 等运行维度、应用预注册 ablation 或设计候选。它不得在 profile 之后静默重设 optimizer、weight decay、scheduler 或 router pattern setting。每个生成配置和 `mmw_all_weather_protocol` 记录两个 profile 的 id、canonical values 和 SHA256 指纹；任一 profile 不一致的结果不得汇总。

将 H4 或 RouterNoPattern 直接写入 `t2.yaml` 被拒绝，因为 S1、legacy H0 screening 和 active BPA/CMA config regeneration 都从它派生。直接改 `_base.yaml` 被拒绝，因为会污染 AMBER/RMBP；不提供 H0/pattern-on selector 被拒绝，因为会改写 active ablation 的对照。RouterNoPattern 作为 T2/S1 共用 mainline profile，而非仅作用于 T2，避免最终 T2-vs-S1 比较同时混入 router pattern 差异和 T2 objective 差异。package CLI 若需主线 H4/RouterNoPattern，必须使用 generated config 或新的 explicit profile launcher，而不是猜测 YAML 继承。

### 2. 设计筛选使用固定 inner validation 与 8-GPU 分波

新 launcher 复用既有 group-safe inner split、显存 probe、40-epoch completion 和 manifest 验证逻辑，但使用独立 output root、独立 protocol id 与 `development_only=true`。每波八个候选一 GPU 一进程，batch 必须是 probe 得到的 16 倍数共同安全值；GPU0--3 与外部任务共存时仍遵守同一个 peak-reserved 阈值。

第一波只使用当前配置级参数：H4 control、`d_model=48/96`（同步四 encoder `output_dim`）、`router_hidden_dim=32/128`、关闭 pattern feature、GPS MLP `hidden_size=32/128`。这样先测试容量和 GPS 依赖，避免新结构与多项 loss 同时变化。

第二波单因素搜索 BPA：prototype temperature `0.08/0.15`、sigma `1.5/3.0`、仅提升 fused BPA weight、仅提升 modality BPA weight、二者同时提升，并保留 H4 control。CMA 不与 BPA 共存；它在后续波从 H4-NoBPA matched control 出发，单独比较预注册 weight/temperature。现有 superset KL 是唯一允许的额外 loss 候选，必须保持 same-model teacher 契约。

每个候选只有同时满足 `delta J >= +0.5pp`，且 Clean、Drop1--3 均值、Drop80 相对 H4 均不低于 `-0.5pp` 才进入 seed2/3。未通过者不与其他候选组合。

development-only config 还必须显式关闭 trainer 的 final test：它只构造 inner train/validation loader，既不迭代 outer test，也不发布任何 outer-test metric。final artifact 可以将 `final_test_metrics` 记录为明确的未执行状态，但该记录不得含测试证据。outer test 只属于候选晋级后的独立 evidence protocol；即使当前自动选择代码未读取 final-test 字段，预先暴露该指标也会破坏筛选盲性。

候选的 full-config 与 candidate-recipe 指纹只覆盖生成 YAML 的可比较内容。CLI 注入的 `runtime` 字段（例如 config 路径、run dir）是运行定位信息，必须从两类指纹中排除；否则同一生成 YAML 会因启动方式而被错误拒绝。该豁免不适用于 model、data、training profile、candidate id 或 inner-split 身份。

### 3. 结构扩展保持最小且默认不变

`u_mask_beam_jepa` 保留 `supervised_router` + `masked_mean` 为默认。新增 `reliability_mean` fusion：对可用模态的 learned reliability 归一化加权，router oracle weight 必须为零；新增 `masked_attention` pooling：以共享 learned query 在每模态历史帧内做 mask-aware softmax，任何不可用 temporal cell 均不参与权重归一化。两者输出保持现有 diagnostics keys，并额外声明 type/parameter count。

GPS 第一阶段仅调 `gps_mlp` 的 hidden size/dropout 等已有 module 容量；可选 normalized-feature jitter 必须只在训练期生效、在 provenance 中记录且不伪称米级物理噪声。GPS 两帧、原始坐标米级噪声和 GPS-GRU 属于后继数据/模型 change，不在本轮临时加入。

image/lidar encoder search 只允许已 tracked、scratch、输入契约兼容的 registry encoder；若现有 ResNet scratch smoke 不能满足输入/显存/完成性校验，记为不合格而不新增第三方或外部权重。

### 4. CMA/new loss 的边界

CMA 继续是 pooled-feature objective analogue，且配置层面保持与 BPA 互斥。CMA weight/temperature search 的 matched control 必须关闭 BPA，结果只能回答“替代 BPA objective 的敏感性”。本轮不引入第二个全新 loss；先将 H4 + existing confidence-gated superset KL 作为预注册 loss 候选。任何真新增 loss 必须先有明确失败模式、独立 OpenSpec 和配对消融。

## Risks / Trade-offs

- [H4 与 H0 旧产物混合] → profile id、canonical values、fingerprint 和 summary validation 必须 fail closed。
- [GPU0--3 正在被其他作业使用] → 先做每卡真实 step probe，使用共同 16 倍数 batch，保留 10% reserved-memory headroom；probe 失败不启动该卡正式任务。
- [单 seed 假阳性] → 首轮只做开发淘汰，所有晋级候选必须在相同 inner protocol 上补 seed2/3。
- [新 fusion/pooling 破坏 router 或 mask 语义] → 单元测试覆盖 all-masked rejection、missing cell zero weight、默认输出兼容和 auxiliary loss diagnostics。
- [development candidate 意外消费 outer test] → generated config 设置 `training.final_test.enabled=false`，dataloader 不构造 test split，trainer 不调用 final-test；测试断言 outer test loader 不被迭代。
- [CMA 被误读为叠加收益] → 强制 BPA/CMA 互斥并在 manifest/summary 标注 matched control。
- [H4 旧结果来自 pre-canonical 模型] → 当前 canonical T2 上的 H4 control 是所有新筛选和后续独立 evidence 的唯一基线。

## Migration Plan

1. 实现 profile selector、H4 mainline generated recipe、legacy H0 overrides 和 provenance tests；确认 BPA/CMA/H0 screening 生成配置保持 H0。
2. 实现设计-screen launcher 与第一波 config-only variants，完成 config dry-run、single-step smoke 和 GPU0--7 common-batch probe。
3. 启动第一波 seed1 训练；只使用 inner validation 决定晋级。
4. 实现并测试 minimal fusion/pooling branches，再启动结构波与 BPA 单因素波；CMA/loss 波在 active BPA/CMA change 收口后运行。
5. 对晋级项补 seed2/3，并在未消费的独立 outer evidence 上确认；RouterNoPattern 在此之前仅是开发主线。失败时回退到 `umask_router_pattern_v1`，并删除仅本 change 新增的筛选入口即可回滚。

## Open Questions

- 现有 scratch ResNet-18 同时替换 image/lidar encoder 后的实际 batch 上限由 per-GPU probe 决定；若无法形成共同 batch，该 encoder 行不进入正式矩阵。
- H4 通过设计筛选后是否立即替换论文主表，仍取决于后续独立 outer fold 的多 seed 确认，不由本 change 的 development 指标决定。
