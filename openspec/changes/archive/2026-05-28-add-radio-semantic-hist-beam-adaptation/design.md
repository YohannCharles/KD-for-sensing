## Context

当前仓库的 HiST-Beam 已经从最初 DeepSense6G 快速验证推进到 MMW geometry-aware 方向：`HistBeamFusionNet` 已有 CLS-token fusion、shared/private branch、coarse/fine head、zero-init private adapter、geometry token、coarse-conditioned adapter、angular smoothing、geometry consistency、private/adapter prototype artifact、MMW data availability、scenario LOSO planner 和 single-scene claim guard。

这意味着本变更不应重写数据准备、LOSO 编排或 fusion 主干。新的方案要做的是把 prototype 的语义单位从 coarse sector/private geometry 聚类升级为由 beam-power profile 派生的 radio-semantic pattern，并把它作为 shared branch 的监督、source shared prototype 的聚类单元，以及 target 推理时可选的融合条件。

关键约束：

- sensing 输入仍是 image/radar/GPS/LiDAR/IMU 等可配置模态；CSI/channel/beam_power 只作为 label、derived target、metric 或 few-shot target 标注来源。
- `label_budget=0` 的 target adaptation 不能读取 target beam、beam_power、q_power 或 radio_semantic_label 作为训练监督。
- 当前工程中 `v6_full_finetune` 已被用作 full fine-tuning baseline；新论文口径中的 V6 radio method 必须用显式配置名区分，避免静默改变旧配置含义。
- OpenSpec 与说明文字使用简体中文；项目 Python 命令使用 `conda run -n kd_mm_beam <command>`。

## Goals / Non-Goals

**Goals:**

- 新增可审计的 `RadioSemanticLabelBuilder`，支持 `coarse`、`peak_spread` 和预留 `kmeans_power`，快速主路径采用 peak group + entropy spread bin。
- 在 HiST-Beam 中增加 `radio_head(c)` 和可选 `radio_embedding`，让 beam head 支持 `[c, s*]` 与 `[c, s*, e_alpha]` 两种推理输入。
- Source training 后保存 shared radio prototypes `mu_radio_c/count_radio`，并保留 coarse prototypes 作为 V5 baseline。
- Target adaptation 支持 `proto_type: radio_semantic`，通过 source shared prototype assignment 生成 `alpha/r_hat/conf`，并维护 target-private prototype bank `nu_radio_s` 做 target 内部 private clustering。
- 增加 radio semantic loss、radio accuracy、power metrics、prototype diagnostics、leakage metadata 和 V5/V6/V7 消融配置。
- 保持现有 MMW geometry-aware smoke/scenario-LOSO、DeepSense6G HiST-Beam 回归和旧 adapter/prototype baseline 可运行。

**Non-Goals:**

- 不把 CSI/channel/beam_power 作为默认 sensing 输入模态；它们不得绕过配置成为模型输入。
- 不实现完整论文系统或自动下载/移动/提交 MMW 数据、cache、checkpoint、日志。
- 不承诺 `kmeans_power` 在第一阶段可用于主实验；它作为后续增强，快速验证优先 `peak_spread`。
- 不用 prototype 直接映射到 beam；最终 beam 必须由 beam head 输出。
- 不强制 source private prototype 与 target private representation 对齐；source private artifact 可保留为兼容诊断，但默认不用于 radio method。

## Decisions

### 1. Radio-semantic label 在 runtime/dataset 层派生，prepared manifest 只记录可追溯 metadata

**Decision:** 将 radio-semantic label builder 放在 `src/kd_sensing/data/mmw/` 或 `src/kd_sensing/data/` 的可复用模块中。MMW prepared manifest 记录 beam_power 路径、beam label、label builder 配置版本、是否可派生和 unavailable reason；训练 dataset 按配置返回 `radio_semantic_label`。

这样做可以复用当前 MMW beam power 产物，不必重新打包数据；也能在阈值、spread bin 或 KMeans 版本变化时通过配置重算，避免旧 prepared 产物与新 label 定义绑定过死。

**Alternatives considered:** 在 MMW preparation 阶段把 radio label 固化到 CSV。拒绝原因是阈值和模式仍处于快速验证阶段，固化后会增加重建数据的成本；但 manifest 可以缓存派生结果摘要以便审计。

### 2. 快速主方法使用 peak group + entropy spread，而不是先上 KMeans

**Decision:** `peak_spread` 是默认主路径：从 beam power 归一化分布计算 best beam、coarse peak group 和归一化 entropy，用阈值 `[0.35, 0.65]` 生成 narrow / medium / wide spread bin，最终 `radio_label = peak_group * num_spread_bins + spread_bin`。

该标签保留了 beam 主方向，又显式区分传播能量集中程度，能直接覆盖文档中的窄峰、宽峰、多径/分散形态。`kmeans_power` 保留接口和 artifact metadata，但不作为第一阶段必须完成项。

**Alternatives considered:** 继续只用 `beam // group_size`。拒绝原因是它无法区分同一 coarse sector 内的传播形态，也无法解释 radio prototype 相对 coarse prototype 的增益来源。

### 3. 新 variant 不覆盖现有 `v6_full_finetune`

**Decision:** 工程配置新增 `v6_radio_proto` 或 `adapter_radio_proto` 作为 radio-semantic full method；论文/报告可将 full fine-tuning baseline 称为 V7，但现有 `v6_full_finetune` 配置含义不静默改变。

这能保护当前 quick validation、测试和历史输出。后续如果需要严格 V0-V7 命名，可以新增 `v7_full_finetune` 配置作为等价 baseline，但迁移必须显式。

**Alternatives considered:** 直接把 `v6_full_finetune` 改成 radio method。拒绝原因是会破坏现有 OpenSpec、配置和 summary 对 full fine-tuning baseline 的语义。

### 4. Radio prototype 只提供 assignment / condition，不直接预测 beam

**Decision:** Source prototype artifact 保存 `mu_radio_c`，target 阶段用 cosine assignment 得到 `alpha`。若启用 radio-conditioned inference，则 `e_alpha = alpha @ radio_embedding.weight`，最终仍由 `beam_head([c, s*, e_alpha])` 输出 64-beam logits；若关闭，则 beam head 只读 `[c, s*]`。

这避免把 `radio_semantic_label` 误当成 beam hierarchy parent。radio label 包含 spread/multipath 信息，不满足 `p(b)=p(r)p(delta|r)` 的严格层次关系。

**Alternatives considered:** nearest radio prototype 后直接映射 beam。拒绝原因是会丢失 private adapter 的 target-specific fine correction，也容易把 prototype 变成粗分类器。

### 5. Target-private prototype bank 是 target 内部稳定器，不对齐 source private prototype

**Decision:** `proto_type=radio_semantic` 时，source shared prototype 只用于给 target 样本分配 radio semantic condition；target private side 维护 `nu_radio_s/count`，用高置信 target `s_adapt` 通过 EMA 更新，并在 warmup 后计算 private clustering loss。

该设计承认 private branch 是 scene/town/weather-specific refinement，不强迫 target private 贴近 source private。source private prototypes 可以继续保留给 V5 coarse/private baseline 与回归诊断。

**Alternatives considered:** 使用 source `private_prototypes` 直接监督 target `s_adapt`。拒绝原因是与 shared/private 分工相冲突，并且上一轮设计已把 source private 强对齐识别为可能导致 v4/v5 不可区分的风险。

### 6. Leakage guard 进入 adaptation batch/loss 边界

**Decision:** 在 target adaptation 的 unlabeled 路径显式记录并断言是否访问 target beam、beam_power、radio label。`label_budget=0` 时 supervised target loss、radio CE、power-profile KL 和基于真实 target radio label 的采样/threshold selection 必须关闭；target_test 的 beam_power/radio label 仅可在 evaluation 使用。

**Alternatives considered:** 依赖调用方不传 label。拒绝原因是当前 dataset 可以返回多种 label/metadata，必须在 adaptation 入口和日志层提供机器可读证据，才能支撑论文防泄漏声明。

### 7. 指标以“可用则计算、不可用则说明”为原则

**Decision:** Radio metrics 包括 radio accuracy、radio prototype coverage/confidence/used count、target-private prototype initialized count、NRP、beam power loss dB 和 leakage flags。缺少 beam_power 或 radio label 时，metrics/summary 必须写 unavailable reason，而不是填 0。

**Alternatives considered:** 只报告 Top-K。拒绝原因是无法证明 radio semantics 是否真的参与训练、适配和推理。

## Risks / Trade-offs

- **[Risk] Radio label 阈值不适合当前 MMW beam power 分布。** → 先在 source train 记录 entropy histogram、radio class counts、empty classes；若类别严重失衡，调整 thresholds 或启用 top-k mass / KMeans 后续模式。
- **[Risk] `beam_head([c, s*, e_alpha])` 改变输入维度，影响旧 checkpoint 加载。** → radio conditioning 默认 opt-in；未启用时维持旧 head 形状；启用时要求新 run 或明确迁移 checkpoint。
- **[Risk] target-private prototype bank 早期噪声大。** → 使用 confidence threshold、warmup、EMA momentum、min count 和 diagnostics；coverage 为 0 时标记 no-op/inconclusive。
- **[Risk] 0-label adaptation 误用 target derived labels。** → 加 leakage guard、adapt log flags 和测试；unlabeled loss 只允许 entropy / consistency / prototype assignment，不允许真实 label CE。
- **[Risk] 当前只有单场景 MMW 时无法证明跨场景。** → 保持 existing claim guard；single-scene 只跑 smoke / within-scenario sanity，至少两个 ready scenario 后再做 scenario-LOSO。
- **[Risk] 新 V6 与旧 `v6_full_finetune` 命名混淆。** → 工程名使用 `v6_radio_proto`，summary 显式记录 `method_family` 和 `baseline_role`。

## Migration Plan

1. 在当前 geometry-aware HiST-Beam 基础上新增 radio semantic label builder 和 dataset 返回字段；先用 synthetic / fixture beam power 验证 label 构造和 fallback。
2. 扩展模型配置与 forward 输出：新增 radio head、radio embedding、radio-conditioned beam head；默认关闭，旧 V0/V1/V3/V4/V5/full fine-tune 不变。
3. 扩展 source loss 与 prototype artifact：启用 radio CE 时保存 `mu_radio_c/count_radio`，并在 metadata 中记录 label mode、thresholds、class counts。
4. 扩展 target adaptation：加入 radio assignment、target-private prototype bank、leakage guard 和 diagnostics；先在 smoke 配置下验证 0-label 不读取 target label。
5. 扩展 LOSO configs 和 summary：增加 V5 coarse vs V6 radio、V6-a/off vs V6-b/on、full fine-tune baseline 对比。
6. 在至少两个 MMW ready scenario 后运行最小 scenario-LOSO；单场景阶段只报告 loader/forward/loss/adaptation smoke。

回滚策略：所有 radio semantic 路径保持 opt-in；关闭 `radio_semantic.enabled` 和 `use_radio_condition_in_beam_head` 后，应恢复当前 geometry-aware HiST-Beam 行为。

## Open Questions

- MMW 当前及后续 ready 场景的 beam_power entropy 分布是否支持固定 `[0.35, 0.65]` 阈值，还是需要 per-source quantile bin？
- `radio_semantic_label` 是否应按当前预测 horizon 只使用 `future_beam1`，还是为多 horizon 输出 `[B, H]` labels？
- V6 radio method 是否允许训练整个 beam head，还是只训练最后一层，以满足 trainable ratio < 2% 的目标？
- `kmeans_power` 的 source-only cluster artifact 应由 preprocess 生成还是 source training 后生成？
