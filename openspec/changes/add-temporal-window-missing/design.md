## Context

当前 DeepSense6G dataset 已通过 sequence CSV 返回 `[B, T, ...]` 模态张量，`seq_len` 控制历史长度，`num_pred` 控制未来 beam horizon。训练路径在 `BatchStepRunner` 中先 `prepare_task_batch`，再应用 difficulty pipeline；评估主路径也通过 `prepare_evaluation_batch` 应用 difficulty。已有 missing modality 和 pattern-balanced dropout 已经以 difficulty operator 形式置零输入并生成 `*_valid_mask`，因此 temporal missing 应复用该管线，而不是重写 Dataset/DataLoader。

## Goals / Non-Goals

**Goals:**

- 将 `history_window=5` 与 `prediction_window=1` 显式接入配置、CLI 和 summary metadata，同时兼容既有 `seq_len` / `num_pred`。
- 在 batch 层生成 `temporal_mask [B,T]`、`modality_temporal_mask [B,T,M]` 和 `available_modalities [B,M]`，并把不可用输入置零。
- temporal missing 与已有 modality missing 组合后，最终 mask 表达两者交集，且 target 字段不被修改。
- 提供本地可运行检查、launcher、summary 和 focused tests。

**Non-Goals:**

- 不重写数据框架，不新增旧入口或仓库根训练脚本。
- 不新增模型架构或强制改动 checkpoint 格式。
- 不自动运行大规模训练，不提交 `outputs/`、`logs/`、cache 或 checkpoint。

## Decisions

1. 使用 `history_window` / `prediction_window` 作为 alias，同步到现有 `seq_len` / `num_pred`。
   - 理由：当前数据 CSV 已按历史窗口和未来 horizon 组织，直接复用现有字段最小化改动。
   - 备选：新增独立 Dataset wrapper 重新滑窗；放弃，因为会绕开现有 split、target provider 和 cache/normalizer 边界。

2. temporal missing 实现为 difficulty operator。
   - 理由：训练和评估已有 stage/split 可控扰动管线，且有 target-preserved guard。
   - 备选：在 Dataset `__getitem__` 注入；放弃，因为难以复用 train/eval apply 语义，也会污染 sample cache。

3. 默认启用 `modality_frame_bernoulli`/`0.2`，CLI flag 转成 difficulty profile。
   - 理由：当前实验面向 temporal missing 鲁棒性，不再把 clean 行为作为默认路径。
   - 备选：新增全局 DataLoader transform；放弃，因为当前没有统一 collate transform owner。

4. 模型兼容优先依赖现有 temporal core，`temporal_aggregation` 先记录并在 `masked_mean` helper/test 中验证，不强行改变已有模型输入。
   - 理由：主线模型已经消费 `[B,T,...]`，过早把时间维压平或聚合会改变当前训练语义。

5. H5/P1 矩阵在现有 change 内扩展，而不是另建 parallel workflow。
   - 理由：已有窗口 alias、temporal difficulty operator 和 U-Mask eval matrix 都属于同一能力边界；新增脚本只作为 local/manual research workflow。

6. 训练使用在线 stratified sampler，评估使用固定 JSON cache。
   - 理由：训练需要均衡覆盖缺失层级，不能按组合数枚举；评估需要跨 5 方法完全一致的 mask set，cache 记录 seed/checksum 防止误用。

7. 5 个方法映射到现有实现优先。
   - `ours_c2_main` 复用 U-MaskBeamJEPA `supervised_router` + prototype/soft hard-subset 配置。
   - `ours_b4_nonrouter_soft_jepa` 复用 U-MaskBeamJEPA `pcpg` + JEPA + soft hard-subset 配置。
   - `ours_e5_low_lr_pcpg` 复用 U-MaskBeamJEPA `pcpg` + image/lidar low LR 配置。
   - `amber_full` 复用 `configs/fusion/amber_full_architecture.yaml` 的 modular AMBER full local reproduction。
   - `rmbp_mm` 若当前仓库没有可训练 RMBP-MM 配置，则 launcher 保留显式 fallback metadata 并要求用户传入 `--method_config_overrides` 或后续补齐 config；不静默伪装为其它方法。

8. S1-S4 temporal router 使用 `u_mask_beam_jepa` 的显式 opt-in 分支，而不新增第二个整模型注册名。
   - 理由：S1-S4 都复用 C2 的 encoder、prototype head、router reliability feature 和 supervised router 诊断，只改变模态/时间 cell 的 gate 拓扑。
   - `temporal_router_type=none` 保持既有行为；launcher 生成配置时才设置 `s1_temporalagg_modality`、`s2_pertime_modality`、`s3_two_level` 或 `s4_global`。
   - S1 先按 `modality_temporal_mask[:, :, m]` masked-mean 聚合每个模态，再走现有 supervised modality router。
   - S2 在每个历史时刻走一次 modality router，再对可用时刻 masked-mean 聚合 logits/features。
   - S3 复用 S2 的 per-time modality router，并增加轻量 temporal router，对可用时刻 logits/features 做第二级 gate。
   - S4 把 `[T,M]` cell 展平成 token，用轻量全局 router 在所有可用 cell 上 masked softmax。
   - Temporal oracle 默认 hard target：对可用 modality/time/cell 选 circular beam error 最小者；soft target 暂不实现，必须在 diagnostics 中记录 fallback。

9. AMBER Full 和 RMBP-MM 的 `[B,5,4]` mask 接入先走配置与 batch metadata。
   - AMBER Full / RMBP-MM 通过 `missing_modality_metadata.enabled=true` 接收 `modality_temporal_mask`；AMBER 若其 native core 未消费 per-cell mask，评估脚本仍先在 batch 层 zero-fill 缺失输入并记录适配说明。
   - RMBP-MM 的 channel-attention core 消费 `modality_available [B,K,T]`，由 modular forward 的 missing-modality metadata 路径传递；不得用缺失位置真实输入做 imputation target。

## Risks / Trade-offs

- [Risk] 部分历史 CSV 已经是预处理后的窗口行，不能在源码中重新统计原始滑窗跳过原因。→ Mitigation：metadata 明确记录当前 loader 消费的是 prepared sequence CSV，并在检查脚本输出可见窗口/target index 示例；真实 preprocessing 的滑窗统计作为后续扩展保留。
- [Risk] U-Mask eval matrix 既有 force modality mask 与 temporal missing 同时存在。→ Mitigation：temporal missing 通过 batch valid masks/zero-fill 生效，missing pattern 仍通过 force mask 控制模态级评估。
- [Risk] `flatten` aggregation 可能牵涉模型输入维度。→ Mitigation：本轮保留参数和 TODO，不作为默认路径。
