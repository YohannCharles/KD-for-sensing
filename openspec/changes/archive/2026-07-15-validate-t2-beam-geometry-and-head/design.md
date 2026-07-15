## Context

当前 S1/T2 使用相同的 prototype head、Gaussian soft target 和 `beam_label_circular=true`，T2 只额外启用同模型 temporal superset teacher 的 confidence-gated KL。seed1 固定四个 mask/缺失率的结果显示 T2 在 0/20/40/60/80% Top1 上均优于 S1，但 Top3 和 MAE 没有同步改善，证据尚未覆盖随机种子波动。

实现依赖 active change `improve-s1-lightweight-temporal-robustness` 已完成的 temporal pooling、superset consistency、launcher/eval 和 seed1 筛选。本 change 是该 change 条件式多 seed 阶段的后继：旧 tasks 7.2-7.3 由本 change tasks 4-5 取代，最终按依赖顺序先收口旧 change 的执行记录，再收口本 change。

进度汇报第 9 页把 circular index distance 直接解释成 beam 物理几何。实现实际在 soft target、两条 supervised-router oracle loss、router eval diagnostics、DBA 和 T1 risk 中将 0 与 63 视为相邻；prototype 向量本身没有被约束成圆。对本地 Scene31-34 的 18,479 个有限 beam-power sweep 审计后，标签与 0-based argmax 完全一致，普通相邻列相关显著高于 0/63，且时序标签没有 0/63 端点跳转，因而该数据集更适合线性有序解释。

已有 final C2 `c2_no_circular_soft_targets` 同时关闭 Gaussian 并切到 one-hot，不能隔离 linear 与 circular。已有 classifier/prototype-loss-off 三 seed 结果又优于 prototype 主配置，因此 head 假设也必须与 T2 机制分开验证。这里的 `CLS` 是行为口径：共享模型仍实例化 unused `prototype_bank` 以保持 checkpoint 结构，但它不参与 logits、loss 或 router。

## Goals / Non-Goals

**Goals:**

- 补齐当前 circular-prototype S1/T2 的 seeds2/3，检验 T2 相对收益是否稳定。
- 以相同 S1/T2 训练协议分别筛选 linear-Gaussian prototype 和 classifier/prototype-loss-off。
- 明确区分训练 target geometry、模型 head 与评估 distance mode，并保留可复算 provenance。
- 用两轮 GPU0-7 门禁矩阵控制算力，只给通过 seed1 的候选补多 seed。
- 输出可直接落实进度汇报红字的中文报告和保守结论。

**Non-Goals:**

- 不把 DeepSense 结论外推到 MMW Town3；MMW 天气实验另开 change。
- 不修改 canonical U-Mask 默认配置、模型注册名、公共 package CLI 或 checkpoint schema。
- 不新增 learnable beam angle、真实 codebook 文件、外部 teacher、T1 ranking 或新 loss。
- 不把本地多 seed 结果自动写入正式 claim registry。

## Decisions

### 1. 只增加四个 launcher 方法标签

在现有 `s1_overrides` 上组合已有配置字段，增加 `S1-LG`、`T2-LG`、`S1-CLS`、`T2-CLS`。这些方法仅在用户显式传入 `--methods` 时出现，`default` 与 `s1_lightweight` profile 默认列表保持不变。

`LG` 保持 prototype 和 Gaussian target，只设置 `beam_label_circular=false`、`circular_beam_distance=false`、`use_gaussian_beam_targets=true`、`use_circular_soft_targets=false`，并使用 linear evaluation profile。该距离口径同时作用于 prototype target 和 supervised-router oracle target，不能残留把 beam 0/63 当邻居的混合几何；它也不能退化为 one-hot。

`CLS` 设置 classifier head，关闭 fused/modality prototype loss 与 prototype-margin router feature，并将仍然存在的 supervised-router oracle target 显式设为 linear distance；S1/T2 的 pooling、router 和 T2 KL 其它字段不变。该候选同时改变 head、prototype losses 和一个 router feature，因此只检验当前 prototype/head package 是否受支持，不能解释为纯 head 因果消融。

备选方案是增加实体 YAML 或第二套 launcher；这会复制当前生成逻辑并扩大实验表面积，因此拒绝。

### 2. 训练几何、head 和评估距离分别记录

Top1/Top3 与 beam 距离定义无关，可用于 circular 与 linear checkpoint 的直接比较。ADBA、Within@3 和 MAE 必须随 `evaluation.dba_distance_mode` 解释；eval 输出至少记录 `metric_profile`、`dba_distance_mode`、head type、prototype enabled、prototype target geometry、router oracle geometry 和 training geometry。

每个 cache entry 还记录 0-based `mask_index`、生成 `mask_type`、cache checksum/seed，以及只由原始模态顺序和实际 `[5,4]` mask 计算的稳定 `mask_digest`。digest 不包含 type/index，因此不同生成路径得到相同实际输入时可识别为重复。

summary 先按 `(seed, rate, drop_count, mask_index, mask_type, digest, cache checksum)` 严格配对；任何缺失或冲突都使该 pair `unavailable`。之后按 `(seed, rate, drop_count, digest)` 折叠重复 entry，并先在每个 rate 内对 unique digest 等权、再对五档 rate 等权，形成独立的 paired-mask 证据。候选与 final 主门禁继续使用冻结的每 cell 4-entry protocol matrix 均值，避免在实验中途改变选择指标；paired 去重结果不得冒充该主门禁。输出 `seed_summary.csv`、`seed_deltas.csv`、`paired_mask_deltas.csv` 和 `gate_decisions.csv`；不同 distance mode 的距离指标不得合并。

评估脚本增加可选 `--dba-distance-mode config|circular|linear`，默认 `config` 保持兼容。显式覆盖只改变评估指标，不改变 checkpoint 或 logits，可用于同一 checkpoint 免费复算 linear/circular 指标。

### 3. 两个独立 output root 并行组成第一轮

第一轮同时启动两个各四作业的 launcher，避免为非矩形 seed×method 集合新增 job DSL 或 manifest append 逻辑：

- GPU0-3：S1/T2 current circular-prototype seeds2/3；
- GPU4-7：S1-LG、T2-LG、S1-CLS、T2-CLS seed1。

两组使用不同 ignored root 和 manifest，因此不会覆盖 seed1 原始 provenance。第二轮只在独立 advancement root 为通过候选运行 seeds2/3；最终分析从各 root 的 immutable per-run config、run status 和 eval CSV 汇总。

### 4. 晋级与主线判定

新候选 seed1 只有同时满足以下条件才补 seeds2/3：相对匹配 head/geometry 的 S1 五档 mean Top1 为正、Drop80 为正、Drop0 下降不超过 0.005，且自身五档 mean Top1 相对 current T2 不低于 0.005。linear 候选若 Top1 与 circular 差距不超过 0.002，因物理口径更可信而优先。

最终 T2 机制保留要求：三 seed 的五档 mean Top1、Drop0-60 mean 和 Drop80 平均均优于匹配 S1；至少 2/3 seed 的五档 mean delta 为正；平均 Drop0 下降不超过 0.005。classifier 若三 seed 稳定优于 prototype，则 PPT 将 prototype 降为基础消融，主线表述改为 T2 superset consistency。

### 5. PPT 报告区分事实、推断和待验证项

报告逐页列出红字原文、建议替换文案和证据来源。seed1 结果必须标注 screening，不能写成正式统计结论；prototype 对物理连续性的解释必须写为机制假设；single-modality gate=1 必须说明是 masked softmax 约束结果。MMW weather 与轨迹图保留为后续阶段，不用示意图冒充已完成实验。

## Risks / Trade-offs

- [Risk] 第一轮同时读取 Scene31-34 造成 I/O 竞争。→ 沿用 batch64、每卡一进程、persistent workers 和 12 intra-op threads，不在单卡启动第二进程。
- [Risk] `use_circular_soft_targets=false` 被旧 config normalization 误解为 one-hot。→ 同时保持 Gaussian enabled，并用 focused test 断言 resolved `proto_target_type=gaussian` 和 `beam_label_circular=false`。
- [Risk] 不同 output root 使汇总入口不能一次自动消费所有 seed。→ 保留每个 root 的 manifest，最终用原始 eval CSV 做只读合并；不为一次实验增加通用 manifest 数据库。
- [Risk] classifier 改变 router prototype-margin 输入。→ 显式关闭该 feature，并把变化列入方法定义；不声称是纯 head 参数量消融。
- [Risk] prototype target 已切成 linear，但 router oracle 仍按 circular 选择分支。→ 同一 `circular_beam_distance` 贯穿两条 router loss 路径和 eval diagnostics，并用 beam0/beam63 端点测试区分两种口径。
- [Risk] circular 与 linear ADBA 不可直接混合。→ 方法和输出记录 distance mode；跨口径主选择只用 Top1，ADBA/MAE 分口径报告。
- [Risk] 固定 cache 的不同 entry 可能生成完全相同的实际 mask。→ evaluator 写入 digest 与 entry provenance，summary 在严格配对后按 digest 去重，并保留 source indices/types 和 duplicate count。

## Migration Plan

1. 添加 OpenSpec、报告骨架、四个方法标签和 focused tests。
2. dry-run 检查八任务 GPU0-7 映射、独立 root、Gaussian/classifier 关键配置和默认 profile 不变。
3. 运行第一轮训练、固定 mask evaluation 和 seed1 门禁。
4. 为晋级候选运行第二轮 seeds2/3，并用同 cache 评估。
5. 汇总 current 与候选的三 seed mean/std、paired mask delta 和 diagnostics，回填中文报告。
6. 回滚只需停止使用新增显式方法标签；默认配置和运行时没有迁移。

## Open Questions

- MMW Town3 的 codebook 是否环形由其 channel/codebook metadata 单独决定，本 change 不复用 DeepSense 结论。
- 是否将最终胜出候选升级为正式 config/claim，待三 seed 与 paired per-sample evidence 审查后另行决定。
