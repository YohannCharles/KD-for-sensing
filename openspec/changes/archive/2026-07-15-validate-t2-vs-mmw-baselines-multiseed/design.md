## Context

现有 `run-mmw-all-weather-missing-modality-matrix` 已完成四方法 seed1 的 15-domain 固定 `last.pth` 训练与共享 mask 评估。T2 在主任务 Top1 上明显优于 AMBER-Full 与 RMBP-MM，但当前 launcher、evaluator 和融合特征提取器都硬编码 seed1；聚合 `metrics.csv` 也不保留逐样本 logits 或稳定 sample id。已有融合特征 NPZ 保存了 label 和 argmax，初步只读复算显示 T2 在 Drop80 的共同 clean-correct 样本 exact 保持率高于 AMBER，而 AMBER 的错误圆周距离更小，说明证据必须区分“保持正确”和“错误邻近性”。

本 change 只形成 local/manual MMW 证据，不把本地适配 baseline 冒充论文等价复现，也不自动写入 reviewed claim registry。

## Goals / Non-Goals

**Goals:**

- 用与 seed1 完全相同的训练和数据协议并行完成三方法 seed2/3。
- 对 seeds1-3 使用同一固定 temporal mask cache，输出可严格配对的聚合指标和逐样本任务输出。
- 生成直接对应 beam 决策的绝对性能、保持率、logit margin、JS、圆周误差及 15-domain 差值图。
- 明确三 seed 波动、domain 异质性、baseline 本地适配范围和结论停止条件。

**Non-Goals:**

- 不增加 S1 多 seed、BPA-off、天气专用模块、传感器噪声训练或新模型架构。
- 不恢复 AMBER 历史 beam 输入、RMBP partial beam/单模态预训练/label-guided imputation，亦不声称论文等价复现。
- 不把 PCA、t-SNE、Isomap 或 prototype 圆环作为 T2 优于 baseline 的主证据。
- 不修改 public package CLI、checkpoint schema、原始 dataset 或既有 seed1 产物。

## Decisions

### 1. launcher 使用显式 methods × seeds × GPUs

`build_config` 接收 seed，并只同步训练随机性字段：`experiment.seed`、domain-balanced sampler seed、temporal-missing seed、输出 run name，以及 T2 已存在的实验 provenance seed。数据 split/portion seed 保持不变，确保三个 seed 使用相同样本。CLI 增加逗号分隔的 `--seeds` 与 `--gpus`，按方法优先、seed 次序构成作业；作业数必须与 GPU 数相等且 GPU 不重复。

启动前对 config、log、run directory 和 manifest 冲突 fail closed，禁止 trainer 自动改成 timestamp 目录掩盖重复运行。每个作业继续使用一卡一进程、`overwrite=false`、`resume=false`。恢复失败作业不在本 change 自动实现；确认 checkpoint 可恢复后再用现有 trainer `--auto-resume` 人工处理。

### 2. 聚合 evaluator 显式按 seed 分层

evaluator 增加 `--seeds`，从 `<method>_seed<N>.yaml` 与 `<method>/seed<N>/checkpoints/last.pth` 读取，输出到 `<eval-dir>/<method>/seed<N>/metrics.csv`。dataset loader seed、行内 seed 和 provenance 均使用对应 seed。它继续复用 v2 固定 mask cache和现有 15-domain macro协议，不将不同 seed 写进同一未分层文件。

0-80% 主曲线保留 `frame_level`、`block`、`modality_frame` type-equal 口径；85/90/95% 只使用精确 modality-frame cell mask，并作为独立极端稀疏曲线。替代方案是只评估代表性 mask，但会丢失既有公平协议，因此拒绝。

### 3. 复用融合特征提取路径保存逐样本任务输出

现有 `analyze_mmw_fused_feature_geometry.py` 已在同一 forward 中获得 fused feature、logits、label 和 mask identity。最小扩展为：增加 `--seed`；保存 float32 logits 和 batch 中稳定 sample id；输出目录由调用者按 `seed<N>` 隔离。提取器必须验证所有 mask 的 label/sample 顺序一致，bundle loader 必须验证跨方法的 domain、CSV checksum、sample id、label、rate、mask digest 和 cache checksum 完全一致。

不把逐样本逻辑塞进聚合 evaluator，避免两条现有诊断路径都实现 model forward。seed1 需要重新提取一次新 NPZ，但不需要重训。

### 4. 主比较使用 pairwise common-clean，三方法图使用 three-way common-clean

每个 seed、domain 先冻结 clean 预测。T2 与某 baseline 的主比较只使用两者 clean 都正确的共同样本；单张三方法图使用三者都正确的交集，并必须报告共同集合的样本数和覆盖率。集合一旦由 clean 定义，在所有 missing rate/mask 上保持不变，禁止逐 rate 重选。

统计先计算 `domain × mask`，再对同类型 mask 等权、最后对 15 domain 等权；sample micro 只作补充。共同集合为空的 domain 必须显式 unavailable，不能静默丢弃。Top1 保持率与圆周 MAE/Within1/Within3同时报告，因为 exact correctness 和邻近错误可能给出不同排序。

### 5. 决策空间指标避免跨模型 logit 尺度误读

真类 margin 定义为 `z_y - max(z_other)`；只比较同一模型、同一样本的 `missing - clean` 变化，不把三种 head 的绝对 logit 值直接横比。JS 使用 clean/missing softmax 的对称 Jensen-Shannon divergence，并除以 `ln(2)` 归一到 `[0,1]`。两项均作为保持率和圆周误差的解释性指标，不单独形成优越性 claim。

### 6. 多 seed 与 domain 统计分开表达

正式 local evidence 输出每 seed 的 15-domain macro、三 seed mean/std、T2-minus-baseline逐 seed差值，以及对 seed-domain 配对单元的分组 bootstrap 95% 区间。bootstrap 以 seed和domain为组，不把重复 mask或单帧当独立样本。晋级表述要求至少 2/3 seed 的主曲线 AUC delta 为正，三 seed平均 clean、0-80 AUC和Drop80均不低于 baseline；否则按条件缩小结论，不修改模型制造优势。

## Risks / Trade-offs

- [六个训练并发造成 CPU/I/O 争用，实际耗时超过 seed1 的 5.4 小时] → 保持每作业 4 workers/4 OMP threads和一卡一进程，启动后监控 epoch速度，不在同卡叠加评估。
- [逐样本 logits 增加约数百 MB ignored 产物] → 使用压缩 NPZ、只保存 modality-frame clean/20/40/60/80 任务输出，0-80 全 mask-type 聚合仍由 evaluator负责。
- [common-clean 子集偏向容易样本] → 同时报告全样本绝对 Top1/圆周误差和共同集合覆盖率，不用它替代主性能表。
- [AMBER 或 RMBP 某 seed 失败] → manifest保留失败状态，summary输出 unavailable，不用剩余 seed 冒充三 seed均值。
- [baseline 本地适配与论文差异] → 每张表和 summary 保留 reproduction scope；主结论限定为“统一四传感器协议下的本地适配比较”。

## Migration Plan

1. 扩展 launcher/evaluator/提取器并完成 focused tests、dry-run和OpenSpec strict validation。
2. 在 GPU0-5 并行启动 T2、AMBER-Full、RMBP-MM seed2/3；保留 seed1不动。
3. 训练完成后运行三 seed固定 mask聚合评估，并为三个 seed重新提取逐样本任务输出。
4. 生成三 seed CSV、图和中文说明；按晋级门禁给出支持、部分支持或不支持。
5. 回滚只需停止使用新参数；现有 seed1默认行为和产物路径保持兼容。

## Open Questions

- 当前 prepared split 只有冻结的 local validation，正式投稿是否需要另建独立 group-safe test，需在本轮结果稳定后单独决定。
