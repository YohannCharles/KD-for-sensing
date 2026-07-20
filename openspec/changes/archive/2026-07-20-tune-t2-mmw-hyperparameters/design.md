## Context

MMW T2 当前使用 15 个 `weather/scenario` domain、四个 sensing modalities、40 epoch 和 epoch-40 `last.pth`。tracked `configs/mmw/t2.yaml` 及其 shared base 是唯一基线输入；筛选从 outer train 侧派生独立 validation，不把 output 配置或 outer test 用作训练期输入。

并行 active change `validate-t2-mmw-bpa-cma-ablation` 已冻结六方法、五个新增方法 x 三 seed、固定训练预算和正式 paired evaluation。该 change 回答 BPA、CMA、circular topology 与 prototype package 的机制问题；本 change 只在完整 T2 架构内做 seed-1 development screening，不能改写前者的配置、任务、运行目录或正式结论。

显存探测会解析一个全卡共同 batch size。batch size 会影响每 epoch optimizer step 数，因此它是本轮预注册的 development hardware protocol，而不是可在运行中按 variant 或 GPU 自适应的调参项。

## Goals / Non-Goals

**Goals:**

- 从 tracked T2 recipe 和现有 MMW T2 builder 建立可审计的基线 fingerprint，生成六个具名 seed-1 development variants。
- 保持四模态、T2 architecture、15 domains、outer train/test、40 epoch、评估 mask 和 epoch-40 `last.pth` 不变。
- 从 outer train 侧生成所有 variants 共用、与 inner-train 资源身份不相交的独立 validation，用于每 5 epoch 的观测。
- 以真实 AMP training step 解析请求 GPU 都可用的最大预注册 16 倍数 batch，并保证 probe 不改变其余 protocol。
- 用固定选择分数和保护门槛只产生 development candidate 或 `no_change` 结论。

**Non-Goals:**

- 不新增或替换 T2 module，不改 head type、prototype 数、circular beam geometry、history/prediction window、domain inventory、outer test 或正式 mask protocol。
- 不启用 early stopping、best-checkpoint selection、按 GPU 设置不同 batch、自动学习率缩放、梯度累积或动态缩短 epoch。
- 不把筛选结果写入 reviewed claim、论文主表或 BPA/CMA formal ablation 产物。
- 不提交 generated YAML、split CSV、manifest、probe report、日志、checkpoint、metrics 或图表。

## Decisions

### 1. Tracked T2 recipe 提供基线事实，现有 builder 继续拥有配置生成

launcher 必须读取 tracked `configs/mmw/t2.yaml` 及其 shared base，记录 recipe SHA256，并抽取 T2 architecture、loss、optimizer、domain、split、missing curriculum 与 evaluation fingerprint。随后复用 `scripts/launch_mmw_all_weather_matrix.py` 的 T2 config builder 生成 H0；recipe 缺失、字段不一致或不是 15 domains 时 fail closed。

这样既保留可审计基线来源，又避免复制 MMW builder。直接读取 output resolved config 的备选方案被拒绝，因为其中包含运行期 metadata 和本地路径，容易把历史状态带入新 run。

所有 generated configs、inner split artifacts 和 manifest 写入单独的 ignored output root，例如 `outputs/mmw_t2_hyperparameter_screening_v1/`。源码只保留 launcher 与测试，不新增六份 tracked YAML。

### 2. outer split 不变，validation 从 outer train 侧生成

每个 15-domain outer train CSV 使用现有 MMW `group_safe_time_block` owner 派生固定 `10%` development validation；为避免同一时刻 RSU radar 被不同 CAV 样本复用，inner routing 将所有 CAV 窗口映射到共同 RSU time axis 后再分块。split seed 固定为 `1`，block/guard 参数继承 outer metadata。生成的 train/validation CSV 位于本 change 的 ignored output root，outer test CSV 原样保留。

split preflight 必须证明每个 domain 的 inner train/validation 在 stable sample identity、sequence group、history frame、target frame 和全部 referenced frame 上不相交，并记录输入 CSV digest、输出 CSV digest、样本/组数量、label histogram、seed 与 guard band。现有 outer split 在不同 CAV 间复用 RSU radar/BS-GPS 上下文，因此 launcher 会记录 outer-test 的资源重叠诊断，而不把它伪称为全资源独立的 strict test。任一 inner role 为空、inner 身份冲突或 metadata 缺失时，不生成训练任务。

所有 variants 使用完全相同的 inner split artifacts。配置设置 `training.model_selection.enabled=false`、`training.use_early_stopping=false` 和 `training.validation.interval_epochs=5`。validation 可在 epoch 5、10、15、20、25、30、35、40 及 runtime 必需的首次观测时运行，但不得更新 best checkpoint、scheduler 选择或停止条件；test 只在训练结束后显式评估 epoch-40 `last.pth`。

### 3. 六行矩阵使用具名 matched control 和严格 resolved diff

矩阵与 override 固定为：

- `H0-base`：无超参数 override，作为同硬件、同 inner split 重跑基线；
- `H1-BPA+`：相对 H0 只把 effective BPA outer weight 从 `0.2` 调为 `0.25`、modality BPA weight 从 `0.1` 调为 `0.15`；
- `H2-BPA-sharp`：相对 H1 只把 prototype temperature 从 `0.1` 调为 `0.08`、Gaussian sigma 从 `2.0` 调为 `1.5`；
- `H3-mask-tail`：相对 H0 只把 temporal rate 列表设为 `0.0,0.2,0.4,0.6,0.8,0.8`，drop-count 列表设为 `0,1,2,3,3`，保持 mask types 不变；重复值表示对 `0.8` 和 drop-3 进行预注册重加权；
- `H4-optimizer`：相对 H0 只使用 AdamW、`weight_decay=3e-4`，并把现有 cosine-warm-restarts scheduler 固定为单个 40-epoch cycle (`T_0=40`, `T_mult=1`, `eta_min=1e-6`)；学习率保持 `5e-4`；
- `H5-KL+`：相对 H0 只把 confidence-gated superset KL weight 从 `0.2` 调为 `0.5`。

launcher 必须为每行记录 `matched_control`、允许变化的 canonical field paths、resolved diff 和 effective values。每个语义字段只有一个配置来源，不写入 mirrored alias；出现未登记差异时 fail closed。H2 以 H1 为 matched control，因此 BPA strength 与 sharpness 不会被错误描述为同一单因素差异。

### 4. probe 只解析共同 batch，不修改训练协议

probe 在每个目标 GPU 的全新子进程中，用 H0 的真实 train dataloader、模型、loss、AMP、backward 和 optimizer step 测试预注册 batch candidates。默认候选是从 `16` 到显式 `--max-batch-size` 的所有 16 倍数；候选上限、顺序和 90% peak-reserved 门槛必须在运行前写入 manifest，结果产生后不得扩展候选集合。

每次 probe 后销毁子进程及模型/optimizer 状态，不把 probe checkpoint、optimizer state 或 RNG state传给正式训练。probe 只可写入统一 `train_batch_size`；不得按卡或 variant 设置不同 batch，不得缩放 learning rate，不得改变 optimizer/scheduler、epoch、split、loss、gradient accumulation、AMP、evaluation masks 或 checkpoint policy。最终选择所有 GPU 都成功且 peak reserved 不超过总显存 90% 的最大候选；没有候选通过时 fail closed。

probe report 记录 GPU identity、CUDA/PyTorch version、candidate、exit status、OOM/error、peak allocated/reserved、total memory 和 config fingerprint，并写入 ignored output root。probe 成功不等于训练成功，正式任务仍需独立记录 OOM 与完成状态。

### 5. 固定预算比较与选择规则

六个任务按 H0-H5 确定性映射到请求 GPU，每卡至多一个进程，统一 seed `1`、共同 batch、40 epochs、相同 inner validation 和相同 outer test/mask artifacts。只有 `checkpoints/last.pth` 的 metadata 证明 `completed_epoch=40` 且 config/split fingerprint 匹配时，运行才可进入汇总。

复用现有 MMW all-weather evaluator，对每行 epoch-40 `last.pth` 计算相同的 Clean、Drop1/2/3、temporal missing AUC 与 temporal Drop80。预注册分数为：

`J = 0.20 * Clean + 0.20 * mean(Drop1, Drop2, Drop3) + 0.25 * temporal_AUC + 0.35 * temporal_Drop80`。

相对 H0，任一候选的 Clean、`mean(Drop1, Drop2, Drop3)` 或 temporal Drop80 下降超过绝对 `0.005`（0.5 percentage point）即失去资格。合格行按 `J` 降序、variant id 升序确定唯一 development candidate；若 H0 排名第一，则结果为 `no_change`。缺少指标、mask/sample identity 不一致、运行未满 40 epoch 或非有限值时不得补值或改权重，整项汇总必须 fail closed。

### 6. development evidence 与 BPA/CMA formal change 永久隔离

由于 outer test 被用于六行超参数比较，本 change 的全部指标都必须标记 `development_only=true`、`claim_eligible=false` 和 `screening_consumed_test=true`。validation 独立只解决训练期 test 泄漏，不会把筛选结果升级成正式证据。

本 change 不修改 `validate-t2-mmw-bpa-cma-ablation` 的六方法定义、15 个待运行任务、固定 configs、batch protocol、checkpoint 或输出目录，也不得用筛选胜者替换其 T2 行或 CMA/BPA 权重。筛选胜者若需进入多 seed 正式比较，必须在本 change 之外先冻结配置，再通过独立 OpenSpec change 或明确扩展的 formal protocol 使用未参与调参的 evaluation evidence；不得复用本轮 test 指标作为论文 claim。

## Risks / Trade-offs

- [inner validation 减少 development train 样本] -> H0-H5 共享同一 group-safe inner split；只做筛选，不与既有 full-train T2 数值直接比较。
- [更大 batch 改变 optimizer step 数] -> batch 是全矩阵统一、预注册的 development hardware protocol；不做学习率缩放，也不写成单因素性能结论。
- [单 step probe 不能预测长训练碎片化] -> 保留 10% 显存余量，正式 run 独立记录 OOM；失败行不得用更小私有 batch 重跑后混入矩阵。
- [六行消费 outer test 导致调参偏置] -> 所有结果永久保持 development-only；正式证据必须使用冻结后的新多 seed 协议和未参与选择的 evidence。
- [重复字段产生隐性差异] -> 每个语义字段只保留一个 canonical path，并以该 path 做 allowlist diff。

## Migration Plan

1. 实现 launcher、inner split 生成、严格 config diff 与纯 dry-run manifest，先用 synthetic fixtures 覆盖 15-domain 和身份审计。
2. 实现 fresh-process batch probe 和 focused tests；预注册 probe candidates 后再在请求 GPU 上运行。
3. 生成 ignored configs/splits，完成六个 40-epoch 任务，并验证 epoch-40 `last.pth` 与 fingerprint。
4. 复用既有 evaluator 生成 development summary，只登记 candidate 或 `no_change`，不更新 claim registry。
5. 回滚时删除本 change 新增的 launcher/tests 以及 ignored output root；现有 MMW builder、BPA/CMA formal change 和历史 T2 artifacts 不受影响。

## Open Questions

无。任何把胜者升级为正式多 seed 结论的请求都属于后继 OpenSpec 范围，不在实现时临时决定。
