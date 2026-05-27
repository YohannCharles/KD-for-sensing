## Context

当前项目已经完成 `src/kd_sensing` 包化、配置驱动训练/评估、DeepSense6G 场景描述符、模态感知数据加载、ResNet-18 RGB image encoder、GPS/Radar/LiDAR/mmWave 单模态与 fusion 配置，以及 `cls_token_transformer_fusion` 的 token 化融合能力。现有 DeepSense6G 场景注册覆盖 9、31、32，训练数据构建默认是一组 train/test CSV 对应一个 scene；输出目录也按单 scene 分组。

《跨场景自适应方案.md》提出的 HiST-Beam 快速验证需要跨 31-34 场景训练、target adapt/test 防泄漏、层次化 beam label、shared/private 解耦、target adapter 和 prototype alignment。它不是简单新增一个模型文件，而是跨数据构建、模型输出、loss、metrics、配置矩阵、checkpoint/prototype 产物和实验编排的横切变更。

本设计遵守现有项目边界：不恢复旧根目录脚本，不复制 dataset 或训练循环，不把本地数据和运行产物纳入源码；新增能力通过注册表、窄模块和包内 CLI 接入，并继续使用 `kd_mm_beam` 环境运行项目相关 Python 命令。

## Goals / Non-Goals

**Goals:**
- 支持 DeepSense6G scenarios 31-34 的 leave-one-scene-out 快速验证。
- 支持 `image`、`radar`、`gps` 三模态 HiST-Beam 变体 V0-V6，通过配置开关切换。
- 支持 target label budget `0/5/10/20/50`、seed 矩阵、target adapt/test 防泄漏和粗分组优先的 few-shot 采样。
- 在现有训练/评估契约上接入 hierarchical loss、decoupling loss、prototype loss、adapter/full fine-tuning 适应和 coarse/fine 指标。
- 保存可复现实验产物：final config、metrics、checkpoint、prototype、predictions、adapt log 和 LOSO 汇总表。
- 为后续 LiDAR、angular smoothing、更复杂 transformer 或论文完整对比保留配置边界。

**Non-Goals:**
- 不在第一阶段实现 LiDAR 主实验、复杂 BEV、MAML、完整 AMBER 对比或复杂 MI/contrastive loss。
- 不新增 `train_source.py`、`adapt_target.py`、`run_loso.py` 等根目录长期入口；如需命令入口，使用 `kd_sensing.cli` 下的包内 CLI 和 `pyproject.toml` console script。
- 不把方案中的 10 维 GPS 手工特征作为首阶段默认数据契约。快速版优先复用现有 `gps_feature_mode: relative_polar` 和 `[B, T, 3]` 输入；10 维局部平面特征可作为后续独立 GPS profile。
- 不改变既有单场景训练、评估和 canonical fusion 配置的默认行为。

## Decisions

### Decision 1: 将 HiST-Beam 作为新注册模型接入 fusion 工作流

新增 `hist_beam_fusion` 或等价注册名，复用 `experiment.task: fusion` 的 batch preparation 和模态字段。模型内部包含三层结构：
- 模态 encoder/projector：首阶段启用 `image`、`radar`、`gps`；image 默认使用 `resnet18_imagenet_rgb` 或项目已支持的 RGB/ImageNet encoder 配置，radar/GPS 复用现有 feature extractor 风格。
- tiny token fusion：每个模态产生 token，投影到 `d_model=256` 后做 2-layer Transformer 或等价轻量融合，得到 fused representation。
- HiST head：按配置选择 flat head、hierarchical head、shared/private branches、scene classifiers、private adapter 和 prototype diagnostics。

原因：现有 `cls_token_transformer_fusion` 已证明 token fusion 和 diagnostics 适合本仓库，但 HiST-Beam 需要输出 `coarse_logits`、`fine_logits`、`beam_log_probs`、`c`、`s`、`s_adapt` 和 scene logits。把这些塞入既有模型会扩大默认 fusion 的语义；独立注册模型能降低回归风险。

备选方案：直接扩展 `cls_token_transformer_fusion`。放弃原因是默认 fusion 目前服务通用 beam/multitask/KD，加入 hierarchical/adapter 状态会让通用模型过早承担研究特化逻辑。

### Decision 2: 用训练扩展或 objective hook 接入 HiST loss，而不是改默认 CE 路径

HiST-Beam 的主 logits 可以是 beam-level log probability，用于 Top-K 评估；训练时需要组合 `L_hier + lambda_flat L_flat + lambda_orth L_orth + lambda_scene_c L_scene_c + lambda_scene_s L_scene_s`，adapter 阶段还需要 supervised/unsupervised/prototype loss。实现应在 `engine` 下新增窄模块，例如 `hist_beam_training.py`、`hist_beam_losses.py` 或 prediction objective hook，默认 beam CE 路径保持不变。

原因：现有 trainer 和 evaluation pass 对普通模型仍应稳定；HiST loss 只在配置显式启用时接管。

备选方案：把 hierarchical loss 注册为普通 criterion。放弃原因是 criterion 当前只接收 logits/labels，不足以读取 diagnostics、scene labels、prototypes 和 adapter state。

### Decision 3: LOSO 数据构建使用跨场景窄模块，不复制 DeepSense6GDataset

新增跨场景 split builder，负责生成四个 fold：
- A: source `[31, 32, 33]`，target `34`
- B: source `[31, 32, 34]`，target `33`
- C: source `[31, 33, 34]`，target `32`
- D: source `[32, 33, 34]`，target `31`

source training loader 由多个 scene 的 train split 组合而成；target scene 使用确定性 20%/80% 拆成 `target_adapt` 和 `target_test`。每个底层 dataset 仍是现有 `DeepSense6GDataset`，并通过 `retarget_deepsense_dataset_config` 或等价逻辑切换 scene，normalizer/scaler 只允许从 source train 或 target_adapt 的训练允许部分产生，不能从 target_test 产生。

原因：现有 dataset 已处理 CSV、模态按需读取、beam label cache、GPS/Radar/Image/LiDAR 等细节；复制 dataset 会制造长期维护成本。

备选方案：预先生成跨场景合并 CSV。放弃原因是它容易隐藏 target_test 泄漏，且会让本地数据产物进入流程核心。

### Decision 4: few-shot sampling 以 coarse group 分层为默认，失败时确定性退化

few-shot labeled target 样本从 `target_adapt` 中抽取。优先按 `beam // group_size` 做分层覆盖；当某些 group 样本不足或 budget 小于可覆盖组数时，用 seed 控制的确定性随机补齐或退化。采样 manifest 必须记录 fold、target scene、budget、seed、样本 id、beam、coarse group 和是否 labeled。

原因：方案的核心假设是 shared coarse semantics，因此 few-shot 样本覆盖 coarse group 比纯随机更可解释。

备选方案：完全随机采样。保留为退化路径和 ablation，但不是默认。

### Decision 5: target adaptation 使用同一模型权重的两种训练策略

V4/V5 adapter 适应加载 V3 source checkpoint，冻结 encoder、fusion、shared branch、coarse head 和原始 private branch，只训练 private adapter、fine head 末层、可选 LayerNorm affine 和 prototype bank。V6 full fine-tuning 加载同一 V3 checkpoint，在 labeled target 上更新全部参数，使用更小学习率。

原因：这样 V4/V5/V6 的初始条件一致，效率和性能比较可解释。

备选方案：为 adapter 和 full fine-tuning 分别训练 source model。放弃原因是会增加运行成本并混入 source seed 差异。

### Decision 6: 第一阶段配置固定三模态，接口允许后续扩展

快速验证默认使用 `modalities: ["image", "radar", "gps"]`。LiDAR 不进入 P0-P5 主线，但模型和配置校验不应写死只能三模态；后续可以增加 `lidar` adapter/encoder 配置。

原因：LiDAR 预处理和 cache 成本高，且本次创新主要在层次化 label、shared/private 解耦和 target adapter。

备选方案：一开始使用全部模态。放弃原因是会把验证风险转移到数据预处理吞吐和 cache 稳定性上。

## Risks / Trade-offs

- [Risk] Scenario 33/34 本地数据目录或 CSV 命名与 31/32 不完全一致 → Mitigation: 场景描述符支持显式 `data_root`、`train_csv_name`、`test_csv_name` 覆盖；CLI smoke 先验证场景解析和 CSV 可读性。
- [Risk] `target_adapt`/`target_test` 拆分在样本或 `seq_index` 层泄漏 → Mitigation: split builder 以完整 `seq_index` 优先拆分；metadata 记录 adapt/test `seq_index`，测试拒绝交集。
- [Risk] HiST loss 接入默认 trainer 后影响普通 beam 实验 → Mitigation: 所有 HiST loss 仅在 `hist_beam.enabled` 或模型类型匹配时启用；普通配置走既有 CE。
- [Risk] Scene confusion 的 GRL 和 scene classifier 在小 batch 或不均衡 source scene 上不稳定 → Mitigation: scene loss 默认较小，记录 scene-from-c/s probe 指标，允许配置关闭对应 loss。
- [Risk] Prototype alignment 的伪标签噪声导致 0-label 适应退化 → Mitigation: 使用 confidence threshold、warmup 开关和忽略低置信样本；metrics 记录 proto coverage。
- [Risk] ResNet-18 预训练权重依赖 torchvision 或首次下载 → Mitigation: 沿用现有 ResNet-18 encoder 错误信息和权重配置；smoke 配置可允许 `pretrained: false` 或冻结 backbone。
- [Risk] 运行矩阵过大导致 2-4 周验证变慢 → Mitigation: tasks 明确 P0-P5 顺序，先跑 V0/V1/V3 和 budgets `0/10/50`，完整 `0/5/10/20/50` 作为后续扩展。

## Migration Plan

1. 扩展 DeepSense6G scene descriptor 到 33/34，并补场景解析测试。
2. 增加跨场景 LOSO split builder、few-shot sampler 和 metadata 测试。
3. 增加 HiST-Beam 模型、loss、metrics 和单 batch 单元测试。
4. 接入 source training、prototype 保存、adapter/full fine-tuning adaptation 和 evaluation/prediction 导出。
5. 增加配置矩阵、包内 CLI/orchestrator 和 smoke 测试。
6. 运行 OpenSpec strict validate、相关 pytest、CLI help smoke；最终实现完成后运行 `conda run -n kd_mm_beam pytest -q`。

回滚策略：该能力通过新增模型类型、配置目录和 CLI 入口启用；若发现不稳定，可移除或停用 HiST 配置，不影响既有单场景 canonical 训练路径。

## Open Questions

- Scenario 33/34 的默认 CSV 名是否与现有 `train_seqs_RA_GPS_LIDAR.csv` / `test_seqs_RA_GPS_LIDAR.csv` 完全一致，需要在本地数据到位后确认。
- target adapt/test 是否必须严格按 `seq_index` 划分；若部分场景缺少 `seq_index`，需要退化到 sample_id 级拆分并明确记录。
- HiST-Beam 快速版默认 `num_pred` 是 1 还是沿用项目默认 3。实现应支持 horizon 维，但 quick config 可以先用 `num_pred: 1` 降低验证复杂度。
