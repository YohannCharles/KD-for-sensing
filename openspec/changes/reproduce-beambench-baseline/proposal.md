## Why

当前仓库已经具备 DeepSense6G 多模态 beam prediction、跨场景评估、GPS residual/BGAM 和 fusion 的基础能力，但还缺少一条对官方 BeamBench baseline 的干净复现链路。先复现官方 BeamBench，可以为后续 image/LiDAR 关键区域注意力、beam-guided attention 和 cross-attention fusion 提供可比较、可回滚、可扩展的起点，避免在 baseline 尚未闭环前引入创新模块。

## What Changes

- 新增 BeamBench 官方仓库审计与复现 workflow，记录官方代码来源、commit、目录结构、入口、依赖、数据路径、模型权重路径、评估命令和已知缺口。
- 新增 BeamBench 数据接口检查工具，验证 CSV 是否存在、传感器文件引用是否存在、label/beam index 是否合法、各模态缺失比例以及 scene/sample/sequence 标识解析情况。
- 新增最小 mock dataset 方案，仅用于 dataloader、forward、loss、metric、checkpoint save/load 和 evaluation smoke test；所有 mock 产物和日志必须显式标记 `MOCK`，不得被报告为真实复现结果。
- 新增官方 baseline 复现入口或薄 wrapper，优先保留官方 `challenge.py` 评估语义，再在本仓库现有 `src/kd_sensing` 架构内提供可测试、可记录的训练/评估辅助入口。
- 针对用户明确指定的 Arnold22 BeamBench Table III `Camera=AE, GPS=Direct, Fusion=Yes` 行，新增本地训练模型、专用 CLI 和合理默认配置：先训练/加载 Camera AE，再冻结 AE encoder 与 GPS direct feature 做 concat fusion，不以 residual/gated/attention 模型替代该 baseline。
- 针对 RTX 3090 + 多核 CPU 的本地训练反馈，为专用 Image AE + GPS Direct 入口新增冻结 AE latent cache、AMP/TF32/fused AdamW 和 DataLoader 并行优化；不通过减少样本、epoch 或输入分辨率换速度。
- 针对“尽可能复现并达到论文指标”的要求，新增论文 split runner：scenes 32-34 联合训练、scenes 31-34 分别测试同一个 checkpoint，并输出 Table III 风格汇总，记录本地 DBA、论文目标 DBA、差距以及本地 sequence split 与官方 unseen test 的可比性限制。
- 新增 BeamBench 指标核对与测试，覆盖 DBA、top-k accuracy、top-3 DBA 或官方等价口径；如官方实现不清晰，则在本仓库补充独立指标实现并说明与官方指标的差异。
- 新增复现文档与报告：`README_REPRODUCE.md`、`ENVIRONMENT.md`、`DATASET_STRUCTURE.md`、`BASELINE_REPORT.md`、`PATCH_NOTES.md`、`TODO_FOR_ATTENTION_MODULE.md` 和 `results/reproduce_baseline.md`。
- 不实现新的 attention、beam-guided attention 或 cross-attention fusion 模块；本 change 只标注后续最适合插入这些模块的文件、类和 `forward` 位置。
- 不删除关键模态、不跳过 metric、不伪造论文或官方结果；真实数据、权重或 CUDA 不满足时，必须记录阻塞点和下一步，而不是给出虚假指标。

## Capabilities

### New Capabilities
- `beambench-baseline-reproduction`: 定义官方 BeamBench baseline 审计、数据检查、mock smoke、训练/评估 wrapper、指标核对、复现报告和后续 attention 插入点记录的行为契约。

### Modified Capabilities
- 无。本 change 应复用现有 `dataset-directory-layout`、`deepsense6g-scene-selection`、`modality-aware-data-loading`、`configurable-multimodal-fusion` 和 `experiment-workflow` 能力，不改变它们的既有 requirement。

## Impact

- 代码：新增或整理 `scripts/check_dataset.py`、`scripts/train_baseline.py`、`scripts/eval_baseline.py`、`scripts/train_beambench_image_ae_gps.py`，以及必要的 `src/kd_sensing/baselines/beambench/`、`src/kd_sensing/data/`、`src/kd_sensing/models/`、`src/kd_sensing/evaluation` 或 `src/kd_sensing/utils/` 窄模块。
- 配置：新增 BeamBench 复现相关配置，优先放在 `configs/` 下，使用现有配置加载与路径解析约定。
- 文档与报告：新增根目录复现文档、patch notes、attention TODO 和 `results/reproduce_baseline.md`；真实训练输出、日志、checkpoint 和数据仍属于本地产物，默认不得纳入源码变更。
- 外部依赖：官方 BeamBench README 指向 Ubuntu 18.04、CUDA 11.4、Python 3.7 和 PyTorch CUDA wheel；本仓库最小可运行方案必须优先使用 `kd_mm_beam`，并把任何版本偏差写入 `ENVIRONMENT.md`。
- 官方代码：如需修改或 vendoring 官方代码，只能使用最小 patch，并在 `PATCH_NOTES.md` 说明修改原因、范围和是否影响官方结果可比性。
