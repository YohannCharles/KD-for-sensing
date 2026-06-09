## Context

本仓库当前以 `src/kd_sensing` 包为主入口，训练、评估、数据构建、模型注册、metrics 和诊断均已有清晰模块边界。DeepSense6G 默认通过 `data.dataset.type: deepsense6g` 和 `data.dataset.scene` 选择场景，Scenario 31-34 已在现有 OpenSpec 中作为受支持场景；fusion 模型也已支持 image、radar、gps、lidar、mmwave 的可配置组合。

官方 BeamBench 仓库位于 `https://github.com/ITU-AI-ML-in-5G-Challenge/BeamBench`，本次审计到的临时 clone commit 为 `8e2c29a2afc898a69b9f9f7ece039d1e48ba60e8`。官方 README 标注环境为 Ubuntu 18.04 + CUDA 11.4，Dockerfile 使用 `nvidia/cuda:11.4.2-runtime-ubuntu18.04`、Python 3.7，并通过 pip 安装 PyTorch CUDA 11.3 wheel、Open3D、h5py、OpenCV、pandas、sklearn 等依赖。官方推荐评估命令为：

```bash
python3 challenge.py --gpu_id 0 --data_folder ./raw_data/test/ --csv ml_challenge_test_multi_modal.csv
```

官方仓库主要文件包括 `challenge.py`、`challenge_lstm.py`、`classical.py`、`config/camera_ae.cfg`、`config/gps_dense.cfg`、`libraries/general.py`、`models/ae_camera_model.py` 和 `models/dense_model.py`。需要注意的是，`challenge.py` 引用的 `ae_lidar_model`、`ae_radar_model`、`cl_camera_model`、`cl_radar_model`、`lstm_model`、`mmWave_*` 等源码在该仓库快照中只存在 `.pyc`，不完整源码会影响“官方原样运行”的可审计性。复现实现必须先记录这一事实，再决定采用官方评估脚本、最小 wrapper、或本仓库等价 baseline smoke。

## Goals / Non-Goals

**Goals:**

- 建立 BeamBench 官方 baseline 的可复现审计链路，覆盖官方代码、环境、数据结构、权重路径、评估入口和已知缺口。
- 在本仓库内提供数据检查、mock smoke、baseline 训练/评估 wrapper、指标测试和报告生成，不破坏现有 `kd_sensing` 包结构。
- 至少打通一个完整 pipeline：data loading、model forward、loss、metric、checkpoint save/load、validation/test evaluation。真实数据不可用时，使用明确标记的 mock pipeline 完成 smoke，不报告真实结果。
- 明确记录后续 image/LiDAR 关键区域注意力、beam-guided attention 和 cross-attention fusion 的插入位置。

**Non-Goals:**

- 不在第一阶段实现新的 attention、beam-guided attention、cross-attention fusion 或创新模型。
- 不重写官方 BeamBench 成完整新框架，也不把官方代码大规模复制到本仓库。
- 不伪造论文指标、官方指标或真实数据结果。
- 不为了跑通 smoke 删除关键模态、跳过 DBA/top-k metric 或绕过 checkpoint load/save。

## Decisions

### Decision 1: 官方代码采用“审计 + 薄 wrapper”，不直接深度合并

实现时优先保留官方 `challenge.py` 的评估语义，并通过 `scripts/eval_baseline.py` 或包内 CLI 记录命令、工作目录、官方 commit、权重路径和输出路径。若官方代码因缺失源码、权重或依赖无法运行，wrapper MUST 早失败并写出阻塞原因。

备选方案是把官方模型和数据读取逻辑全部迁入 `src/kd_sensing`。该方案会更容易测试，但会模糊“官方 baseline”与“本仓库改写 baseline”的边界，也更容易影响结果可比性，所以第一阶段不采用。

### Decision 2: 数据检查独立于训练入口

新增 `scripts/check_dataset.py` 作为可单独运行的检查工具，读取 CSV、解析官方 BeamBench/DeepSense6G 传感器列、检查文件存在性、统计缺失比例和 label 范围。该工具只做检查和报告，不自动移动、生成或删除真实数据。

备选方案是在 dataloader 初始化时顺手检查全部文件。该方案会让训练入口承担审计职责，且真实数据量较大时会拖慢训练；因此只保留必要的 lazy loading 错误，完整审计放到独立脚本。

### Decision 3: mock dataset 只服务 pipeline smoke

当 DeepSense6G Scenes 31-34 真实数据或官方 checkpoint 不可用时，实现一个极小 mock dataset，包含兼容 CSV、少量 image/LiDAR/radar/GPS/beam label 占位文件或等价张量。mock pipeline 必须能运行 dataloader、forward、loss、metric、checkpoint save/load 和 evaluation，但所有日志、报告和输出必须显式标记 `MOCK`。

备选方案是跳过 smoke 等待真实数据。该方案无法验证代码路径，后续实现风险较高；mock smoke 能较早发现 shape、dtype、metric 和 checkpoint 问题，但必须和真实结果严格隔离。

### Decision 4: 指标优先复用本仓库 circular metrics，同时核对官方 DBA

官方 `libraries/general.py` 中包含 `compute_DBA_score`，按 top-k minimum beam distance 和 `delta=5` 计算 DBA。本仓库已有 `kd_sensing.evaluation.metrics` 中的 `circular_beam_distance`、`circular_topk_min_distance`、`dba_from_circular_distances` 和 beam summary。实现应新增 BeamBench 指标 adapter 或测试，明确官方非环形/环形口径差异；如果复现实验使用 64-beam circular 语义，报告必须声明口径。

备选方案是完全照抄官方 metric。该方案有利于官方提交格式一致，但会和本仓库后续跨场景 residual/BGAM 的 circular 口径脱节；因此实现以“官方核对 + 本仓库可测试 metric”并存为目标。

### Decision 5: 后续 attention 插入点以现有 encoder/fusion 输出为准

后续 image 关键区域注意力优先插入 `src/kd_sensing/models/image.py` 或 `src/kd_sensing/models/image_encoders.py` 中 image encoder 的 frame feature/token 输出处；LiDAR 关键区域注意力优先插入 `src/kd_sensing/models/lidar.py` 的 `LidarFeatureExtractor.forward` 或现有 BGAM 相关模块；GPS embedding 位置为 `src/kd_sensing/models/gps.py` 的 `GpsFeatureExtractor.forward` 输出；cross-attention fusion 优先插入 `src/kd_sensing/models/fusion/cls_token_transformer.py` 中各模态 encoder 输出后、`torch.stack(modality_features, dim=1)` 前后或 transformer token 序列化前。

备选方案是在 dataloader 里提前做 attention mask 或特征筛选。该方案会把模型行为混入数据层，破坏模态懒加载与可替换 encoder 边界，因此不作为主插入点。

### Decision 6: 论文 Image AE + GPS Direct 使用专用朴素 fusion 实现

用户纠偏后，目标收窄为 Arnold22 BeamBench Table III 的 `Camera=AE, GPS=Direct, Fusion=Yes` 行。为避免把现有 residual/gated/attention 实验误当作论文 baseline，本 change 新增专用 `image_ae_gps.py`：直接读取本地 DeepSense6G sequence CSV，先训练或加载 `CameraAutoEncoder`，冻结 AE encoder 后将 image latent 与 GPS direct feature concat，再训练 64-beam classifier。该入口输出 BeamBench DBA/top-k metrics、checkpoint、history 和 predictions，但报告必须声明：未使用官方 pretrained AE/fusion 权重和官方完整训练搜索流程时，本地数值不能等同 Table III。

备选方案是继续复用通用 `modular_sequence` 或现有 camera residual workflow。前者会受到通用 image profile、sequence core 和配置验证影响，后者结构上不是论文目标行；因此保留它们作为相关能力，但论文目标行采用窄而明确的本地训练入口。

### Decision 7: 训练加速优先缓存冻结 AE latent，而不是减少训练量

3090 + 多核 CPU 环境中，Image AE + GPS Direct 的主要瓶颈不是 fusion MLP，而是每个 epoch 重复从磁盘解码图片并运行冻结 AE encoder。为保持样本数、image size、epoch 上限、early stopping 和 DBA 选 best 语义不变，专用入口默认在 AE checkpoint 确定后预计算 train/test 的 camera AE latent，并将 cache 写入当前 run 的 ignored 输出目录。fusion 阶段只读取 latent、GPS 和 label；当 AE encoder 未冻结或用户关闭 cache 时，仍可回到在线 forward 路径。

同时启用可配置的 AMP、TF32、fused AdamW、pin memory、persistent workers、prefetch 和 non-blocking transfer。AMP/TF32 属于 CUDA 吞吐优化，可通过 CLI 或 override 关闭；dry-run 继续强制小样本、一轮和零 worker，避免验证命令变慢。报告记录加速开关和 cache 路径，便于审计训练结果是否来自 cached latent 路径。

### Decision 8: 四场景复现实验采用论文 split runner + Table III 汇总

用户进一步纠正论文协议：Table III 的 Camera AE+GPS 行不是逐场景分别训练，而是在 scenes 32、33、34 上联合训练，并在 scenes 31、32、33、34 上测试同一个模型，其中 scene31 是未见分布。因此新增 `run_beambench_image_ae_gps_tableiii.py`，一次构建 scenes 32-34 的联合训练集、训练或复用一个 Camera AE、冻结 AE encoder 后训练一个 fusion classifier，再分别评估 scenes 31-34，并在汇总中同时列出本地 `official_top3_dba`、论文目标值和差距。

该 runner 支持两类 GPS Direct 特征：`paper_distance_angle` 贴近官方 `challenge.py` 的 `[distance, calibrated_angle_deg]` 二维输入；`paper_calibrated_relative_polar` 作为三维 `[distance, sin(theta), cos(theta)]` ablation。best checkpoint 选择保留可审计字段：为了更干净的本地科学比较，推荐 `validation` 模式；为了观察本地 upper-bound，可运行 `test_as_validation`，但报告必须标注该口径不等同官方完全 unseen test evaluation。

### Decision 9: scene31 泛化优先修复官方 GPS 角度与 AE 表征维度

用户进一步收窄为“先提高 scene31 泛化”。审计发现 `paper_distance_angle` 的本地实现曾使用 `atan2(x, y)`，而官方 `challenge.py` 使用 `arctan(x/y)`，前者会让 scene31/34 角度跨越 `±180` 断点；同时 scene32 的官方校准角不是 `-0.76`，而是 `-0.8125375604986421 + pi/2 = 0.7583`。这两点会污染 scenes 32-34 联合训练中的 GPS 坐标系，尤其影响对 scene31 的迁移。

修复后，frozen AE feature cache 签名必须包含 GPS 特征版本和 scene 校准角，防止旧 cache 被静默复用。Camera AE 默认 latent 也从本地早期 128 维调到更贴近官方 `camera_ae.cfg` 的 512 维；本地 scene31-only validation 实验显示，GPS 修复 + 512 维 AE 可将 scene31 `official_top3_dba` 提升到 `0.6824`，略高于 Table III scene31 `0.6731`。该决策只服务 scene31 单项泛化，暂不声称 scenes 32-34 或 overall 已重新优化完成。

### Decision 10: 完整四场景复查支持 eval-only checkpoint 汇总

用户随后要求同时重新追 scenes 32-34 和 overall。为避免每次复查都重新训练 fusion 并引入 CUDA 随机性，Table III runner 新增 eval-only 模式：传入 `--fusion-checkpoint` 后直接加载已有 paper-split checkpoint、恢复 checkpoint 中保存的 GPS scaler 和 AE checkpoint，并分别评估 scenes 31-34，输出同样的 Table III CSV/Markdown/JSON 汇总。

使用 scene31 泛化专项得到的 strict validation checkpoint 做 eval-only，scenes 31-34 的 `official_top3_dba` 为 `0.6824/0.7431/0.8371/0.8158`，weighted overall 为 `0.7594`，四个场景和 overall 均高于 Table III 的 `0.6731/0.6173/0.8171/0.7313` 和 `0.7127`。该结果仍然是本地 sequence split 和本仓库训练流程，不等同官方 unseen test packaging。

## Risks / Trade-offs

- [Risk] 官方仓库缺失部分模型源码或预训练权重，导致原样 `challenge.py` 无法运行。→ Mitigation：在 `BASELINE_REPORT.md` 和 `results/reproduce_baseline.md` 记录缺失文件、缺失权重和最小复现阻塞点；mock smoke 只标记为 MOCK，不替代真实结果。
- [Risk] 当前 `kd_mm_beam` 环境与官方 Ubuntu 18.04/CUDA 11.4/Python 3.7 不完全一致。→ Mitigation：`ENVIRONMENT.md` 同时记录官方要求、当前环境、偏差和最小可运行方案；项目相关 Python 命令一律使用 `conda run -n kd_mm_beam`。
- [Risk] 官方 DBA 与本仓库 circular DBA 口径可能不同。→ Mitigation：测试同时覆盖官方公式和本仓库 circular helper，报告明确 metric 口径与是否可和官方 leaderboard 直接比较。
- [Risk] mock 数据过于简化，掩盖真实 CSV/path/shape 问题。→ Mitigation：mock 只用于 smoke；真实复现必须先运行 `scripts/check_dataset.py`，并在报告中分开记录 mock 与 real data。
- [Risk] 引入官方代码 patch 后难以判断结果可比性。→ Mitigation：所有 patch 写入 `PATCH_NOTES.md`，说明修改原因、最小范围和是否影响官方结果。

## Migration Plan

1. 创建 BeamBench 复现配置、数据检查脚本、wrapper 和文档骨架。
2. 先运行官方源审计和 dataset check；若真实数据或权重缺失，记录阻塞并运行 mock smoke。
3. 在真实数据和权重到位后运行官方推荐评估命令或等价 wrapper，生成 `BASELINE_REPORT.md` 与 `results/reproduce_baseline.md`。
4. 完成指标测试、架构边界测试和 OpenSpec 校验。

回滚策略：所有新增内容集中在本 change 对应脚本、窄模块、配置和文档中；不自动迁移数据、不删除输出、不修改官方源码大块逻辑。若 wrapper 方案不可行，可删除 wrapper 并保留审计文档与 mock smoke，不影响现有 `kd_sensing` 训练入口。

## Open Questions

- 官方预训练模型下载位置是否仍可访问，以及权重文件名是否与 `challenge.py` 中的 `results/models/*` 路径完全一致。
- DeepSense6G Scenes 31-34 本地数据 CSV 是否采用官方 `ml_challenge_test_multi_modal.csv` 字段，还是本仓库已预处理的 sequence CSV 字段；实现需要通过 checker 同时支持或明确转换关系。
- 最终报告中“官方 DBA”和“本仓库 circular DBA”是否都展示，还是以官方 DBA 为主、circular DBA 作为后续研究口径。
