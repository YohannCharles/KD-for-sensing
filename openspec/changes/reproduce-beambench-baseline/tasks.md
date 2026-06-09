## 1. 官方源码与环境审计

- [x] 1.1 记录 BeamBench 官方仓库 URL、临时 clone 位置、commit hash、README 推荐评估命令、默认数据目录、默认模型目录和预测输出目录。
- [x] 1.2 审计官方 `README.md`、`Dockerfile`、`challenge.py`、`challenge_lstm.py`、`classical.py`、`config/*.cfg`、`libraries/general.py` 和 `models/`，整理训练入口、评估入口、配置文件、模型权重命名和已缺失源码。
- [x] 1.3 使用 `conda run -n kd_mm_beam python -c "import sys, torch; print(...)"` 或等价命令采集当前 Python、CUDA、PyTorch、torchvision、GPU 型号和 CUDA 可用性。
- [x] 1.4 创建 `ENVIRONMENT.md`，同时记录官方 Ubuntu/CUDA/Python/PyTorch 要求、当前 `kd_mm_beam` 环境、版本偏差、最小可运行方案和不能完全匹配时的阻塞说明。

## 2. 数据接口检查与 mock 数据

- [x] 2.1 设计 BeamBench/DeepSense6G CSV 字段解析 helper，覆盖官方 `unit1_rgb_*`、`unit1_lidar_*`、`unit1_radar_*`、`unit1_loc`、`unit2_loc_*` 以及本仓库 sequence CSV 中的 label、scene、sample、seq、timestamp 等等价字段。
- [x] 2.2 实现 `scripts/check_dataset.py`，支持 `--data-root`、`--csv`、`--scene`、`--num-beams`、`--beam-shift`、`--output` 和必要的 dry-run 默认行为。
- [x] 2.3 为 dataset checker 添加测试，使用 `conda run -n kd_mm_beam pytest <相关测试> -q` 验证 CSV 缺失、传感器路径缺失、非法 label、缺失比例统计和 scene/sample/sequence 解析。
- [x] 2.4 创建极小 mock dataset 或 mock fixture，包含兼容 CSV、可读取的 camera/LiDAR/radar/GPS/beam label 占位输入，并确保所有 mock artifact 明确标记 `MOCK` 或 `mock_data: true`。
- [x] 2.5 运行 mock dataset checker 和 mock pipeline smoke，命令必须使用 `conda run -n kd_mm_beam`，并将结果写入 `BASELINE_REPORT.md` 和 `results/reproduce_baseline.md` 的 mock 分节。

## 3. 指标核对与测试

- [x] 3.1 对照官方 `libraries/general.py::compute_DBA_score`，在本仓库新增或复用 BeamBench metric helper，明确官方 DBA、top-k accuracy、top-3 DBA 和 64-beam circular DBA 的字段命名。
- [x] 3.2 添加指标单元测试，覆盖 perfect prediction 指标最高、beam index 偏差越大 DBA 下降、ground truth 在 top-k 中时 top-k accuracy 命中。
- [x] 3.3 使用 `conda run -n kd_mm_beam pytest <metric tests> -q` 运行指标测试，并在 `BASELINE_REPORT.md` 记录 metric 口径、beam shift 和是否 circular。

## 4. baseline wrapper 与完整闭环

- [x] 4.1 新增 BeamBench baseline 配置，至少覆盖官方 Camera AE + GPS、GPS dense、late fusion pretrained features，以及官方仓库中可识别的 camera/radar/LiDAR/GPS 组合；不可运行项必须有 blocked reason。
- [x] 4.2 实现 `scripts/eval_baseline.py` 或包内薄入口，优先调用官方 `challenge.py` 语义，记录 command、official commit、data folder、CSV、type list、seed、checkpoint path 和 prediction path。
- [x] 4.3 实现 `scripts/train_baseline.py` 或等价 wrapper；若官方训练入口不可用，则用本仓库 `kd_sensing` 架构打通一个等价 baseline 的 data loading、forward、loss、metric、checkpoint save/load 和 validation/test evaluation。
- [x] 4.4 运行至少一个 baseline 闭环；真实数据和权重不可用时，运行 mock 闭环并显式记录 `MOCK`，真实结果栏必须标记 blocked 而非填入虚假指标。
- [x] 4.5 确保 wrapper 不删除关键模态、不跳过 DBA/top-k metric，并且 checkpoint 与运行日志写入 ignored 的 outputs/logs/results 路径或被明确标记为本地产物。

## 5. 复现文档与后续扩展说明

- [x] 5.1 编写 `README_REPRODUCE.md`，给出环境检查、数据检查、mock smoke、官方评估、训练/评估 wrapper 和报告生成的逐步命令，所有项目相关 Python 命令使用 `conda run -n kd_mm_beam`。
- [x] 5.2 编写 `DATASET_STRUCTURE.md`，说明官方 BeamBench 期望目录、CSV 字段、camera/LiDAR/radar/GPS 读取方式、label/beam index 范围和 Scenes 31-34 放置建议。
- [x] 5.3 编写 `BASELINE_REPORT.md` 和 `results/reproduce_baseline.md`，逐次记录 command、当前仓库 commit、官方 commit、environment、dataset split、modalities、checkpoint path、metrics、日志路径和 mock/real 标记。
- [x] 5.4 编写 `PATCH_NOTES.md`，列出所有官方代码或本仓库代码改动、修改原因、是否影响官方结果可比性和回滚方式。
- [x] 5.5 编写 `TODO_FOR_ATTENTION_MODULE.md`，明确 image encoder feature 输出、LiDAR encoder feature 输出、GPS embedding、late fusion/concat/classifier head、`CLSTokenTransformerFusionNet.forward` 和 dataloader batch 字段中的插入点。

## 6. 验证与收尾

- [x] 6.1 运行 `openspec validate reproduce-beambench-baseline --strict` 和 `openspec status --change reproduce-beambench-baseline`，修复所有 OpenSpec 问题。
- [x] 6.2 运行新增测试和相关快速检查，例如 `conda run -n kd_mm_beam pytest <新增测试> -q`、`conda run -n kd_mm_beam python scripts/check_dataset.py --help`、`conda run -n kd_mm_beam python scripts/eval_baseline.py --help`。
- [x] 6.3 对涉及架构、CLI 或公共 workflow 的改动，运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` 和相关 CLI help 检查。
- [x] 6.4 最终检查 `git status --short`，确认未纳入真实数据、训练输出、日志、cache、新生成 checkpoint 或临时验证产物。

## 7. 用户纠偏：论文 Image AE + GPS Direct 本地训练

- [x] 7.1 明确 Arnold22 BeamBench Table III 目标行为 `Camera=AE, GPS=Direct, Fusion=Yes`，目标 DBA 为 Scene31=0.6731、Scene32=0.6173、Scene33=0.8171、Scene34=0.7313、Overall=0.7127。
- [x] 7.2 实现贴合论文行的本地模型：Camera AE latent + GPS direct feature concat fusion + 64-beam classifier，不以 residual/gated/attention 模型替代。
- [x] 7.3 实现专用训练入口，直接从本地 DeepSense6G scene31-34 sequence CSV 读取 camera/GPS/future beam，先训练或加载 Camera AE，再冻结 AE encoder 训练 fusion classifier。
- [x] 7.4 为专用训练入口补合理默认配置、CLI、thin script、测试和架构入口清单，所有验证使用 `conda run -n kd_mm_beam`。
- [x] 7.5 使用本地 scene31 数据运行 dry-run，确认真实本地数据训练闭环可运行，并在报告中标明 dry-run 指标不可作为论文数值。

## 8. 用户反馈：3090 多核服务器训练加速

- [x] 8.1 保持样本数、epoch、image size、early stopping 和 DBA 选 best 逻辑不降级，明确加速不能通过少训练冒充。
- [x] 8.2 为 Image AE + GPS Direct 入口实现冻结 AE latent cache，使 fusion 阶段不再每个 epoch 重复读图和运行 AE encoder。
- [x] 8.3 为专用入口补 AMP、TF32、fused AdamW、non-blocking transfer、pin memory、persistent workers 和 prefetch 配置。
- [x] 8.4 更新合理默认配置和 CLI 开关，面向 RTX 3090 + 多核 CPU 提高吞吐，同时保留可关闭加速的 override。
- [x] 8.5 补测试并使用 `conda run -n kd_mm_beam` 验证新增加速路径、CLI help、架构边界和 OpenSpec strict 校验。

## 9. 用户要求：尽可能复现论文 Camera AE+GPS 四场景指标

- [x] 9.1 核对论文 Camera AE+GPS 方法、DBA 口径、Table III 目标值和复现边界，明确官方权重/官方测试集不可用时的本地可比性限制。
- [x] 9.2 修复单场景 CLI 的 scene/output 默认行为，避免 `--scene 32/33/34` 覆盖 scene31 输出目录。
- [x] 9.3 实现论文 split runner：scenes 32-34 联合训练，同一 checkpoint 分别评估 scenes 31-34，聚合 Table III 风格 CSV/Markdown/JSON 报告，并记录与论文目标值差距。
- [x] 9.4 保留或补充可选的本地 validation selection 模式，避免只能用 test CSV 选择 best checkpoint；报告中必须区分 `test_as_validation` 和 `validation`。
- [x] 9.5 运行本地 Camera AE+GPS 论文 split 实验，尽可能接近 Table III 目标，并将命令、指标、差距和限制写入 `BASELINE_REPORT.md` 与 `results/reproduce_baseline.md`。
- [x] 9.6 使用 `conda run -n kd_mm_beam` 运行新增测试、CLI help、架构边界和 OpenSpec strict 校验。

## 10. 用户收窄：优先提升 scene31 泛化

- [x] 10.1 统计 scenes 32-34 与 scene31 的 GPS Direct 角度/距离分布和 future beam label 分布，定位 scene31 泛化瓶颈。
- [x] 10.2 修复 `paper_distance_angle` 特征与官方 `challenge.py` 不一致的问题：角度使用 `arctan(x/y)` 而不是 `atan2(x, y)`，避免 scene31/34 的 180 度断点。
- [x] 10.3 为 frozen AE feature cache 签名加入 GPS 特征版本，确保旧 `atan2` cache 不会被继续复用。
- [x] 10.4 更新文档与推荐命令，说明 scene31 专项复现实验需要重新生成 cache，并先只评估 scene31。
- [x] 10.5 使用 `conda run -n kd_mm_beam` 运行相关测试、CLI help 和 OpenSpec strict 校验。

## 11. 用户继续：重新追 scenes 32-34 和 overall

- [x] 11.1 使用已修复 GPS 公式、scene 校准角和 512 维 AE，在 scenes 32-34 联合训练并评估 scenes 31-34，记录 strict validation 的每场景和 overall。
- [x] 11.2 若 strict validation 未接近 Table III overall，运行 `test_as_validation` 本地 upper-bound，区分 checkpoint 选择口径。
- [x] 11.3 根据 31-34 指标判断剩余主要缺口，必要时补可审计配置而不是只报告单场景结果。
- [x] 11.4 更新 `BASELINE_REPORT.md`、`results/reproduce_baseline.md`、OpenSpec artifacts 和推荐命令。
- [x] 11.5 使用 `conda run -n kd_mm_beam` 运行相关测试、CLI help 和 OpenSpec strict 校验。
