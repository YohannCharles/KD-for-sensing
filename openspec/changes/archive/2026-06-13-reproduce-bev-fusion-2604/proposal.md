## Why

arXiv:2604.05668 提出面向 mmWave beam prediction 的顺序多模态 BEV-Fusion，在 DeepSense6G scenarios 32/33/34 上报告 86.52% overall DBA，但论文当前只给出方法和结果，代码与训练权重尚未随稿开放。仓库已有 DeepSense6G 多模态、2604-style 对照复核和 JEPA/Image+GPS baseline，需要补上一条可审计的论文本体复现方案，避免把现有对照模型误写成 BEV-Fusion 复现。

## What Changes

- 新增 arXiv:2604.05668 BEV-Fusion 复现能力，覆盖论文协议、模型结构、训练配置、ablation、指标和报告。
- 新增 paper-aligned DeepSense6G S32/S33/S34 实验配置族：5 帧历史输入、`future_beam1` 单 horizon 标签、64 beam 分类、linear DBA 口径、scene-level 与 macro/overall 汇总。
- 新增 `bev_fusion_2604` 模型注册方案：camera-to-BEV cross-attention、LiDAR/Radar/GPS-to-BEV 分支、逐时隙 BEV spatial fusion、temporal transformer 和 beam classifier。
- 新增 dual-path GPS-to-BEV 设计：GPS 空间 mask 参与 BEV 融合，GPS 全精度 MLP embedding 通过 gated residual 在时序聚合后注入。
- 新增 paper-style ablation 配置与报告：1D fusion 对照、移除 camera/LiDAR/radar/GPS、single-frame、mean-pooling temporal、GPS spatial-only/global-only 对照。
- 新增复现实验报告产物规范，明确本地 split、seed、样本数、metric profile、与论文目标 DBA 的差距、GPU/latency/参数量统计和不可比性 caveat。
- 新增 mock/synthetic smoke 与配置/forward/metric 测试，保证真实数据不可用时仍可验证 shape、loss、metadata 和报告路径，但 mock 结果不得被描述为真实 DeepSense6G 结果。
- 不新增旧入口、不恢复 KD/distillation、不提交真实数据、训练输出、cache、checkpoint 或日志。

## Capabilities

### New Capabilities
- `bev-fusion-2604-reproduction`: 定义 arXiv:2604.05668 BEV-Fusion 复现实验的模型、数据协议、配置、指标、ablation、报告和验证契约。

### Modified Capabilities
- 无。该 change 复用现有 `deepsense6g-scene-selection`、`modality-contracts`、`lidar-preprocessing`、`resnet18-image-encoder`、`configurable-multimodal-fusion`、`experiment-workflow` 和 `vision-position-baseline-suite` 能力，不改变它们的既有 requirement。

## Impact

- 代码：预计新增 `src/kd_sensing/models/bev_fusion_2604.py` 或等价窄模块，并在默认组件注册中暴露 `bev_fusion_2604`；必要时新增 BEV query、camera cross-attention、GPS BEV encoder、spatial fusion、temporal aggregation 和 model metadata helper。
- 数据与预处理：复用现有 DeepSense6G dataset、image RGB profile、LiDAR BEV cache、radar RA/DA、GPS feature 和 scene selection；必要时新增 2604 split/manifest helper，但不得绕过 `src/kd_sensing` 包结构。
- 配置：新增 `configs/fusion/experiments/bev_fusion_2604/` 配置族，包括 full model、quick smoke、ablation 和 low-memory 变体。
- 评估与报告：复用现有 top-k/DBA metric，新增 2604 report helper 或配置化输出，记录 S32/S33/S34 DBA、Top-K、macro/overall、论文目标值、差距和可比性限制。
- 测试：新增配置加载、模型 forward、GPS dual-path、BEV shape、ablation config、metric/report metadata 和 CLI/help 回归测试；所有项目相关 Python 验证使用 `conda run -n kd_mm_beam`。
- 文档：更新 `docs/experiment_matrix.md` 或新增实验说明，给出推荐命令、数据/cache 准备、报告位置和本地产物边界。
