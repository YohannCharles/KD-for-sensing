## Why

当前项目已经具备 RGB image encoder、GPS-Rel-Polar 特征、配置驱动训练和统一运行产物记录，但视觉表征仍主要依赖 supervised beam loss 或专用残差/候选重排 workflow。引入 GPS 条件化 JEPA 预训练，可以在不恢复旧 KD/teacher 体系的前提下，用跨场景几何信息约束 image latent prediction，为后续少样本 beam prediction、GPS residual 和 fusion 实验提供更稳的视觉初始化。

## What Changes

- 新增 GPS-conditioned JEPA 预训练能力：使用 context encoder、EMA target encoder、GPS conditioner 和 predictor，在 latent 空间预测 target patch/token 表征。
- 新增 `gps_conditioned_jepa` 自监督 objective，使训练入口可在无需 beam 分类主 loss 的情况下优化 latent prediction loss，并以 `val_jepa_loss` 作为默认早停指标。
- 新增可注册 `model.primary` JEPA 模型和配套 loss/训练扩展，保持 target encoder 是模型内部 EMA 分支，不引入旧 distillation registry、frozen teacher checkpoint 或 KD 配置。
- 新增 image patch/token 采样与 GPS 条件化配置，支持随机 patch mask 和 GPS-angle-biased patch mask 两类初始策略。
- 新增 canonical smoke 配置，用于 DeepSense6G image+GPS JEPA 预训练；输出目录、checkpoint、`final_config.yaml`、TensorBoard 和运行状态仍沿用当前训练 workflow。
- 新增测试覆盖模型构建、forward/loss、EMA 更新、objective metadata、配置加载、训练 smoke 和架构边界。

## Capabilities

### New Capabilities

- `gps-conditioned-jepa-pretraining`: 定义 GPS 条件化 JEPA 预训练的模型、采样、loss、EMA、配置和运行产物契约。

### Modified Capabilities

- `first-class-prediction-tasks`: 将 `gps_conditioned_jepa` 作为受支持的一等训练 objective，明确其自监督 target、loss、metric、history/TensorBoard 和 runtime metadata 语义。
- `experiment-workflow`: 扩展配置驱动训练 workflow，使训练入口能运行 JEPA 自监督预训练，同时保持 supervised/adaptation 入口和 KD 退役边界不变。

## Impact

- 受影响代码：`src/kd_sensing/models/`、`src/kd_sensing/losses/`、`src/kd_sensing/engine/`、`src/kd_sensing/config/`、`configs/`、`tests/` 和相关 README/实验矩阵文档。
- 运行接口：新增配置路径和可注册模型名；不新增旧脚本入口，不改变现有 supervised beam、GPS v2、Top8、BGAM、CSI 或 Raymobtime 默认行为。
- 依赖：继续使用现有 PyTorch/torchvision 依赖，不新增外部训练框架。
- 产物：JEPA checkpoint、日志、TensorBoard、manifest 和 latent 诊断均写入 `outputs/` 或 `logs/`，不进入源码变更。
