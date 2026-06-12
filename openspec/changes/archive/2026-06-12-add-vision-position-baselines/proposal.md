## Why

当前仓库已有 DeepSense6G/BeamBench 复现、模态感知数据加载和通用 fusion 配置能力，但缺少一组面向“图像序列 + GPS -> 64 beam 分类”的可并排训练、评估和对照的 baseline 套件。将 Camera AE + GPS、ResNet + GPS、Transformer 强融合和 GPS-only 神经 baseline 统一纳入现有 `kd_sensing` 架构，可以让后续模型搜索、论文对照和诊断分析有稳定起点。

## What Changes

- 新增 Vision-Position baseline 套件，覆盖四类 preset：Camera AE + GPS、ResNet/视觉 backbone + GPS、Transformer token fusion、GPS-only LSTM/MLP。
- 复用现有 DeepSense6G dataset、模态启用推导、模型 registry、训练/评估 engine 和 top-k 指标，不新增绕过 `src/kd_sensing` 包结构的旧式训练脚本。
- 为每类 baseline 提供可加载配置或 virtual config recipe，默认面向 5 帧 RGB 序列、GPS 序列和 64-class beam logits。
- 明确 GPS 归一化、image normalization/augmentation、sequence 聚合、checkpoint、TensorBoard/metrics 输出和 top-1/top-3 验收口径。
- 对 Camera AE + GPS Direct 复现实验的现有规格做增量：将其作为 baseline suite 的一个受控 preset，而不以 residual、gated 或 attention 模型替代论文目标行。
- 对 configurable fusion 规格做增量：要求 image+gps preset 能选择 late-concat 或 transformer token fusion primary model，并保持 canonical 数据字段和评估入口一致。
- 不引入破坏性变更；旧入口、旧 KD/distillation 路线和本地产物边界保持现有规则。

## Capabilities

### New Capabilities
- `vision-position-baseline-suite`: 定义 DeepSense6G 图像序列 + GPS beam prediction baseline 套件的模型 preset、配置、训练评估闭环、指标和产物边界。

### Modified Capabilities
- `beambench-baseline-reproduction`: 将 Camera AE + GPS Direct 本地实现纳入 baseline suite preset，并补充与套件配置、指标和报告字段的一致性要求。
- `configurable-multimodal-fusion`: 补充 image+gps fusion preset 对 late-concat ResNet/AE 与 transformer token fusion primary model 的选择和配置语义。

## Impact

- 影响代码：`src/kd_sensing/models/`、`src/kd_sensing/data/` 的既有 batch 字段消费、`src/kd_sensing/engine/` 的模型构建与指标记录、`src/kd_sensing/evaluation/` 或 metrics helper、配置解析和 registry。
- 影响配置：新增或扩展 `configs/fusion/`、`configs/gps/`、`configs/image/` 下的 baseline preset/virtual config；配置必须可由现有 `kd-sensing-train` 与 `kd-sensing-evaluate` 使用。
- 影响测试：增加模型构建/forward、配置加载、模态字段、防泄漏、top-k 指标和 CLI help/smoke 测试。
- 影响文档：在 README 或实验矩阵文档中补充 baseline suite 推荐命令、mock/小样本 smoke 说明和产物位置；不要求提交真实数据、训练输出、cache 或 checkpoint。
