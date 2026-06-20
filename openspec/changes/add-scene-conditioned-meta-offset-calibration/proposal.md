## Why

当前主线已经支持 Image+GPS JEPA、BGAM、MMW GPS v2、geometry prior 和 difficulty/reliability 诊断，但跨场景适配仍主要由单一 workflow 或最后层校准表达，难以系统区分几何、视觉域、模态可靠性、目标关联、无线传播残差和 beam-logit 校准各自的贡献。需要一个在现有 `src/kd_sensing`、registry、dataset runtime、target-shot split 和 config recipe 内可复现实验的场景条件化元学习框架，并以当前 visual architecture sweep 中表现最好的 `overlap_k16_s8_stage1` 作为默认 canonical visual/JEPA 基底，用 synthetic/smoke 先跑通张量、训练、评估和防泄漏，再逐步接入 DeepSense6G/MMW 当前数据。

## What Changes

- 新增场景条件化多偏移校准能力：在 shared canonical predictor 之上引入 scene encoder、support-set encoder、hierarchical hypernetwork 和多个小型 offset/adaptation 组件，分别覆盖 geo、image、fusion、align、radio、object 和 beam-logit 残差。
- 默认 canonical predictor 基底改为 `overlap_k16_s8_stage1`：使用 overlap patch tokenizer（kernel 16、stride 8、max tokens 729）和 GPS-query pooling；patch16 mean、GPS-biased、ResNet+GPS、GPS-only 等只作为显式 control、ablation 或 fallback。
- 新增 meta/few-shot adaptation runtime：支持 zero-shot、unlabeled support、labeled K-shot、MAML/FOMAML/ANIL、hypernetwork 初始化和 hypernetwork + gradient adaptation，并复用 target-shot split 与 sensitive field guard。
- 新增 synthetic scenario-hyperbeam 数据路径：不依赖真实 `dataset/` 即可生成可控 scene shift、support/query episode、offset-head attribution 信号和 smoke/sanity 测试。
- 新增 config recipe 和实验矩阵生成：用 base config + overrides 生成用户需求中的 baseline、单头/多头消融、adapter、radio、fusion、object、meta、few-shot、泛化 split、模态缺失和 loss ablation，避免手写 80 个重复实体 YAML。
- 新增评估与诊断输出：记录 canonical logits、offset 分量、scene embedding、offset norm、modality gate、few-shot adaptation curve、per-scene/town/weather metrics 和 config/checkpoint/metrics provenance。
- 不创建独立顶层 `scenario_hyperbeam/` 包，不新增旧式根训练入口，不恢复 KD/HiST/Top8/GPS residual/camera residual/geometry-residual 等退役路线。
- **BREAKING**: 无。新增功能默认 opt-in，现有训练、评估、config、dataset 和模型注册名保持兼容。

## Capabilities

### New Capabilities

- `scene-conditioned-meta-offset-calibration`: 定义场景条件化 canonical predictor、scene/support encoder、hierarchical hypernetwork、多偏移头、meta/few-shot adaptation、synthetic sanity、实验矩阵和报告产物的端到端需求契约。

### Modified Capabilities

- `model-architecture-extension-contract`: 增加该方法的扩展路径分类、whole-model 例外条件、可组合子组件边界和训练策略 metadata 要求。
- `dataset-runtime-contracts`: 增加 synthetic scenario-hyperbeam descriptor、episode/support/query runtime、target-state/object-token/scene-parameter 字段和 target-side sensitive field guard 要求。
- `modality-contracts`: 明确 `target_state`、`object_tokens`、`scene_params`、support metadata 与 canonical sensing modalities 的关系，避免伪模态和 label leakage。
- `canonical-config-resolution`: 增加 scenario meta-offset 的 base recipe、override matrix 和可生成配置边界。
- `first-class-prediction-tasks`: 增加 beam power、angle、LOS/path 等辅助输出/损失作为训练辅助或诊断字段的目标契约，不允许作为测试输入。

## Impact

- 代码区域：`src/kd_sensing/models/` 下新增 scene/meta/offset/adapters/hypernetwork 相关窄模块，必要时新增一个明确 OpenSpec 例外的 `MODELS` 注册入口，并复用或封装 `overlap_k16_s8_stage1` 的 JEPA visual base；`src/kd_sensing/data/` 下新增 synthetic descriptor、episode sampler、support/query collate/helper；`src/kd_sensing/engine/` 下新增 meta-training/adaptation helper、loss/metric/metadata 扩展；`src/kd_sensing/config/` 下新增 recipe/override matrix；`src/kd_sensing/cli/` 只新增包内薄 CLI 或复用现有 train/evaluate。
- 配置与文档：新增最小实体 example config，矩阵由 generator/recipe 生成；README 只保留 quickstart 索引，完整说明放入 docs/OpenSpec 相关文档。
- 测试：新增 dummy dataset、model shape、offset head、hypernetwork、meta episode、loss、config matrix、registry build、ModelOutput adaptation 和 leakage guard focused tests。
- 产物边界：sanity、metrics、plots、checkpoints 和 generated matrix 输出均写入 ignored `outputs/`、`logs/` 或显式本地目录，不提交真实数据、cache、checkpoint 或运行报告。
