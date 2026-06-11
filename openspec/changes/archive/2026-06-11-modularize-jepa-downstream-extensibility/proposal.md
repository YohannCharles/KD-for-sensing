## Why

`fair_gps_biased` 已经成为当前 Image+GPS+JEPA 下游主线，`add-gps-query-jepa-pooling` 也证明了通过 `modular_sequence` 的条件 encoder 机制插入 GPS-aware pooling 是可行的。继续优化该路线时，如果仍把 pooler、adapter、条件依赖、参数组学习率和运行 metadata 分散在单个 encoder、YAML 片段和手写 metadata 解析中，后续每插一个模块都会增加隐式契约和实验对比漂移风险。

本 change 目标是在不改变训练数值语义、不替换 `fair_gps_biased` baseline、不重训 JEPA Stage 1 的前提下，为 JEPA 下游 supervised fusion 建立更清晰的可插拔边界，方便后续快速比较 mean pooling、GPS-query pooling、K-token pooling、轻量 adapter、冻结策略和差异化学习率。

## What Changes

- 将 `jepa_context_image` 下游复用拆成更明确的“checkpoint 加载 + token encoder + pooler/adapter”结构，保留默认 mean pooling 和现有 GPS-query pooling 行为。
- 新增或正式化 JEPA downstream pooler/adapter 扩展点，使后续 pooler 可通过配置注册和构建，而不是继续向 `JepaContextImageEncoder` 增加专用分支。
- 正式化模块化模型中的 conditioned encoder 契约：encoder 可声明依赖哪些模态、使用 raw/encoded/projected 哪类条件特征、期望的 forward kwarg 和 shape 校验语义。
- 为 `fair_gps_biased` 系列实验提供可配置 optimizer 参数组能力，使 JEPA context encoder、GPS encoder、pooler/adapter、fusion core 和 head 能使用不同学习率、weight decay 或冻结策略。
- 将 JEPA downstream 运行 metadata 从“手动解析配置”推进为模型/子模块可声明的训练策略 metadata，并继续写入 `final_config.yaml` 或 runtime metadata。
- 保留现有 `fair_gps_biased` mean-pooling 配置作为主 baseline；新增结构只服务派生配置和后续模块插拔，不恢复 KD、HiST/Hist、Top8 selector、GPS residual、camera residual 或旧 fusion 入口。

## Capabilities

### New Capabilities

- `jepa-downstream-extensibility`: 定义 JEPA context encoder 下游 pooler/adapter、conditioned encoder、参数组优化和 metadata 的可扩展契约。

### Modified Capabilities

- `gps-conditioned-jepa-pretraining`: 扩展 JEPA context encoder 下游复用契约，使 `jepa_context_image` 支持通过可插拔 pooler/adapter 复用 context tokens，同时保持 Stage 1 预训练和 checkpoint schema 不变。
- `modular-sequence-model`: 扩展模块化序列模型契约，正式定义 encoder 条件依赖、条件特征来源、依赖排序和错误诊断。
- `component-registry`: 扩展内置组件注册契约，使 JEPA downstream pooler/adapter 可通过配置构建，同时保持 registry 轻量导入边界。
- `project-architecture`: 扩展训练 builder 边界，要求 optimizer 参数组和 runtime metadata 收集仍位于窄模块，不回流到训练主循环或兼容 facade。

## Impact

- 影响代码：预计涉及 `src/kd_sensing/models/jepa.py` 的职责拆分或新建 JEPA downstream pooler/adapter 模块、`src/kd_sensing/models/modular.py` 的 conditioned encoder 契约整理、`src/kd_sensing/engine/optim.py` 的参数组构建、`src/kd_sensing/engine/run_metadata.py` 的 metadata 收集，以及必要的 registry/default component import。
- 影响配置：新增或调整 `configs/fusion/experiments/jepa_image_gps/` 下派生配置，用于比较 pooler/adapter、冻结策略和参数组；现有 `fair_gps_biased` baseline 配置不改名、不删除。
- 影响测试：新增 focused tests 覆盖 pooler registry、conditioned encoder dependency、optimizer 参数组、runtime metadata、配置加载和 synthetic forward smoke。
- 影响文档/OpenSpec：更新 JEPA Image+GPS 实验 README 和相关 specs，说明 baseline 与派生实验的边界。
- 不影响本地数据和产物：不删除或迁移 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 或 `All_models/` 历史权重。
