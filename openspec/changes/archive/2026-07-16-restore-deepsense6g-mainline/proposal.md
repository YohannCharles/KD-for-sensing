## Why

上一轮主线收敛将 DeepSense6G 与已退役的蒸馏、CSI 和多目标路线一并移除，但当前研究主线仍需要以 DeepSense6G 与 MMW 两个数据集验证相同的四模态 T2 工作流。现有代码只能加载 MMW，无法在不恢复历史复杂分支的前提下运行 DeepSense6G。

## What Changes

- 新建严格的 DeepSense6G 主线数据集实现，仅支持 Scene31–34、图像/雷达/GPS/LiDAR 四模态与未来波束功率的 64 类硬标签。
- 将共享数据工厂、注册表、配置校验、训练和评估契约扩展为可显式选择 `mmw` 或 `deepsense6g`，保持模型与 batch 接口不变。
- 提供 DeepSense6G T2 配置和针对数据加载、标签、GPS 标准化及双数据集边界的测试与文档。
- 不恢复 CSI、毫米波原始输入、蒸馏、软标签、缓存、跨场景别名、历史兼容入口或 DeepSense6G 专属 CLI。

## Capabilities

### New Capabilities

- `deepsense6g-mainline`: 在共享四模态训练工作流中加载 DeepSense6G Scene31–34，并生成未来波束硬标签。

### Modified Capabilities

- `project-architecture`: 项目保留 MMW 与 DeepSense6G 两个主线数据集，并定义其共享和专属边界。
- `t2-baseline-surface`: T2/baseline 公共运行时可通过显式数据集配置服务两个主线数据集。
- `canonical-config-resolution`: 配置解析和校验接受受支持的 DeepSense6G 数据集标识与场景约束。
- `dataset-loader-behavior`: 数据加载器依据数据集类型构建 MMW 或 DeepSense6G 的显式训练、验证和测试 split。
- `training-evaluation-runtime`: 训练和评估运行时将 DeepSense6G 四模态样本纳入与 MMW 相同的 batch 契约。
- `project-entrypoint-lifecycle`: 保留的训练与评估入口可消费 DeepSense6G 主线配置，不增加旧入口。

## Impact

影响 `src/kd_sensing/data`、数据工厂、组件注册、配置校验、DeepSense6G 配置、相关测试与当前主线文档。无需新增依赖、模型、CLI 或迁移旧数据格式；现有 MMW 专属脚本和 active MMW 实验变更保持 MMW 范围不变。
