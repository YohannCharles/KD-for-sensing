## Why

Raymobtime s008 已不再作为当前项目目标维护，但它仍占据数据集、预处理、模型、配置、测试、文档和 OpenSpec 契约，持续扩大维护面并干扰 DeepSense6G、MMW、CSI、viewer 等仍保留工作流的边界。现在应一次性退役并删除 Raymobtime s008 相关代码、配置和本地数据契约，让仓库的支持面回到仍需要维护的数据集家族与实验流程。

## What Changes

- **BREAKING** 删除 `raymobtime_s008` dataset type、Raymobtime s008 预处理器、Raymobtime s008 专用模型、配置规则、配置文件、文档入口、focused tests 和默认注册路径。
- **BREAKING** 删除 `dataset/Raymobtime/s008` 及 Raymobtime s008 专属 cache、审计、训练、评估、日志和 checkpoint 运行产物的支持契约；真实本地数据删除必须仅限 Raymobtime s008 路径，并先生成可审计清单。
- 更新 README、实验矩阵、研究笔记和项目表面清单，移除 Raymobtime s008 作为当前支持 workflow、推荐实验或健康检查的一部分。
- 更新 registry、dataset layout descriptor、runtime metadata、模态契约、配置验证和迁移提示，确保旧 `raymobtime_s008` 配置快速失败并给出已退役说明。
- 删除或替换依赖 Raymobtime s008 的测试、示例配置和 egg-info 源列表，保留架构、CLI、DeepSense6G/MMW/CSI/viewer 等非 Raymobtime 工作流可用。

## Capabilities

### New Capabilities

- `raymobtime-s008-retirement`: 约束 Raymobtime s008 被退役后的用户可见行为、清理边界、错误提示和验证要求。

### Modified Capabilities

- `raymobtime-s008-selection`: 将现有 Raymobtime s008 selection 能力改为退役状态，不再要求审计、cache、dataset、模型、训练或评估入口存在。
- `dataset-directory-layout`: 删除 Raymobtime s008 默认数据目录与本地产物契约，明确退役清理只能作用于 Raymobtime s008 路径。
- `dataset-runtime-contracts`: 删除 Raymobtime s008 作为当前保留 dataset descriptor/runtime metadata 的要求。
- `experiment-workflow`: 从当前推荐实验、配置驱动 workflow 和健康检查中移除 Raymobtime s008。
- `project-architecture`: 更新源码与实验产物边界，明确用户要求退役数据集时允许按 manifest 删除对应本地数据与产物，但不得影响其它数据集和源码边界。
- `modality-contracts`: 删除 Raymobtime 3D occupancy grid profile 作为当前保留模态契约的要求。

## Impact

- 受影响源码：`src/kd_sensing/data/datasets/`、`src/kd_sensing/preprocessing/`、`src/kd_sensing/models/`、`src/kd_sensing/config/`、`src/kd_sensing/engine/run_metadata.py`、`src/kd_sensing/modalities.py`、`src/kd_sensing/registries.py`、dataset layout/descriptor 相关模块和包级 lazy exports。
- 受影响配置与文档：`configs/raymobtime/`、`configs/preprocess/raymobtime_s008_*.yaml`、`docs/Raymobtime_s008_selection.md`、README、实验矩阵、研究笔记和项目表面清单。
- 受影响测试：Raymobtime s008 focused tests、配置测试、评估通过性测试、架构边界测试、注册/导入 smoke 和文档/配置引用扫描。
- 受影响本地数据与产物：`dataset/Raymobtime/s008`、`outputs/raymobtime_s008`、相关 logs/cache/checkpoint/diagnostic 产物；清理需要 manifest，不得删除非 Raymobtime 数据、`All_models/` 已跟踪权重、OpenSpec artifacts 或其它活跃实验产物。
