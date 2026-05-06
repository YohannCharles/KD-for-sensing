## Why

当前项目已经完成 `src/kd_sensing` 包化和配置驱动训练，但轻量配置/路径模块会被数据集重依赖间接拖住，模态规则分散在 `config`、`engine`、`data`、`models` 和 `diagnostics` 多处，几个大文件也承担了过多横切职责。现在继续增加模态、编码器和诊断能力会放大这些耦合，因此需要先把架构边界收紧。

## What Changes

- 建立统一的模态契约，集中描述合法模态、顺序、dataset flag、batch key、model input key、默认字段和归一化/cache 支持。
- 调整轻量导入边界，使 `kd_sensing.config`、`kd_sensing.utils.paths`、场景元数据和配置校验不因导入 dataset/model 组件而要求 pandas、scipy、skimage、matplotlib 等训练/数据依赖可用。
- 将 `engine/builders.py` 中的模态解析、cache policy、normalization artifact、run metadata、optimizer/scheduler/device 构建拆成职责明确的模块，并保持现有公开训练/评估入口兼容。
- 将 `data/transforms.py` 按 image、radar、lidar、gps、mmwave 和通用 cache/atomic IO 边界拆分，保留兼容导入路径或明确迁移入口。
- 将 `diagnostics/modality_visualization.py` 拆分为配置解析、数据集准备、样本选择、统计、渲染和写出模块，保持 CLI 行为和产物格式不变。
- 更新扩展文档和 OpenSpec spec purpose，说明新模态、新转换、新诊断能力应该落在哪些模块。

## Capabilities

### New Capabilities

- `modality-contracts`: 定义模态元数据、顺序、输入输出命名和 cache/normalization 能力的单一来源。

### Modified Capabilities

- `project-architecture`: 增加轻量导入边界、大模块职责拆分和源码/实验产物边界要求。
- `component-registry`: 增加默认组件延迟导入和注册发现边界要求，避免注册机制破坏轻量模块导入。

## Impact

- 主要影响 `src/kd_sensing/config/`、`src/kd_sensing/utils/`、`src/kd_sensing/engine/`、`src/kd_sensing/data/`、`src/kd_sensing/diagnostics/`、`src/kd_sensing/models/fusion/` 和 `src/kd_sensing/registries.py`。
- 训练、评估、预处理、诊断 CLI 参数和配置文件语义应保持兼容；重构期间允许新增内部模块和兼容 re-export。
- 测试需要覆盖轻量导入、配置加载、模态解析、数据转换兼容导入、训练/评估 smoke 路径和诊断产物稳定性。
- 不引入新的运行时功能依赖；目标是降低现有依赖的导入传播范围。
