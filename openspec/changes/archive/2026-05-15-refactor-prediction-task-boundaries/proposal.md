## Why

`beam`、`occlusion`、`position` 和 `multitask` 已经具备一等预测任务的雏形，但 objective、验证路径、dataset target、canonical overlay 和包级导入边界仍分散在编排层。若继续在 `trainer.py`、`validator.py`、`DeepSense6GDataset` 和 `canonical.py` 中叠加新策略，后续新增 objective、模态或诊断会继续扩大核心文件并增加双路径回归风险。

## What Changes

- 将 prediction objective 的 target、loss、默认指标、可用指标、early stopping alias、指标方向和日志字段名收口到 objective 层，训练、验证、评估只消费统一描述。
- 抽取共享 evaluation pass，使普通验证、force-mask subset 验证和 standalone evaluate 复用同一套 batch 准备、forward、loss 计算和指标收集流程。
- 将 DeepSense6G dataset 内部拆成样本索引、模态 loader 和 target provider，优先让 occlusion/position target 从 dataset 主类中迁出，并保持样本字段兼容。
- 将 canonical 配置 overlay 改为 recipe/table 驱动，并按 objective、G2D、CRAF、MARF 等职责拆分 overlay 定义，降低 `canonical.py` 与 `config/io.py` 的手写分支密度。
- 复用现有 `modality_resolution.py` 等中心化能力，移除 evaluator/trainer 中重复的启用模态判断。
- 为 `models/__init__.py` 等包级导出增加延迟导入，避免 `import kd_sensing.models` 拉起所有模型和重依赖模块。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `first-class-prediction-tasks`: objective 层必须成为 prediction task 的 metrics、early stopping alias、指标方向和日志字段名的唯一来源。
- `experiment-workflow`: 训练验证、force-mask subset 验证和 standalone evaluate 必须复用共享 evaluation pass，并保持输出指标语义兼容。
- `modality-aware-data-loading`: DeepSense6G dataset 必须通过 target provider 提供 beam、occlusion、position 和 multitask targets，且目标扩展不再要求修改 dataset 主类的核心取样流程。
- `canonical-config-resolution`: canonical overlay 生成必须由可审查的 recipe/table 组成，并按 objective 与训练扩展职责拆分。
- `project-architecture`: 核心编排层必须变薄，训练策略、目标调度、模态解析、dataset target、canonical overlay 和模型包级导出必须落在窄职责模块中。

## Impact

- 影响代码：`src/kd_sensing/objectives/` 或现有 objective 模块、`src/kd_sensing/engine/trainer.py`、`src/kd_sensing/engine/validator.py`、评估入口、`src/kd_sensing/data/datasets/deepsense6g.py`、`src/kd_sensing/config/canonical.py`、`src/kd_sensing/config/io.py`、`src/kd_sensing/models/__init__.py`。
- 影响测试：需要覆盖 objective metric/early-stopping contract、共享 evaluation pass、force-mask subset 验证、standalone evaluate、DeepSense6G target provider、canonical recipe 输出等价性和轻量导入边界。
- API 兼容：用户配置、训练/评估 CLI、输出指标键和已有 checkpoint 加载语义应保持兼容；若迁移期间保留内部兼容 facade，内部新代码不得依赖它。
