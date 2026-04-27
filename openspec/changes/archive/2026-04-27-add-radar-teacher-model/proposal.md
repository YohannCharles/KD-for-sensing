## Why

现有项目已经支持 image-only teacher 和 image+radar fusion teacher/student，但缺少论文表格中用于对照的 radar-only teacher。补齐 RadarTeacher 可以复现原论文的 Radar Top-3/Top-5/ADBA 基线，并为后续多模态蒸馏实验提供可独立训练、评估和复用的雷达教师模型。

## What Changes

- 新增 radar-only teacher 模型能力，输入仅使用 RA/DA 雷达序列，结构遵循论文片段中的 embedding、GRU temporal modeling、MHA prediction、MLP classifier 三段式设计。
- 复用现有 `RadarFeatureExtractor` 或等价的任务特定 CNN embedding，将 RA/DA 拼接后的雷达张量映射到低维特征序列。
- 注册 `radar_teacher` 模型，使其可通过配置构建，并返回与现有 teacher/student 兼容的 `(pred, features, enhanced_seq_out)` 输出。
- 新增 radar-only 训练/评估配置，支持 `experiment.task: radar` 或等价任务路径，能够只读取并前向雷达输入。
- 增加基础构建和前向测试，覆盖 RadarTeacher 的注册、输出形状、Top-K/DBA 评估路径可用性。
- 不引入破坏性变更；现有 image 与 fusion 配置、权重兼容性和训练语义保持不变。

## Capabilities

### New Capabilities
- `radar-teacher-model`: 定义 radar-only teacher 的模型结构、注册名称、配置驱动训练/评估和输出契约。

### Modified Capabilities
- `experiment-workflow`: 扩展配置驱动实验，使训练与评估入口支持 radar-only 任务。
- `component-registry`: 扩展模型注册能力，要求 `radar_teacher` 可按配置构建并保持统一模型输出约定。

## Impact

- 影响模型代码：`src/kd_sensing/models/` 下新增或扩展 radar teacher 实现，并更新模型公共导出。
- 影响批处理与训练/评估入口：`src/kd_sensing/engine/batch.py`、`trainer.py`、`validator.py`、`evaluator.py` 需要识别 radar-only 任务并准备雷达输入。
- 影响配置：新增 `configs/radar/*.yaml`，包含 no-KD radar teacher 训练/评估基线，必要时添加 KD 配置占位但不改变现有 image/fusion 默认配置。
- 影响测试：新增或扩展模型构建、前向形状、配置解析相关测试。
