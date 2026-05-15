## 1. 配置与目标解析

- [x] 1.1 在配置加载层增加 `experiment.objective` 解析，默认值为 `beam`，并校验合法值为 `beam`、`occlusion`、`position`、`multitask`。
- [x] 1.2 扩展配置校验，确保 `occlusion` objective 启用 dataset 遮挡目标和模型遮挡 head，缺失时给出明确修复提示。
- [x] 1.3 扩展配置校验，确保 `position` objective 启用 dataset 位置目标和模型位置 head，缺失时给出明确修复提示。
- [x] 1.4 增加 objective-aware early stopping 默认值：beam 使用 `val_adba/max`，occlusion 使用 `val_occlusion_blocked_f1/max`，position 使用 `val_position_rmse/min`，multitask 使用可配置默认主指标。
- [x] 1.5 保持旧配置兼容，验证未设置 `experiment.objective` 的现有配置仍解析为 beam-only。

## 2. Objective 核心抽象

- [x] 2.1 新增 prediction objective helper 或 registry，提供 objective 解析、所需 targets、所需 outputs、默认 metrics 和主 loss 名称。
- [x] 2.2 实现 objective-aware target 准备，将现有 beam labels、`occlusion_label/valid`、`position_target/valid` 统一到一个 target bundle。
- [x] 2.3 实现 objective-aware loss helper：beam 使用现有 task criterion/KD 基础 loss，occlusion 使用 masked BCEWithLogits，position 使用 masked MSE/SmoothL1，multitask 使用配置权重加权组合。
- [x] 2.4 迁移或包装现有 `compute_auxiliary_multitask_loss()`，使旧 auxiliary 配置和新 objective 配置不会产生重复 loss。
- [x] 2.5 为 objective loss diagnostics 定义稳定 key，包括主 loss、beam loss、occlusion loss、position loss 和 multitask total。

## 3. 训练、验证与评估流程

- [x] 3.1 改造 `trainer.py`，让总 loss 和 `train_task_loss` 由 objective helper 决定，occlusion/position 单任务不再依赖 `loss.alpha: 0.0`。
- [x] 3.2 改造 `validator.py`，按 objective 聚合主 metrics，并继续输出可用的诊断性 beam、occlusion 和 position metrics。
- [x] 3.3 改造 `evaluator.py`，让 standalone evaluate 能读取 objective metadata、复用遮挡阈值和位置 scaler，并输出 objective-aware metrics。
- [x] 3.4 扩展 TensorBoard、epoch log、`training_outputs.npz` 和 final config runtime metadata，记录 objective、主 metric、metric mode、启用 targets 和启用 heads。
- [x] 3.5 扩展 checkpoint registry metadata，归档 best checkpoint 时记录 objective-aware best metric 和分任务指标。

## 4. Fusion 配置矩阵

- [x] 4.1 扩展 canonical/virtual fusion config helper，支持 `<slug>_<objective>_no_kd.yaml` 命名和解析。
- [x] 4.2 提供五模态 recommended objective 配置入口：beam、occlusion、position 和 multitask，并保持现有 `all_modalities_no_kd.yaml` 作为 beam 兼容入口。
- [x] 4.3 提供 strong-only 和 weak-only objective 配置或 virtual config 支持，用于多任务模态失衡对照。
- [x] 4.4 确保 objective 配置复用同一 split、horizon、模态顺序、CLS-token fusion backbone、遮挡阈值和位置 target scaler 语义。
- [x] 4.5 更新 README 或实验文档，写明一等任务运行命令和推荐实验矩阵。

## 5. 测试与验证

- [x] 5.1 增加配置解析和校验测试，覆盖默认 beam、未知 objective、缺失 occlusion head、缺失 position head 和旧配置兼容。
- [x] 5.2 增加 objective loss 单元测试，覆盖 masked occlusion BCE、masked position MSE/SmoothL1、multitask weighted sum 和旧 auxiliary 兼容。
- [x] 5.3 增加训练 smoke test，分别运行 beam、occlusion、position 和 multitask 小样本路径，确认 occlusion/position 不需要 `loss.alpha: 0.0`。
- [x] 5.4 增加 validator/evaluator 测试，覆盖 objective-aware metrics、early stopping metric alias、遮挡阈值 artifact 复用和位置 scaler artifact 复用。
- [x] 5.5 增加 canonical config 测试，覆盖五模态、strong-only、weak-only 的 objective virtual config 解析和模型构建。
- [x] 5.6 运行回归测试：`conda run -n kd_mm_beam pytest tests/test_student_configs.py tests/test_cls_token_transformer_fusion.py tests/test_mmwave_modality.py tests/test_training_io_workflow.py`。
- [x] 5.7 运行 OpenSpec 校验：`openspec status --change add-first-class-prediction-tasks`，确认变更 apply-ready。
