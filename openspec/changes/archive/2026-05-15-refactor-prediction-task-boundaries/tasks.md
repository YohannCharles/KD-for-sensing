## 1. Objective Metadata 收口

- [x] 1.1 扩展 `engine/prediction_objectives.py` 的 objective spec，加入 available metrics、metric aliases、metric modes、history fields、TensorBoard scalar fields 和 runtime metadata。
- [x] 1.2 将 trainer 中的 early stopping alias、metric mode 和 objective 日志字段迁移到 objective metadata，并保持旧公开字段名兼容。
- [x] 1.3 将 validator 中的 objective auxiliary metric 选择和 available metrics 生成迁移到 objective metadata 或 objective metrics helper。
- [x] 1.4 添加 objective metadata 单元测试，覆盖 beam、occlusion、position、multitask 的默认指标、方向、aliases、available metrics 和日志字段。

## 2. 共享 Evaluation Pass

- [x] 2.1 新增 `engine/evaluation_pass.py`，实现共享 batch 准备、forward、objective loss、指标收集和结果结构。
- [x] 2.2 将 `validator.validate()` 改为调用共享 evaluation pass，并保持现有 metrics 返回键和 `metrics.json` 写出兼容。
- [x] 2.3 将 force-mask subset 验证改为通过共享 evaluation pass 的 mask 选项执行，删除重复的 forward/loss/collect 代码。
- [x] 2.4 将 standalone evaluate 入口接入共享 evaluation pass，确保评估报告包含 objective metadata、available metrics 和启用模态。
- [x] 2.5 使用 `conda run -n kd_mm_beam pytest -q` 运行或补充 evaluation pass 相关测试，覆盖普通验证和 force-mask 验证等价性。

## 3. DeepSense6G Dataset 组件化

- [x] 3.1 新增 DeepSense6G target provider 模块，封装 beam、occlusion、position 和 multitask target 字段生成。
- [x] 3.2 将 occlusion target 的统计、cache 和字段生成从 `DeepSense6GDataset` 主类迁入 provider，并保持样本字段 shape/dtype 不变。
- [x] 3.3 将 position target 的 source、normalization/scaler 和字段生成从 `DeepSense6GDataset` 主类迁入 provider，并保持现有配置语义不变。
- [x] 3.4 新增模态 loader 组件边界，先让 dataset 主类通过 loader 调用 GPS、LiDAR、mmWave 等已有 transform helper。
- [x] 3.5 添加 dataset provider/loader 测试，验证未启用 target 或模态时不会读取相关资源，并通过 `conda run -n kd_mm_beam pytest -q <test-path>` 运行目标测试。

## 4. Canonical Recipe 化

- [x] 4.1 新增 `config/canonical_recipes/`，放置通用 merge/validation helper 和 recipe registry。
- [x] 4.2 将基础 fusion config recipe 从 `canonical.py` 迁入 recipe 模块，`build_virtual_config()` 保持公开入口不变。
- [x] 4.3 将 objective fusion overlay recipe 拆出，覆盖 occlusion、position、multitask 的 dataset target、head、loss 和 early stopping 默认值。
- [x] 4.4 将 G2D、CRAF、MARF advanced overlay recipe 分文件迁出，并保持未知 overlay 的可诊断错误。
- [x] 4.5 添加 canonical 输出等价测试，使用 `conda run -n kd_mm_beam pytest -q <test-path>` 验证关键 virtual config 的核心字段不变。

## 5. 架构边界和 Lazy Import

- [x] 5.1 替换 trainer、validator、evaluator 和诊断路径中的重复启用模态 helper，统一调用 `engine.modality_resolution`。
- [x] 5.2 将 `models/__init__.py` 改为 lazy export mapping，保留 `__all__` 和 removed alias 错误兼容。
- [x] 5.3 添加轻量导入测试，验证 `import kd_sensing.models` 不 eager import 模型实现模块，同时 `from kd_sensing.models import <公开符号>` 仍可用。
- [x] 5.4 添加架构引用测试或静态扫描，防止新内部代码重新引用重复模态 helper、trainer 内 objective alias 表或 validator 双验证路径。

## 6. 回归验收

- [x] 6.1 使用 `conda run -n kd_mm_beam pytest -q` 运行全量测试并修复回归。
- [x] 6.2 使用 `conda run -n kd_mm_beam python -m pytest -q <smoke-test-path>` 或现有 smoke 命令验证 beam、occlusion、position、multitask 的短路径。
- [x] 6.3 检查训练输出 payload、final config、metrics/report 和 checkpoint metadata，确认 primary metric、objective metadata、enabled targets/heads 和旧指标字段兼容。
- [x] 6.4 更新相关扩展文档或 README 中关于 objective、evaluation pass、DeepSense6G target provider、canonical recipe 和 lazy import 的开发说明。
