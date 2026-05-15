## Context

当前实现已经有正确的边界雏形：`engine/prediction_objectives.py` 负责 target/loss/default metric，`engine/runtime.py` 负责 batch 和 model output adapter，CRAF/G2D/MARF 已经通过训练扩展接入，`engine/modality_resolution.py` 能统一解析启用模态。问题是这些边界没有被训练、验证、评估、dataset 和 config 生成路径完全复用，导致 `trainer.py`、`validator.py`、`DeepSense6GDataset` 和 `canonical.py` 继续承担多个职责。

这次变更优先处理架构收口，不改变用户配置、CLI、checkpoint 和现有输出指标的兼容语义。所有训练验证和测试命令继续通过 `conda run -n kd_mm_beam ...` 执行。

## Goals / Non-Goals

**Goals:**

- 让 prediction objective 成为 target、loss、metrics、early stopping、日志字段和 runtime metadata 的单一来源。
- 让普通验证、force-mask subset 验证和 standalone evaluate 复用同一个 evaluation pass。
- 将 DeepSense6G dataset 主类缩小为样本索引协调器，模态读取和 target 构造通过窄组件实现。
- 将 canonical overlay 生成迁移到 recipe/table，并按 objective、G2D、CRAF、MARF 等职责拆分。
- 收敛重复的启用模态判断，并为 `kd_sensing.models` 做延迟导出。

**Non-Goals:**

- 不新增预测目标、模态、模型结构或新的训练算法。
- 不改变现有 canonical config 路径、默认超参或已有输出指标键。
- 不重写整个 trainer；本次只迁出 objective/evaluation/config/dataset 的高风险重复逻辑。
- 不删除历史兼容的公开导入路径，除非已有 spec 明确要求删除。

## Decisions

### 1. 扩展现有 objective 模块，而不是新增平行 registry

`engine/prediction_objectives.py` 已经包含 objective 名称、target、loss 和默认指标。继续扩展该模块的 dataclass，让它输出 `available_metrics`、`early_stopping_aliases`、`metric_modes`、`history_fields`、`tensorboard_scalars` 和 `runtime_metadata`。

替代方案是新增一个 metrics registry，但会让 objective 和 metrics 再次分裂。当前问题不是缺少 registry，而是 objective metadata 没有覆盖全链路。

### 2. 新增 `engine/evaluation_pass.py`，现有入口变成薄 wrapper

新增共享 pass，输入为 model、dataloader、cfg、criterion、device 和可选 `force_mask`，输出结构化结果，包含 loss bundle、beam metrics、auxiliary metrics、available metrics、degradation diagnostics 和收集到的中间张量摘要。`validator.validate()`、`_validate_with_force_mask()` 和 standalone evaluate 调用该 pass，再负责写文件或包装旧返回格式。

替代方案是在 `validator.py` 内抽私有 helper。考虑到 evaluate 也需要复用，独立模块能避免 `validator.py` 成为新的聚合文件。

### 3. Dataset 先组件化 target，再组件化模态 loader

DeepSense6G dataset 保持公开类名和返回样本字段不变，内部引入：

- 样本索引组件：封装 CSV sample、portion、scene、metadata。
- 模态 loader 组件：按启用模态读取 image/radar/GPS/LiDAR/mmWave。
- target provider 组件：提供 beam、occlusion、position 和 multitask 所需字段。

优先迁出 occlusion/position target provider，因为它们和新增 objective 的耦合最大。beam label cache 和模态 loader 可以分阶段迁移，但新 target 不允许再直接塞进 dataset 主类。

### 4. `canonical.py` 保留入口，recipe 定义拆到窄模块

`build_virtual_config()` 和 `build_virtual_fusion_config()` 保持对外入口不变，内部改为查询 `config/canonical_recipes/` 下的 recipe/table。基础 fusion、objective overlay、G2D、CRAF、MARF 分文件定义，通用 merge/validation helper 单独放置。

替代方案是直接拆成 `config/canonical/` 包，但这会和现有 `canonical.py` 模块路径冲突，迁移成本更高。

### 5. 复用 `modality_resolution.py` 作为唯一启用模态来源

训练、验证、评估和诊断中关于 GPS/LiDAR/mmWave 是否启用的判断统一调用 `resolve_enabled_modalities()` 或 `config_uses_*()`。删除 evaluator/validator 中重复 `_evaluation_uses_*`、`_cfg_uses_*` 风格 helper。

### 6. 包级导出采用 lazy mapping

`models/__init__.py` 改为声明 `__all__` 和符号到模块的映射，通过 `__getattr__` 按需导入。保留已有 removed alias 错误信息。这样 `import kd_sensing.models` 不会立刻导入 fusion、LiDAR、mmWave 等重依赖模块。

## Risks / Trade-offs

- 旧日志字段遗漏 → 在 objective metadata 中保留旧字段名，添加训练历史和 TensorBoard 回归测试。
- evaluation pass 行为偏移 → 先用旧 `validate()` 和 force-mask 路径的等价测试锁住 beam/occlusion/position/multitask 指标。
- Dataset 组件化引入缓存或 scaler 生命周期 bug → 分阶段迁移，先让 provider 复用现有 helper 和 cache，再拆 loader。
- Recipe 化 canonical config 改变生成结果 → 用 fixture 比较关键 canonical 路径的生成 dict，除字段顺序外必须等价。
- Lazy import 破坏旧 `from kd_sensing.models import X` → 对 `__all__` 中所有旧公开符号做导入测试。

## Migration Plan

1. 扩展 objective spec，迁移 trainer/validator 中的 metric alias、available metrics 和日志字段定义。
2. 引入 evaluation pass，并让 `validate()` 与 force-mask subset 验证走同一实现。
3. 将 standalone evaluate 接到 evaluation pass，统一输出保存路径和指标来源。
4. 为 DeepSense6G 引入 target provider，先迁出 occlusion/position，再迁出模态 loader。
5. 建立 canonical recipe/table，并逐个把 objective、G2D、CRAF、MARF overlay 从 `canonical.py` 迁入。
6. 替换重复启用模态 helper，完成 `models/__init__.py` lazy export。
7. 运行目标单测、架构导入测试和最终 `conda run -n kd_mm_beam pytest -q`。

## Open Questions

- 是否在本次变更内完全迁出 beam label cache，还是保留在 dataset 主类等待下一轮数据层整理。
- standalone evaluate 当前有哪些外部脚本依赖旧报告结构，需要在实现时通过测试或 grep 再确认。
