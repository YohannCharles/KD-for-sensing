## Why

当前 Scenario 9 数据集在 GPS-only、LiDAR-only、radar-only 等任务中仍会无条件读取 image/radar 字段，导致 CPU、I/O 和 worker 资源被无效消耗；同时固定 run name、固定评估输出目录和不一致 split 让实验结果容易被覆盖或误读。现在需要把训练数据加载和实验产物管理改成按配置驱动、可复现且适合并行实验的行为。

## What Changes

- 让 Scenario 9 dataset 按 `experiment.task` 与 fusion `modalities` 懒加载必要模态，未启用模态不得读取文件或要求列存在。
- 为 DataLoader 增加可配置的 `num_workers`、`persistent_workers`、`pin_memory`、`prefetch_factor` 等运行参数，并在默认配置中降低并行训练时的 CPU/I/O 抖动。
- 为训练与评估输出引入唯一 run id 或等价目录策略，避免固定 `run_name` 覆盖旧实验产物，并保持可通过配置关闭或指定确定性目录。
- 统一默认 split 配置，使不同模态和多模态实验默认使用同一 train/test CSV 进行横向比较。
- 默认启用或推荐 LiDAR BEV cache 复用路径，避免每个 epoch 重复点云转 BEV；cache 仍必须逐样本懒加载，不得在 dataset 初始化时全量读入。
- 修复 `target_beam` 在 `num_pred=1` 时被 `.squeeze()` 降维的风险，保持 batch 合约稳定。
- 增加覆盖这些行为的单元测试和短训练 smoke test，所有 Python 验证命令使用 `conda run -n kd_mm_beam`。

## Capabilities

### New Capabilities
- `modality-aware-data-loading`: 约束 Scenario 9 dataset、batch 准备和 DataLoader 必须按任务/模态选择加载必要字段，并提供性能相关加载参数。

### Modified Capabilities
- `experiment-workflow`: 增加唯一运行目录、评估输出隔离、统一 split 配置和可比较实验输出的要求。
- `lidar-preprocessing`: 增加 LiDAR BEV cache 在训练配置中的复用要求，并明确 cache 必须保持逐样本懒加载。

## Impact

- 主要影响 `src/kd_sensing/data/datasets/scenario9.py`、`src/kd_sensing/engine/batch.py`、`src/kd_sensing/engine/builders.py`、`src/kd_sensing/engine/trainer.py`、`src/kd_sensing/engine/evaluator.py` 和配置默认值。
- 需要更新单模态、fusion 与 LiDAR 相关 YAML 配置，补充可选 DataLoader、输出目录、split 和 cache 字段。
- 需要新增/更新测试，覆盖未启用模态不触发 I/O、LiDAR cache 逐样本复用、输出目录不混杂、`num_pred=1` 标签维度和短训练 smoke test。
- 不引入新的第三方运行依赖；可能调整默认 worker/cache 配置以适配并行实验。
