## Why

当前 canonical 实验配置已经膨胀为大量重复 YAML，尤其 `configs/fusion/` 需要为每个多模态组合和 KD 模式维护一个实体文件。随着 mmWave 加入，组合数继续上升，手写文件容易出现字段漂移、命名不一致和维护成本过高的问题。

## What Changes

- 新增 canonical 配置解析能力：训练、评估和测试可以通过现有路径形态加载虚拟 canonical 配置，即使对应 YAML 文件不存在。
- 对 fusion canonical 路径按固定模态优先级 `image > radar > gps > lidar > mmwave` 解析 slug，并拒绝乱序、重复或未知模态命名。
- 由 loader 根据 `configs/fusion/<slug>_<mode>.yaml` 自动生成完整配置，覆盖 `teacher_no_kd`、`student_no_kd`、`logits_kd` 和 `rkd` 四种模式。
- 保留现有命令和推荐路径语义，用户仍可运行 `python scripts/train.py --config configs/fusion/gps_mmwave_logits_kd.yaml`。
- 清理可生成的重复 YAML 文件，只保留 legacy 兼容入口、少量特殊配置和无法从 canonical 规则推导的配置。
- 更新测试和文档，使 canonical 覆盖验证从“文件必须存在”转为“路径必须可解析且配置语义正确”。

## Capabilities

### New Capabilities
- `canonical-config-resolution`: 定义 loader 如何解析不存在于磁盘上的 canonical 配置路径，并生成与旧实体 YAML 等价的最终配置。

### Modified Capabilities
- `configurable-multimodal-fusion`: 将 fusion canonical 矩阵要求从“必须提供每个实体 YAML 文件”调整为“必须提供每个 canonical 路径的可加载配置语义”。
- `experiment-workflow`: 补充训练、评估、dry-run、final config 和 CLI override 对虚拟 canonical 配置的行为要求。

## Impact

- 影响配置加载器：`src/kd_sensing/config/io.py` 需要在文件读取前或失败后识别 canonical 路径并生成配置。
- 影响配置目录：`configs/fusion/` 中可生成的 canonical YAML 可删除或迁移，legacy 和特殊配置保留。
- 影响测试：`tests/test_student_configs.py`、`tests/test_gps_modality.py`、`tests/test_lidar_modality.py`、`tests/test_mmwave_modality.py` 等路径枚举测试需要改为验证 virtual path 可加载。
- 影响文档：README 和 extension guide 需要说明 canonical path 可以是虚拟配置，命名顺序固定为 `image_radar_gps_lidar_mmwave`。
