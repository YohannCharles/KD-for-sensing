## Context

AI 调研指出当前重构版本存在 5 类影响复现的问题：默认 GRU 层数与上游代码/随附权重不一致，teacher 与评估权重加载使用 `strict=False` 静默掩盖 mismatch，`resume` 配置没有真正恢复 checkpoint，canonical 配置中的 batch/lr/seed/KD 超参数与 `All_models/params_*.txt` 漂移，以及部分尺寸配置被模型结构硬编码限制。

本变更的目标不是恢复旧脚本，而是在现有配置驱动入口内把上游原代码实际覆盖的 image-only 与 image+radar 路径变成可复现实验配置。radar-only、GPS-only 和 LiDAR-only 是本项目新增单模态入口，没有上游原代码基准；因此它们不声明为原论文结果复现，但其共享训练参数和模型时序参数必须与 image 单模态保持一致，作为单模态配置矩阵的统一默认。

## Goals / Non-Goals

**Goals:**

- 让 `configs/image/*.yaml` 的模型层数、训练超参数、KD 参数与上游 `train_image.py` 及 `All_models/params_Image*.txt` 对齐。
- 让 `configs/radar/*.yaml`、`configs/gps/*.yaml` 和 `configs/lidar/*.yaml` 在共享字段上与对应的 image 单模态配置保持一致。
- 让 image+radar fusion 配置的模型层数、训练超参数、KD 参数与上游 `train_both.py` 及 `All_models/params_Both*.txt` 对齐。
- 让随附 `All_models/*.pth` 在对应配置下严格加载；结构不匹配时给出 missing/unexpected keys，而不是静默随机初始化。
- 恢复 `training.resume` 的实际作用，能从 checkpoint 恢复模型、optimizer、scheduler、epoch 和 best loss。
- 对 image `224x224` 与 radar `128x64` 的原代码尺寸假设做显式校验和文档说明。
- 更新测试和文档，使二层 GRU 不再被描述为所有默认配置的统一事实。

**Non-Goals:**

- 不重新引入上游旧训练脚本作为正式入口。
- 不保证在不同硬件、PyTorch 版本或随机性条件下逐 bit 复现指标。
- 不把 GPS、LiDAR、radar-only 扩展配置声明为上游原论文结果；这些路径使用 image 单模态参数作为本项目新增单模态的统一默认。
- 不在本变更中重构 image/fusion teacher 的 FC 层以支持任意 image size，也不重写 radar extractor 以支持任意 RA/DA 分辨率。

## Decisions

1. **以随附 params 文件和上游脚本作为原始来源。**  
   image 单模态优先使用 `All_models/params_Image*.txt` 中记录的参数；缺失字段使用上游 `train_image.py` 默认值补齐。image+radar fusion 优先使用 `All_models/params_Both*.txt` 中记录的参数；缺失字段使用上游 `train_both.py` 默认值补齐。这样能覆盖已发布权重的真实训练参数，同时避免只按 argparse 默认值覆盖 KD/no-KD 差异。

2. **新增单模态继承 image 单模态参数。**  
   radar-only、GPS-only 和 LiDAR-only 没有上游 `params_*.txt`，因此它们不单独创造一套参数。实现时按文件角色继承 image 单模态：`teacher_no_kd` 对齐 `configs/image/teacher_no_kd.yaml`，`student_no_kd` 对齐 `configs/image/student_no_kd.yaml`，`logits_kd` 对齐 `configs/image/logits_kd.yaml`，`rkd` 对齐 `configs/image/rkd.yaml`；只保留模态必要差异，例如 dataset 字段、模型注册名、输入通道、run name 和 teacher checkpoint 名称。

3. **直接调整现有单模态与 image+radar canonical/legacy 配置。**  
   用户诉求是让当前项目参数与原代码/image 单模态基准一致，因此 `configs/image/*.yaml`、`configs/radar/*.yaml`、`configs/gps/*.yaml`、`configs/lidar/*.yaml`、`configs/fusion/no_kd.yaml`、`configs/fusion/logits_kd.yaml`、`configs/fusion/rkd.yaml` 和 `configs/fusion/image_radar_*.yaml` 应变成目标兼容配置，而不是另建一套并保留当前二层 GRU 作为默认。需要保留多模态扩展矩阵时，非 image+radar fusion 组合继续作为扩展配置，但不影响单模态一致性。

4. **GRU 层数按单模态和 fusion 角色配置。**  
   所有单模态 teacher/student 均使用 `gru_params: [64, 64, 1]`。image+radar fusion teacher 使用 `[64, 64, 2]`；image+radar fusion student 使用 `[64, 64, 1]`。no-KD student 配置中未使用的 teacher 字段仍保持对应 teacher 结构，便于后续切换 KD 或加载 teacher。

5. **权重加载默认严格，非严格加载必须显式选择。**  
   统一 checkpoint 加载工具应封装 `load_state_dict`，默认 `strict=True`。当用户显式配置非严格加载时，系统仍必须记录 missing/unexpected keys；评估与 KD teacher 加载默认不允许静默 mismatch。

6. **`training.resume` 支持 bool 和路径两种语义。**  
   `resume: true` 从当前 `output.run_name/checkpoints/last.pth` 恢复；`resume: <path>` 从显式路径恢复；`start_epoch` 仅在 checkpoint 不含 epoch 时作为兜底。恢复时必须加载 optimizer 和 scheduler，且 best loss 使用 checkpoint 中保存的值继续 early stopping。

7. **尺寸参数先校验，不扩展。**  
   目标兼容路径固定 image `224x224`、radar `128x64`。由于 motion mask、image/fusion teacher FC 输入和 radar extractor 都依赖这些尺寸，本变更先增加校验和文档限制；真正支持动态尺寸应作为独立架构变更。

## Risks / Trade-offs

- **现有二层 GRU 测试会失败** -> 将测试按原代码兼容矩阵改写，并保留扩展配置的独立断言。
- **严格加载会暴露历史不兼容权重** -> 错误信息列出缺失/多余 key，并在文档中说明如何选择匹配配置或显式非严格加载。
- **把当前 canonical 配置改为单模态一致性参数会影响已有扩展实验基线** -> 文档区分“原代码/image 单模态一致性配置”和“多模态扩展配置”，并在 README 中提示。
- **`resume: true` 依赖稳定 run_name** -> 若未设置 `output.run_name`，系统应给出明确错误或要求使用显式 checkpoint 路径，避免从新的 timestamp 目录恢复失败。
