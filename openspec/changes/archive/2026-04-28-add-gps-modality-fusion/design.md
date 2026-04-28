## Context

当前项目的模态结构已经形成固定风格：image 使用 `ImageFeatureExtractor`、`ImageModalityNet`、`ImageStudentModalityNet`，radar 使用 `RadarFeatureExtractor`、`RadarTeacherNet`、`RadarStudentNet`，fusion teacher/student 都返回 `(pred, features, output_features)` 并复用统一训练、验证、评估和 KD 流程。

Scenario 9 原始 CSV 中包含 `unit1_loc`、`unit2_loc`、`unit2_loc_cal` 等 GPS 路径，但当前序列 CSV 只保留 camera、radar、beam 和 `seq_index`。因此 GPS 变更横跨预处理、dataset、batch 准备、模型、fusion 和配置。

这次六组 GPS no-KD 对比显示：`motion` 和 `motion_smooth` 训练精度较高但验证指标明显变差，属于高维短序列特征带来的过拟合或噪声放大；`raw`、`utm`、`relative`、`relative_polar` 属于同一性能梯队。`relative_polar` 在 DBA 上最高，图中约为 0.7557，且 Top-5 曲线最靠前；Top-1/Top-3 上 `relative` 和 `utm` 略高，但差距较小。综合论文叙事、几何可解释性、配置复杂度和 DBA 指标，本 change 只保留 GPS-Rel-Polar。

## Goals / Non-Goals

**Goals:**
- 将 GPS 数据纳入 Scenario 9 序列 CSV、dataset batch 和训练/验证/评估输入准备。
- 只交付 GPS-Rel-Polar 特征：`[dist, sin_theta, cos_theta]`，其中距离和角度来自 UE-BS 相对 UTM 坐标。
- 新增 GPS-only teacher/student 架构，命名和职责对齐现有 image/radar 模态。
- 让 fusion teacher 和 fusion student 通过配置选择 `image`、`radar`、`gps` 的任意非空组合。
- 保持旧 image-only、radar-only、image+radar 配置兼容。
- 使用训练集统计量 fit scaler，并复用于 test/val，避免 GPS 归一化数据泄漏。

**Non-Goals:**
- 不引入新的 KD 算法或改变 logits KD/RKD 损失定义。
- 不重写 image/radar teacher/student 主干。
- 不强制所有 fusion 配置都启用 GPS。
- 不把 raw、UTM、relative、motion 或 motion-smooth 作为本 change 的公开特征模式、配置入口或测试目标。
- 不声称 GPS-Rel-Polar 在 Top-1/Top-3/loss 上绝对优于所有候选；该选择是基于 DBA、Top-5、可解释性和复杂度的折中。

## Decisions

1. GPS 序列列在预处理阶段生成，而不是在 dataset 中回查原始 CSV。

   `generate_sequence_data` 将在保留 camera/radar/beam 的同时新增 `gps1..gpsN` 和 `bs_gps1..bs_gpsN` 或等价列，默认 UE GPS 优先使用 `unit2_loc_cal`，BS GPS 使用 `unit1_loc`。这样 dataset 只读取序列 CSV，不需要每个 batch 重新按 `seq_index` 和帧位置查原始 CSV。替代方案是在 `Scenario9Dataset.__getitem__` 中根据 `seq_index` 回查原 CSV，但这会增加状态耦合并让训练时 I/O 更难预测。

2. GPS 特征构造收敛为 GPS-Rel-Polar 单一路径。

   dataset 输出统一字段 `gps`，形状为 `[T, 3]`。特征为 `[dist, sin_theta, cos_theta]`，由 UE-BS 相对 UTM 坐标计算。若配置层继续保留 `gps_feature_mode` 字段，唯一合法值为 `relative_polar`；raw、UTM、relative、motion 和 motion-smooth 不再作为本 change 的交付内容。这样可以删除五套 ablation 配置和对应文档说明，并把后续 fusion、KD、论文主表都对齐到一个 GPS 表示。

3. 选择 GPS-Rel-Polar 是实验结果和建模约束共同驱动。

   从效果图看，`motion` 和 `motion_smooth` 的 train accuracy 较高，但 val Top-3、DBA 和 val loss 明显落后，说明短 GPS 历史下显式差分和速度特征未稳定泛化。`relative` 与 `utm` 在 Top-1/Top-3 上略高于 `relative_polar`，但 `relative_polar` 在 DBA 上领先，并以距离加 `sin/cos` 角度直接表达 beam selection 所需的几何先验。由于差距不大且当前目标是确定主 GPS 表示，保留 GPS-Rel-Polar、删除其余 ablation 是合适的；若后续论文需要附录对比，可从已跑日志摘取结果，不再要求代码主路径长期维护全部模式。

4. scaler 由训练集 fit，并通过数据集实例共享到 test。

   `build_dataloaders` 构建 train dataset 后，test dataset 必须复用 train dataset 的 GPS scaler。若用户直接评估单个 test dataset，则配置必须允许传入已保存 scaler 或在无 scaler 时以明确错误提示终止 GPS 归一化路径。替代方案是每个 split 自行 fit scaler，但这会泄漏 test 范围并污染实验结论。

5. GPS 模型沿用模态三件套。

   新增 `GpsFeatureExtractor`、`GpsModalityNet`、`GpsStudentModalityNet`。Teacher 使用小型 MLP embedding、LayerNorm、GRU、temporal attention 或 MHA residual 和 classifier；Student 使用更窄的 MLP、LayerNorm、单层 GRU 和小型 classifier。两者 forward 接收 `[B, T, 3]` GPS 张量，并返回 `(pred, features, output_features)`，保持 KD 兼容。

6. fusion 使用配置字段 `modalities` 控制分支和 fusion layer 输入维度。

   `FusionModalityNet` 和 `StudentModalityNet` 接收 `modalities: ["image", "radar", "gps"]`，默认值保持旧行为 `["image", "radar"]`。初始化时只创建被启用模态的分支，并根据启用模态数量构建 fusion projection。forward 通过可选入参接收对应张量，缺失启用模态时抛出清晰错误。替代方案是新增独立模型名如 `fusion_gps_teacher`，但组合数量会快速膨胀，不利于手动比较任意模态组合。

7. engine 的任务分发保持 `experiment.task` 粒度，但输入准备支持 GPS 和 fusion 模态列表。

   新增 `prepare_gps_inputs`，为历史 GPS 序列追加未来 zero padding，使输出时间长度和 image/radar 对齐。`forward_model` 对 `task: gps` 调用 GPS-only 模型；对 `task: fusion` 读取模型配置中的 `modalities` 并只准备所需输入。这样不需要复制训练循环。

8. 配置模板按实验目的分组，GPS 配置统一使用 GPS-Rel-Polar。

   新增或保留 `configs/gps/no_kd.yaml`、`configs/gps/student_no_kd.yaml`、`configs/gps/logits_kd.yaml`、`configs/gps/rkd.yaml`，其中 `gps_feature_mode: relative_polar`、`gps_input_size: 3`。删除 raw、UTM、relative、motion 和 motion-smooth 的独立 ablation 配置。Fusion 配置增加 `modalities`，例如 `["image", "gps"]`、`["radar", "gps"]`、`["image", "radar", "gps"]`。所有 Python 验证命令在任务中使用 `conda run -n kd_mm_beam`。

## Risks / Trade-offs

- [Risk] 运行环境缺少 `utm` 依赖 -> Mitigation：在项目依赖中显式添加，测试中覆盖经纬度转相对极坐标的基本路径。
- [Risk] 序列 CSV 旧文件缺少 GPS 列 -> Mitigation：未启用 GPS 时保持兼容；启用 GPS 时给出缺列错误并提示重新运行序列预处理。
- [Risk] GPS scaler 在 train/test dataset 构建顺序中未共享 -> Mitigation：在 `build_dataloaders` 中显式传递 train scaler，并增加测试断言 test dataset 未重新 fit。
- [Risk] GPS-Rel-Polar 不是 Top-1/Top-3 的单项最优 -> Mitigation：在设计中记录该取舍，主实现以 DBA、Top-5、几何可解释性和低复杂度为依据；如需论文附录，可引用已跑 ablation 日志。
- [Risk] fusion 任意组合导致输入维度和 forward 参数复杂化 -> Mitigation：集中使用 `modalities` 规范化、校验和分支字典，避免散落条件分支。
- [Risk] RKD 对 teacher/student feature 维度敏感 -> Mitigation：默认 GPS 与 fusion teacher/student 的 output hidden size 保持 64，配置测试覆盖 KD 构建。

## Migration Plan

1. 更新序列预处理，让新生成的 train/test sequence CSV 携带 GPS 路径列；旧 CSV 在不启用 GPS 的配置中继续可用。
2. 新增 GPS 读取、UTM 转换、GPS-Rel-Polar 特征构造和 scaler 逻辑，并将 `Scenario9Dataset` 扩展为按配置返回 `gps` 字段。
3. 更新 dataloader 构建，保证 train scaler 复用于 test。
4. 新增 GPS 模型模块和注册导出。
5. 扩展 batch 准备、forward 分发、trainer、validator 和 evaluator，使 `gps` 和 configurable fusion 能走统一流程。
6. 扩展 fusion teacher/student 支持 `modalities`，默认保持 `["image", "radar"]`。
7. 删除 raw、UTM、relative、motion 和 motion-smooth 的公开配置、文档入口和测试期望，只保留 GPS-Rel-Polar。
8. 使用 `conda run -n kd_mm_beam pytest` 与 `openspec status --change add-gps-modality-fusion` 验证。

## Open Questions

- 是否需要把六组 GPS ablation 的最终数值整理到论文附录或实验记录中；这不阻塞本 change 的实现收敛。
