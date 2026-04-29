## Context

当前项目已经具备配置驱动训练、评估、KD teacher 加载、唯一运行目录、TensorBoard/日志输出，以及 GPS/LiDAR 训练集归一化复用。但这些能力仍依赖“当前运行目录”内的 `checkpoints/best.pth`，当固定 `run_name` 自动追加时间戳后，KD 默认配置中的 `outputs/<slug>_teacher_no_kd/checkpoints/best.pth` 很容易失效。评估入口也会为了拿 GPS scaler 或 LiDAR normalizer 重新构建 train dataset，这让同一个 checkpoint 的评估结果可能被后续数据或配置变化影响。

本 change 需要跨训练、评估、数据加载、预处理和规格文档一起收敛，所有验证命令使用 `conda run -n kd_mm_beam ...`。

## Goals / Non-Goals

**Goals:**

- 提供稳定的最佳 checkpoint 归档目录，使 KD 和评估不受时间戳输出目录影响。
- 将 checkpoint 解析、归档复制、归一化工件保存和运行日志记录做成可测试的公共逻辑。
- 让 GPS scaler 和 LiDAR normalizer/stats 与训练产物绑定，评估默认复用训练时状态。
- 修正序列窗口边界、`portion` 采样语义、删除未使用的 `gps_smooth_window` 死配置，并修正过期 GRU 规格。

**Non-Goals:**

- 不改变核心模型结构、损失函数、优化器或现有指标定义。
- 不迁移历史 checkpoint 内容格式；历史 `state_dict` 和带 `state_dict` 的 checkpoint 仍需兼容加载。
- 不强制删除旧的 `outputs/<run_name>/checkpoints/best.pth` 路径，旧路径作为回退继续可用。

## Decisions

1. 新增公共 artifact registry 工具模块，默认目录为 `outputs/best_checkpoints/`。

   该模块负责生成 slug、格式化精度、清理同一 slug 的旧归档、复制最佳 checkpoint、写入 sidecar 元数据和解析最佳 checkpoint。使用普通文件复制而不是移动，保留每次运行目录内原始产物，便于排查训练过程。备选方案是让训练目录软链接到最新输出，但跨平台和远程文件系统兼容性更差。

2. checkpoint 解析采用“显式路径优先，registry 优先于旧默认路径”的顺序。

   评估入口的 `--weights` 或绝对 checkpoint 路径保留最高优先级；KD teacher 默认解析时先查 registry 中与 teacher baseline slug 匹配的最高精度 checkpoint，再回退到 `paths.weights_dir / distillation.teacher_model_name`。这样既满足默认实验免受时间戳影响，又不剥夺用户显式指定权重的能力。

3. registry 文件名使用配置 slug 和验证 Top-1。

   默认 slug 取 `experiment.name`，缺失时取 `output.run_name`，再缺失时根据 task、模型角色和 KD 模式合成。teacher no-KD 的归档名称满足 `<slug>_teacher_no_kd_acc_<val_top1>.pth`；其他配置也使用相同模式，确保 student/KD 结果可被人工识别。sidecar JSON 记录原始 run_dir、epoch、metric、split、配置路径或实验名、归一化工件路径和源 checkpoint 路径。

4. 归一化工件随训练保存，并由评估优先加载。

   GPSStandardScaler 增加 save/load，训练构建 dataloader 后将 scaler 保存到当前 run_dir 的 `artifacts/gps_scaler.npz`，并可复制或记录到 registry sidecar。LiDAR normalizer 继续复用现有 save/load，但当配置未显式提供 `stats_path` 时，训练也会保存到 `artifacts/lidar_normalizer.npz`。评估加载 checkpoint 后先读取 sidecar 或 checkpoint 附带 metadata 中的归一化路径；只有找不到归档元数据时，才回退到当前构建 train dataset 的兼容路径。

5. 序列和采样语义以“可比较实验”为优先。

   序列生成窗口条件改为包含最后一个合法窗口，即 `start + in_len + out_len <= seq_len_rows`。`portion < 1.0` 默认使用按 `seq_index` 分组的确定性均匀采样或配置化随机采样，而不是 CSV 头部连续样本；采样策略、seed 和实际样本数写入运行 metadata。

6. 删除 `gps_smooth_window` 死配置。

   当前 `gps_smooth_window` 只是从配置传入 dataset 和 GPS 特征构造函数，但实际特征构造没有使用它。实现阶段应移除默认配置、示例配置、`Scenario9Dataset` 显式参数、`load_gps_feature_sequence`/`build_gps_features` 参数和相关文档引用。若旧配置仍携带该字段，可通过 dataset 的通用兼容参数忽略，但不得再表现为受支持配置项。备选方案是补齐平滑实现，但这会改变 GPS 输入分布，不符合本 change 的稳定实验目标。

7. GRU 默认层数以当前配置和测试为准。

   GPS、RadarStudent 和 LiDAR 单模态 teacher/student/KD 默认 `gru_params` 统一为 `[64, 64, 1]`；image+radar fusion teacher 保留 `[64, 64, 2]`，fusion student 保留 `[64, 64, 1]`。

## Risks / Trade-offs

- [Risk] registry 中同一 slug 存在多个精度文件，解析可能误选旧文件 → 复制新最佳 checkpoint 时清理同 slug 的旧归档，并以 sidecar 中的 metric 作为排序依据。
- [Risk] 显式权重与 registry 优先级让用户困惑 → 日志和 `checkpoint_load` 必须记录候选来源、最终路径和是否来自 registry。
- [Risk] 归一化工件路径被移动后 sidecar 失效 → sidecar 使用相对 registry 的路径或同时记录原始 run_dir 路径；缺失时给出清晰错误或进入兼容回退。
- [Risk] `portion` 采样改变 smoke test 样本集合 → 默认保持确定性 seed，并在任务中更新对应测试预期。
- [Risk] 删除 `gps_smooth_window` 后历史外部配置仍传入该字段 → 保留兼容忽略或给出清晰迁移说明，并确认 GPS `relative_polar` 特征输出不变。

## Migration Plan

- 先实现 registry 工具、保存/解析路径和日志记录，并保留旧 `outputs/<slug>/checkpoints/best.pth` 回退。
- 再接入 GPS/LiDAR 归一化工件保存加载，确保无归一化模态不受影响。
- 然后修正序列和采样逻辑，补充单元测试覆盖窗口数量、分组采样和 metadata。
- 最后更新规格、README 和配置文档，运行 `conda run -n kd_mm_beam pytest -q -p no:cacheprovider` 与 `openspec validate --all`。

## Open Questions

- 历史已经存在的带时间戳 teacher 输出是否需要一次性扫描并导入 registry；本 change 默认只要求新训练产物自动归档，历史导入可以通过后续脚本补充。
