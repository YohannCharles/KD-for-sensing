## Context

当前 `generate_sequence_data()` 先按每个 `seq_index` 生成滑动窗口，再用 `all_seq_idx[:80%]` 和 `all_seq_idx[80%:]` 做 train/test 切分。这个方式对 Scenario 9 尚可，因为它有 136 条短 seq；但 Scenario 32 只有 15 条较长 seq，80/20 顺序切分后 test 只来自 3 条末尾轨迹段。

训练日志显示，Scene 32 中只包含 image/radar/LiDAR 的 7 种组合的最佳验证 Top-1 都停在 epoch 1 的多数类基线附近，随后训练准确率上升而验证准确率下降。GPS/mmWave 正常，说明训练框架整体并未失效，更可能是 split 覆盖不足和模态域偏移共同造成。

## Goals / Non-Goals

**Goals:**

- 让序列 CSV 生成使用可复现的标签分布感知 seq-level split 协议，降低小 seq 数场景的验证集偶然性。
- 保持窗口生成不跨 `seq_index`，避免历史输入和未来目标跨轨迹拼接。
- 让 Scene 32 的默认 split 覆盖更多轨迹段和 beam label 分布，而不是只验证最后 3 条 seq。
- 保持所有单模态和 fusion 实验使用同一组 train/test CSV，继续支持横向比较。
- 在 metadata 中记录 split 策略、seed、seq 列表和标签分布摘要，方便复现实验和解释曲线。

**Non-Goals:**

- 不修改模型结构、loss、scheduler、训练循环或评估指标定义。
- 不把单个 seq 内的相邻窗口随机拆到 train/test；这会造成强时间邻近泄漏。
- 不自动重训所有历史实验；实现后需要重新生成 split 和 cache，再启动新的训练。
- 不保留旧顺序 split 配置兼容；旧结果只能通过已保存的旧 CSV 和旧代码环境追溯，不作为新协议的一部分。

## Decisions

1. 在预处理阶段增加 `split_strategy`，而不是在 Dataset 或 DataLoader 中动态重切。

   训练配置已经依赖固定 CSV 路径来保证跨模态可比较。把 split 固化在预处理产物中，可以让 image、radar、GPS、LiDAR、mmWave 和 fusion 完全共享同一批窗口，并让 cache、checkpoint registry 和日志仍然可追踪。

   备选方案是在 Dataset 初始化时按 seed 动态划分窗口。这个方案不适合当前项目，因为不同模态配置可能在不同时间运行，稍有配置差异就会破坏横向比较。

2. split 的最小单位保持为完整 `seq_index`。

   每个窗口仍在单条 seq 内生成，train/test 划分也按完整 seq 分配。这样可以避免同一轨迹内高度相似的相邻窗口同时出现在训练和验证中，减少数据泄漏。

   备选方案是按窗口随机切分。它会让验证集看起来更稳定，但对这种连续采样任务会高估泛化能力。

3. 采用单一 split 协议：`balanced_seq`。

   预处理先生成所有合法窗口并统计每个 seq 的 beam label 分布，再用确定性贪心方式选择 test seq，使 test 窗口数量接近目标比例，同时尽量接近全量 label 分布。`split_seed` 只用于并列候选的稳定打散，避免引入多套策略分支。

   这样可以减少配置冗余：默认预处理配置只有一种实验协议，Scene 9 和 Scene 32 都用同一逻辑。旧的顺序切分不再作为可选模式暴露。

4. 增加 split 数量控制，但不隐藏 `training_set_pct`。

   `training_set_pct` 继续表示目标训练比例。新增 `min_test_sequences`、`test_sequence_count` 或等价字段用于小 seq 数场景。当指定数量控制时，实现应先满足显式数量，再尽量接近比例；未指定时按比例推导 test seq 数。

5. 输出 split metadata sidecar。

   序列预处理应在 train/test CSV 旁写出 JSON sidecar，例如 `split_metadata_RA_GPS_LIDAR.json`，记录：

   - 原始 CSV、输出 CSV、`split_protocol: balanced_seq`、`split_seed`、`training_set_pct`
   - train/test `seq_index` 列表和数量
   - train/test 窗口数
   - `beam8` 和所有目标时隙的 Top label 分布摘要

   训练运行 metadata 可以继续记录 CSV 路径和样本数；如果 sidecar 可发现，也应记录 split metadata 路径或核心字段。

## Risks / Trade-offs

- [Risk] 新 split 会让后续指标不能与旧顺序 split 直接比较。  
  Mitigation: 将变更标记为 breaking，并在文档中明确新旧 split 是不同实验协议；新协议的运行产物必须记录 split metadata。

- [Risk] `balanced_seq` 贪心策略可能过度贴合 beam label 分布，但没有保证视觉/雷达/LiDAR 外观域完全一致。  
  Mitigation: 报告 split seed 和 label 分布摘要；必要时后续单独提出 K-fold 协议，而不是在本变更中保留多策略分支。

- [Risk] Scene 32 seq 数太少，任何单次 train/test split 仍可能不稳定。  
  Mitigation: 提供 `min_test_sequences` 和多 seed 方案；把单次结果视为 baseline，把多 seed 平均作为更稳健结论。

- [Risk] 重新生成 CSV 后，已有 image/LiDAR cache 中仍有可复用单帧或 pair 缓存，但训练曲线会发生变化。  
  Mitigation: cache 目录按原始文件和预处理参数命中，split 变化不会污染样本内容；运行日志必须记录新的 CSV 和 split metadata。

## Migration Plan

1. 实现 `balanced_seq` 和 metadata 输出。
2. 用新协议重新生成 Scene 9 和 Scene 32 的统一 split CSV。
3. 使用所有单模态和 fusion 配置指向同一组新 CSV，重新训练图中 7 种 image/radar/LiDAR 组合，并对比 train/test label 分布与验证曲线。
4. 文档中标注旧顺序 split 与新 `balanced_seq` split 的实验结果不可直接混在同一表格里比较。
