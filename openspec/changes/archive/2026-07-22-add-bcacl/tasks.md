## 1. P0 基线兼容与配置契约

- [x] 1.1 增加 BCACL 配置解析与校验，确保未声明或 `enabled=false` 时不向模型注入参数或运行分支。
- [x] 1.2 增加同 seed、同输入、同 checkpoint 的 state-dict 与 forward 基线等价测试，并使用 `conda run -n kd_mm_beam pytest` 验证。

## 2. P1 单模态私有与共享监督

- [x] 2.1 实现按当前 modality order 建立的独立 Linear+LayerNorm 投影、可选私有头和可选共享 64 类头。
- [x] 2.2 基于自然 `observed_mask` 实现可独立开关的私有/共享 CE 与逐模态 top1/损失诊断。
- [x] 2.3 使用 `conda run -n kd_mm_beam pytest` 验证 U1/U2 梯度所有权、自然缺失过滤和 shape。

## 3. P2-P3 模态原型关系与固定教师

- [x] 3.1 实现独立 `[M,K,D]` 模态原型 buffers、有限 Beam 关系 log-softmax 和 epoch/EMA 训练集更新。
- [x] 3.2 实现配置驱动的 fixed teacher 选择、observed/fusion mask 隔离和 `KL(teacher.detach() || student)`。
- [x] 3.3 使用 `conda run -n kd_mm_beam pytest` 验证教师零梯度、学生非零梯度、缺失过滤、dropout 特权教师和零样本稳定性。

## 4. P4 Detached Two-Stage

- [x] 4.1 在 U-Mask extension 中实现 Phase 1 BCACL base loss 替换，保证融合与融合恢复原型不接收 Phase 1 梯度。
- [x] 4.2 实现 Phase 2 model-only initialization、optimizer 前 encoder/BCACL 冻结和现有融合/恢复原型训练。
- [x] 4.3 将 stage、模态原型与 BCACL 状态纳入 checkpoint/resume，并使用 `conda run -n kd_mm_beam pytest` 验证保存恢复与冻结状态。

## 5. P5-P6 质量矩阵与自动教师

- [x] 5.1 实现 float32/DDP 训练 epoch 统计、类内方差、prototype-nearest hard negatives、质量 EMA 和低样本保护。
- [x] 5.2 实现每学生样本至多一个教师、质量 margin、初始化过滤和 graph-connected zero loss。
- [x] 5.3 使用 `conda run -n kd_mm_beam pytest` 验证 `[M,K]` shape、有限数值、top-1 教师约束和质量差过滤。

## 6. P7 诊断、评估与消融

- [x] 6.1 持久化每 epoch 单模态性能、损失、4x4 迁移矩阵、逐 Beam 教师、样本计数、初始化率和质量 JSON/CSV/TensorBoard 标量。
- [x] 6.2 扩展 fixed-mask 汇总为 15-pattern 明细及 Full、Single/Double/Triple/All-14 macro/worst，并保留现有分类与通信指标。
- [x] 6.3 增加 MMW/DeepSense6G 完整 BCACL 示例、U0--U5 独立消融配置和两阶段 inner-only launcher。
- [x] 6.4 使用 `conda run -n kd_mm_beam pytest` 验证诊断矩阵对角线、逐 Beam 分布、完整 pattern 分组和配置解析。

## 7. 分层回归与 Smoke

- [x] 7.1 运行 `openspec validate add-bcacl --strict`、BCACL focused tests、T2/model/config/runtime focused tests和 `make verify-quick`。
- [x] 7.2 使用极小 inner train/validation 数据运行 Experiment A，验证有限损失、checkpoint 保存恢复和 14 个不完整组合评估。

## 8. Single-Seed Inner 实验

- [x] 8.1 已记录取消 Experiment B：新主线只保留 U2 aux-joint，不启动旧 U1/U2 两阶段 development 训练。
- [x] 8.2 已记录取消 Experiment C：fixed teacher、relation transfer 与教师矩阵已退役。
- [x] 8.3 已记录取消 Experiment D：自动教师与逐 Beam teacher quality 已退役。
- [x] 8.4 已确认未由本次收口触发 outer test、multi-seed、正式 claim 或后台训练任务。
