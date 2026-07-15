## Context

H5/P1 temporal launcher 当前声明 `stratified_by_target_beam_per_scene`，运行时按 label 打散单行；由于相邻 temporal window 共享历史帧和 target 邻域，同一 `seq_index` 及相同帧路径进入多个 split。与此同时，trainer 在没有 `validation` loader 时回退到 `test`，使 checkpoint 与 early stopping 直接消费 final test。内部 train/validation split 又只对 GPS 完整执行 train-subset refit，其他模态可能从完整父 dataset 或 pooled dataset 的首个 leaf 获取统计量。

现有代码已经提供 sequence-group splitter、dataset leaf/index 遍历和 normalization artifact，因此本 change 不新增通用 split 或 scaler 框架，只补齐共享 owner 的严格边界。

## Goals / Non-Goals

**Goals:**

- H5/P1 的 train、validation、test 在序列组、样本身份、历史输入帧和 target 帧上可证明不相交。
- final test 不参与 epoch validation、checkpoint 选择、scheduler 或 early stopping。
- 所有数据拟合统计量只消费实际 train indices，并以可追踪 artifact 传播到 validation/test。
- validation loss 对样本数或有效 token 数加权。
- 受旧协议影响的证据明确降级，只有新 split/provenance 完整后才能重新晋级。

**Non-Goals:**

- 不运行或提交真实训练、checkpoint、metrics、cache 或日志。
- 不在本 change 改造 U-Mask temporal/JEPA 语义、resume RNG 或 GPU 性能热点。
- 不为退役 workflow 增加兼容入口。

## Decisions

### 1. Temporal split 复用现有 group-safe owner

H5/P1 launcher 改用现有按 `seq_index`/稳定 group identity 的 per-scene splitter。split artifact 同时记录 group identity 和每个样本实际引用的历史/target resource identity；任何两 split 存在交集时 fail closed。相比在 launcher 内新增 guard-band splitter，这一选择复用当前 owner，并先解决已经确认的整组与帧重叠。

### 2. Trainer 永不把 test 绑定为 validation

trainer 只读取真实 `validation` loader。`training.use_early_stopping=true` 或其它 best-checkpoint 选择行为缺少 validation 时在训练开始前报错；显式 fixed-epoch 且无 validation 时跳过逐轮 validation，不生成基于验证指标的 `best.pth`，最终 checkpoint 为 `last.pth`。测试集只由显式 final evaluation 入口消费。

### 3. Normalization 采用 leaf + effective indices 协议

统一遍历 `Dataset`、`Subset` 和 `ConcatDataset` 的 leaf 与其实际训练 indices，在 train subset 上拟合 GPS、LiDAR、mmWave、CSI、position、occlusion 等已有统计 owner。validation/test 只接收已拟合对象；pooled leaves 需要共享统计时由训练 leaves 联合拟合，否则按明确 per-domain policy 保存，禁止静默取首个 leaf。

### 4. Validation loss 聚合保留分子与分母

evaluation pass 累加 unreduced loss 对应的有效样本/token 总量；若现有 loss 只返回 batch mean，则以 batch 有效计数恢复加权和。最终 loss 为总和除总计数，不再除 batch 数。

### 5. 旧证据只降级，不推断替代数值

claim/protocol 文档记录 H5/P1 旧 split 为 `not_comparable`、列出泄漏类型与重跑 gate；本 change 不从 ignored outputs 推断新结果，也不把 smoke 升级为正式 claim。

## Risks / Trade-offs

- [组安全拆分改变样本量和标签分布] -> 在 manifest 记录每 split 的 group/sample/label 统计，并接受指标不可与旧运行直接比较。
- [大量旧配置缺少 validation] -> 配置加载在选模启用时 fail closed；仅明确 fixed-epoch/no-selection 的 workflow 可无 validation。
- [多模态统计拟合可能增加启动时间] -> 复用一次性 artifact 和 digest，禁止在 validation/test 重复拟合。
- [旧 checkpoint 依赖 best.pth] -> 历史 artifact 只读保留；新无 validation 运行明确使用 `last.pth`，不伪造 best。

## Migration Plan

1. 先更新文档，将旧 H5/P1 evidence 标为不可比较。
2. 落地 split identity 审计和 trainer fail-closed 测试，再切换 launcher 默认策略。
3. 统一 normalization fit/apply 与 validation loss 聚合，刷新 focused tests。
4. 更新受影响配置为独立 validation 或显式 fixed-epoch/no-selection。
5. 运行 focused、quick、CLI/config、compile、全量 pytest 和 OpenSpec strict；真实重跑在源码变更之外执行。

回滚时可恢复旧代码，但不得把旧泄漏结果恢复为 current claim；数据协议回滚必须通过新的 OpenSpec change 明确说明。

## Open Questions

- 无。真实重跑的资源排期属于本地实验执行，不影响本 change 的源码与证据门禁。
