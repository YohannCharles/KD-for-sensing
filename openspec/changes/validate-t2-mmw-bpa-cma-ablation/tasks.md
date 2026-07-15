## 1. CMA 目标与配置契约

- [x] 1.1 实现基于融合/模态池化特征、availability 和稳定样本身份的跨 batch 多正样本 CMA analogue
- [x] 1.2 为 U-Mask Beam JEPA 增加默认关闭的 CMA 温度、权重和开关配置，并实现 BPA/CMA 互斥校验
- [x] 1.3 将 CMA 接入训练 loss，记录 raw/weighted loss、anchor 数和 batch 唯一样本数，不改变旧配置行为
- [x] 1.4 使用 `conda run -n kd_mm_beam` 增加并运行 CMA 配对、负样本、缺失 anchor、重复身份、label 无关、梯度和互斥测试

## 2. MMW 消融运行面

- [x] 2.1 扩展 MMW launcher，生成 `T2-NoBPA`、`T2-BPA2CMA`、`T2-Linear`、`T2-CLS`、`T2-CLS-CMA` 的严格配置覆盖和 provenance
- [x] 2.2 扩展 evaluator/任务输出提取，使新增方法使用现有持久化 mask identities 和 epoch-40 checkpoint
- [x] 2.3 增加全量、端点 beam 和内部 beam 的 paired 汇总与论文绘图脚本
- [x] 2.4 使用 `conda run -n kd_mm_beam` 增加并运行 launcher、evaluator、身份一致性和汇总单元测试

## 3. 预运行验证

- [x] 3.1 运行 `openspec validate validate-t2-mmw-bpa-cma-ablation --strict`
- [x] 3.2 使用 `conda run -n kd_mm_beam` 对五个新增方法执行配置 dry-run 和单 optimizer-step smoke test
- [x] 3.3 审计 smoke artifacts，确认方法间仅存在预注册差异、CMA loss 有限且 checkpoint 可加载

## 4. 三 seed 训练

- [ ] 4.1 在 GPU0-7 第一波并行运行五个新增方法的 seeds 1/2，共 10 个 40-epoch 任务
- [ ] 4.2 在 GPU0-7 第二波并行运行五个新增方法的 seed 3，共 5 个 40-epoch 任务
- [ ] 4.3 校验 15 个训练任务均完成 40 epochs，记录 checkpoint、最终 loss 尺度和运行完整性

## 5. 配对评估与论文产物

- [ ] 5.1 使用 `conda run -n kd_mm_beam` 对六方法三 seed 运行统一 MMW 缺失率矩阵和 task-output 提取
- [ ] 5.2 校验所有方法的 sample/mask/target 身份与有效样本计数一致后生成 paired delta
- [ ] 5.3 生成三 seed Markdown/JSON/CSV 汇总、缺失率曲线、端点/内部 circular-vs-linear 图及 CMA 配对图
- [ ] 5.4 根据效应大小、seed 稳定性和切片结果撰写受证据约束的中文分析，不把 CMA analogue 声称为完整 AMBER 复现

## 6. 回归与收尾

- [ ] 6.1 运行 `make verify-quick` 及与本 change 相关的 `conda run -n kd_mm_beam pytest ... -q`
- [ ] 6.2 运行 `openspec validate --all --strict` 并检查 `git status`，确认未跟踪本地数据、日志或 checkpoint
