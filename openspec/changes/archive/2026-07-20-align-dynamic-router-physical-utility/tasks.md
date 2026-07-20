## 1. 配置与损失实现

- [x] 1.1 在 dynamic Router 配置解析中加入互斥 fused decision objective、margin 校验和旧配置默认兼容
- [x] 1.2 实现 Joint hard-label CE、beam-power soft CE 和 top-choice hard-negative margin，并保留统一 expected-utility 诊断
- [x] 1.3 将目标配置接入 UMaskBeamJEPA paired Joint loss，保持 power 仅进入 loss

## 2. 测试与运行编排

- [x] 2.1 添加目标数值、梯度、非法输入和配置互斥的聚焦测试
- [x] 2.2 添加固定 `PATR/H2R × 四目标` 八任务 launcher、resolved config 与不可变 manifest
- [x] 2.3 使用 `conda run -n kd_mm_beam` 完成 launcher dry-run、配置加载与聚焦测试
- [x] 2.4 扩展冻结 Joint evaluator 的新协议识别，并添加训练后自动评估 watcher

## 3. 夜间筛选

- [x] 3.1 严格校验 OpenSpec change 与旧 dynamic Router change
- [x] 3.2 在 GPU0--7 启动八个 seed1、batch64、40 epoch 训练任务并确认均进入训练
- [x] 3.3 记录运行 manifest、PID、日志、预计完成时间和后续固定评估计划
- [x] 3.4 启动 watcher，在训练完成后自动并行运行八候选固定 81-condition Joint 评估
