## 1. Evidence 与配置契约

- [x] 1.1 在 `PrototypeReliabilityRouter` 中实现固定宽度 `full/generic_confidence/prototype_topology` evidence profile 和 metadata
- [x] 1.2 将 evidence profile 接入 UMaskBeamJEPA 配置与窗口级特征屏蔽，保持默认 `full` 数值兼容
- [x] 1.3 放宽 JointCE 直接监督 H2R 时的 frame-rank 必选限制，并保留非分层候选拒绝规则

## 2. 八卡筛选与测试

- [x] 2.1 新增固定 GPU0--7 的八候选 launcher、resolved config、不可变 manifest 与产物边界
- [x] 2.2 增加 profile 数值屏蔽、参数公平、梯度、配置拒绝和 launcher identity 聚焦测试
- [x] 2.3 使用 `conda run -n kd_mm_beam pytest` 运行动态 Router 聚焦测试并修复失败

## 3. 校验与执行

- [x] 3.1 运行 `openspec validate simplify-prototype-h2r-screen --strict` 和必要快速架构/编译检查
- [x] 3.2 使用 `conda run -n kd_mm_beam` 完成八候选 dry-run/preflight，确认 batch64、固定 SHA 和 GPU 映射
- [ ] 3.3 在 GPU0--7 并行运行八个 seed1 训练任务并保存完整状态
- [ ] 3.4 在 GPU0--7 对完成 checkpoint 运行固定 81-condition Joint evaluation并汇总结果
- [x] 3.5 增加逐任务完成即在原 GPU 续跑固定评估的可恢复 watcher，避免短任务等待 40-epoch 对照
