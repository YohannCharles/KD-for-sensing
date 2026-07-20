## 1. 实现审计与测试

- [x] 1.1 定位 A0-A3 最佳 checkpoint、resolved config、融合粒度和历史日志并生成 implementation audit
- [x] 1.2 添加 counterfactual target 的 A-E synthetic sanity tests

## 2. 离线诊断实现

- [x] 2.1 实现 A1/A3 D0-D4 验证集先验统计和固定测试 mask 权重替换评测
- [x] 2.2 实现 router 样本、block、模态、时间和 full/masked 动态性统计
- [x] 2.3 实现 A3 target 分布、质量相关性、router 对齐和 checkpoint 稳定性统计
- [x] 2.4 实现单 batch route-only 梯度审计且不执行参数更新
- [x] 2.5 实现 A0-A3 S3 分模态指标、A1/A3 权重迁移、A3 worst 错误与混淆统计
- [x] 2.6 实现 block 与 modality-group contribution 非加性对比
- [x] 2.7 添加统一 shell 入口和架构边界登记

## 3. 运行与结论

- [x] 3.1 使用 `conda run -n kd_mm_beam` 运行 synthetic tests 和脚本级测试
- [x] 3.2 使用空闲单 GPU 运行完整只读诊断并核验全部本地产物
- [x] 3.3 完成 diagnostic summary、A-F 方向判断和最小下一步建议
- [x] 3.4 运行 `openspec validate diagnose-pcer-router-dynamics --strict` 与相关快速验证
