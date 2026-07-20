## 1. 共享模型与损失

- [x] 1.1 扩展 PCER evidence-only、flat block、hierarchical 和 mask-residual router forward 契约
- [x] 1.2 实现 standalone-quality、on-policy block 与 on-policy modality target 和统一 KL 诊断
- [x] 1.3 实现 B7 balanced LOMO distillation 与复用主 forward 的 unimodal auxiliary loss
- [x] 1.4 记录 route/beam 梯度余弦、router/backbone/prototype 梯度范数和 B7 删除计数

## 2. 聚焦测试与 preflight

- [x] 2.1 添加 B0-B7 config 严格解析和默认 checkpoint/forward 兼容测试
- [x] 2.2 添加 B2/B3/B4 target 符号、on-policy removal、detach 和顺序 synthetic tests
- [x] 2.3 添加 hierarchical alpha/beta、mask residual、缺失权重和梯度专项测试
- [x] 2.4 添加 B7 均衡采样、teacher detach、单次主 backbone 证据复用和各分支梯度测试
- [x] 2.5 使用 `conda run -n kd_mm_beam` 实现并运行真实 batch preflight 与一次自动量级审计

## 3. 八路运行与评测脚本

- [x] 3.1 生成共享协议和 B0-B7 resolved config、manifest、implementation notes 与 GPU0-7 shell 入口
- [x] 3.2 实现 fail-independent launcher、PID/资源监控、完成态恢复和失败方向单独重跑
- [x] 3.3 实现 best checkpoint 的 S0-S5、S3 分模态、B7 单模态 evidence 和统一指标评测
- [x] 3.4 实现 B0/B1/B5/B6 权重替换以及所有 router/target 机制诊断
- [x] 3.5 实现历史 A0-A3 合并、compute cost、Pareto gate、八问题和方向报告

## 4. 执行与结论

- [x] 4.1 运行语法/import/config parse、聚焦 pytest、OpenSpec strict、quick 与 compile 验证
- [x] 4.2 检查 GPU0-7 外部进程并完成八方向 16 epoch 单 seed 训练
- [x] 4.3 对所有成功任务运行固定评测；明确修复并仅重跑代码失败任务
- [x] 4.4 核验统一输出、Winner/Promising/Reject 与唯一下一步建议后停止
