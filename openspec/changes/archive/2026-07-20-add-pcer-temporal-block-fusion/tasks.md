## 1. 时间块 mask

- [x] 1.1 实现 `TemporalBlockMaskGenerator` 的六类 mask、逐样本确定性 seed、可选 source-frame grouped masking 和 shape 校验
- [x] 1.2 将 `pcer_curriculum` 三阶段概率接入现有训练 mask runtime，并保留完整视图 payload
- [x] 1.3 添加 mask 语义、curriculum、确定性、S4 和缺失输入清零聚焦测试

## 2. PCER 模型与损失

- [x] 2.1 实现共享 prototype bank 的 block evidence、availability-aware 静态融合和 counterfactual block Router
- [x] 2.2 将 PCER opt-in 配置与 forward payload 接入 UMaskBeamJEPA，并验证默认 current 数值/state-dict 兼容
- [x] 2.3 实现 full-to-masked temperature KL、向量化 leave-one-out topology target 和 Router KL/诊断
- [x] 2.4 添加 A2/A3 forward/backward、权重归一化、缺失权重为零、finite loss、Router 非零梯度和单次 backbone 证据复用测试

## 3. Checkpoint 与 quick runtime

- [x] 3.1 实现 opt-in validation-best checkpoint，保持默认 `last.pth` 行为不变并添加聚焦测试
- [x] 3.2 实现四组 MMW 15-domain、seed1、batch32、16 epoch resolved config 和 GPU4--7 fail-independent launcher
- [x] 3.3 实现逐样本 S0--S5 fixed-mask evaluator、Router 诊断、CSV/JSON/comparison 汇总与预注册 quick gate
- [x] 3.4 完成 15-domain 复制路径审计并生成 `outputs/quick_pcer_validation/implementation_notes.md` 与共同配置/mask examples

## 4. 运行前验证

- [x] 4.1 使用 `conda run -n kd_mm_beam` 完成语法/import/config parse、单 batch forward/backward 和损失量级检查
- [x] 4.2 使用 `conda run -n kd_mm_beam pytest` 完成 PCER、U-Mask、config、architecture、CLI 聚焦测试
- [x] 4.3 运行 `openspec validate add-pcer-temporal-block-fusion --strict`、`make verify-quick` 和 compile 检查

## 5. 四卡执行与结论

- [x] 5.1 检查 GPU4--7 占用并按 A0/A1/A2/A3 映射启动四个正式 quick-validation 任务，保存 PID、resolved config 和日志
- [x] 5.2 等待四个任务完成；明确修复代码错误并仅重跑失败任务，不因结果差而终止
- [x] 5.3 使用各自 `best.pth` 完成固定 mask 评测，生成 combined metrics、Router diagnostics 和 `comparison.md`
- [x] 5.4 核对成功条件、实现限制和潜在 bug，停止而不自动启动下一批实验
