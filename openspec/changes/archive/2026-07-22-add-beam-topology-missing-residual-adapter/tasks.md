## 1. 基础模型与缓存

- [x] 1.1 实现B2静态化审计、C0-static fallback manifest和来源SHA校验
- [x] 1.2 实现inner-train/inner-validation paired residual cache转换、forbidden-field审计和量化一致性gate
- [x] 1.3 生成base metrics、cache report、common config与implementation notes

## 2. Adapter与训练

- [x] 2.1 实现Full/负对照严格bypass、missing image/LiDAR剩余evidence选择和静态sigmoid alpha
- [x] 2.2 实现R0-R5结构、train-only normalization/mean/calibration、共同batch order和validation-loss checkpoint选择
- [x] 2.3 实现plain residual与cyclic topology-aware loss，保证R2/R3结构相同且teacher stop-gradient

## 3. 评估与报告

- [x] 3.1 实现统一beam指标、S3 aggregate、oracle recovery和teacher纠错/新错统计
- [x] 3.2 实现weather/sector/错误距离、residual predictability和D0-D3替换诊断
- [x] 3.3 实现效率、六项success gate、所有CSV/JSON与最终comparison report

## 4. 启动与验证

- [x] 4.1 实现GPU0-5失败隔离launcher、PID/status/resolved config保存和失败任务定向重跑
- [x] 4.2 增加20项preflight定向测试并使用`conda run -n kd_mm_beam pytest tests/test_missing_residual_adapter.py -q`验证
- [x] 4.3 使用`openspec validate add-beam-topology-missing-residual-adapter --strict`和`conda run -n kd_mm_beam python scripts/verify_compile.py`校验

## 5. 本地快速验证

- [x] 5.1 运行`conda run -n kd_mm_beam python analysis/precompute_missing_residual_cache.py`并只在cache gate通过后继续
- [x] 5.2 检查GPU0-5，运行六组single-seed inner任务并汇总主表、分层、动态替换、效率和唯一建议
- [x] 5.3 确认未运行outer test、multi-seed或下一轮完整训练
