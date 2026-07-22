## 1. 模型与风险数学

- [x] 1.1 实现独立 `PRSQualityHead`、按模态 input adapter、非负风险输出和参数量审计
- [x] 1.2 实现有界 beta、masked prior correction、cached evidence fusion和 D0--D3 override
- [x] 1.3 实现 CE/topology risk、transport drift、train-only robust normalization和 ranking loss

## 2. 共享缓存

- [x] 2.1 实现 C0 config/checkpoint 审计、冻结加载、pre-prototype hook、output/sensor statistics和无信道检查
- [x] 2.2 实现 train/validation/development condition bank、稳定 corruption identity和 sample-id 六 shard 分配
- [x] 2.3 实现 100-sample容量估算、clean-once/corrupt-view 分片 `.npz`、index/manifest/SHA与 normalization统计
- [x] 2.4 实现 shard重复/遗漏/pairing/shape/dtype/determinism检查和 C0 cache inference复现 gate

## 3. 轻量训练与统一评测

- [x] 3.1 实现 Q0--Q5固定配置、cache dataset、共享 batch order和一次 train-only loss量级校准
- [x] 3.2 实现 Q1--Q5 AdamW轻量训练、validation-best选择、early stopping和 quality-only checkpoint provenance
- [x] 3.3 实现 E0--E6任务指标、每传感器/severity/weather、D0--D3、质量/单调性和 gradient-alignment诊断
- [x] 3.4 实现 combined CSV、效率表、direction ranking、success gates和 `prsqdf_comparison.md`

## 4. 启动与状态

- [x] 4.1 实现 `scripts/precompute_prsqdf_cache_gpu0_5.sh`，设置物理 GPU后内部只使用 `cuda:0`
- [x] 4.2 实现 `scripts/run_prsqdf_quick_search_gpu0_5.sh`，保存 nvidia-smi/PID/状态并隔离单任务失败
- [x] 4.3 实现 prepare/preflight/status/retry-failed控制面并生成本地 common/resolved config

## 5. 验证与本地快筛

- [x] 5.1 增加 cache、risk、bounded correction、输入泄漏和 launcher定向测试
- [x] 5.2 使用 `conda run -n kd_mm_beam pytest` 运行 PR-SQDF preflight，并运行 OpenSpec、architecture、compile相关验证
- [x] 5.3 在 GPU0--5 完成共享 cache和 Q0--Q5 single-seed inner/development快筛，生成全部要求报告后停止
- [x] 5.4 保留 batch-2048 初筛，使用 batch 256、最低 10 epoch 和训练步数审计在独立目录重跑 Q0--Q5，重新生成报告后停止
