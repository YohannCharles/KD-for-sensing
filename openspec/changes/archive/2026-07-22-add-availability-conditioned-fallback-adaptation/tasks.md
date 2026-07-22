## 1. 规格与 F1 身份

- [x] 1.1 校验 proposal、design、delta spec 与 inner-only/claim-ineligible 停止边界
- [x] 1.2 定位并记录 F1 validation-best checkpoint、resolved config、source cache、fusion/prototype/topology 与 SHA

## 2. Pattern、模型与 focused tests

- [x] 2.1 实现 14-pattern 枚举、condition/token 顺序映射、20-block mask、固定两级均衡 schedule 与 validation manifest
- [x] 2.2 实现冻结 F1 wrapper、Full bypass、U1 pattern SSF、U2 shared hyper-SSF 与 U3 contextual residual
- [x] 2.3 实现 U4/U5 单模态 auxiliary、四个 unimodal teacher/probe、per-sample auxiliary/KD 与 pattern-balanced loss
- [x] 2.4 添加 pattern、bypass、missing-token、结构公平、aux/teacher 隔离、禁止输入与 finite backward focused tests
- [x] 2.5 使用 `conda run -n kd_mm_beam pytest tests/test_availability_fallback_search.py -q` 运行 focused preflight

## 3. Cache 与运行前门槛

- [x] 3.1 实现六分片 F1 token cache 转换、source/SHA/schema/coverage 校验与 compact dtype
- [x] 3.2 实现 Full/四种 single-missing logits 误差和 Top1/Top3/Within-3 parity gate
- [x] 3.3 在 GPU0--5 可用性检查后完成 cache 转换、merge、schedule 与 `preflight_tests.txt`

## 4. Teacher、训练、评测与编排

- [x] 4.1 实现 teacher/probe 训练、train-only loss calibration、U0--U5 cache 训练、group-balanced validation selection 与 early stopping
- [x] 4.2 实现 14-pattern、missing-count、modality-absent、weather/sector/error-distance、adapter replacement、表示、aux/teacher、modality shuffle 与效率评测
- [x] 4.3 实现 GPU0--5 独立 launcher、PID/状态/日志/resolved config 管理与失败隔离
- [x] 4.4 使用 `conda run -n kd_mm_beam` 完成四个 teacher 和 U0--U5 single-seed inner quick search，不运行 outer test、multi-seed 或下一轮训练

## 5. 汇总与验证

- [x] 5.1 生成要求的 CSV/JSON/Markdown 产物、23 个研究问题答案、success gates 与唯一推荐方向
- [x] 5.2 运行 `openspec validate add-availability-conditioned-fallback-adaptation --strict`、focused tests、`make verify-quick` 与 `make verify-compile`
- [x] 5.3 核对源码 diff 不包含 dataset、cache、checkpoint、日志或其他用户已有改动，并更新任务状态
