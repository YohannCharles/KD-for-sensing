## 1. 规格与身份

- [x] 1.1 校验 OpenSpec proposal、design、delta spec 与任务清单，并记录 inner-only/claim-ineligible 停止边界
- [x] 1.2 定位 C0 validation-best checkpoint、resolved config、prototype/topology 与 frozen source cache SHA

## 2. 融合模型与测试

- [x] 2.1 实现统一 modality adapter、time/modality embedding、availability 校验与 masked token 构造
- [x] 2.2 实现 F0、F1 concat MLP、F2 fusion-token Transformer、F3 learned-query、F4/F5 prototype-query 和 F5 auxiliary evidence 输出
- [x] 2.3 添加 mask、索引、shape、query 来源、prototype 冻结、结构公平、zero/shuffle token 与禁止输入 focused tests
- [x] 2.4 使用 `conda run -n kd_mm_beam pytest tests/test_feature_fusion_quick_search.py -q` 运行 preflight tests

## 3. Cache 与一致性

- [x] 3.1 实现六 shard frozen cache 转换、源身份/SHA/coverage 校验、紧凑 dtype 与 manifest/report 输出
- [x] 3.2 实现 F0 Full/单模态缺失 logits 误差和 Top1/Top3/Within-3 parity gate
- [x] 3.3 在 GPU0--5 可用性检查后完成六 shard cache 转换与 merge，并保存 `preflight_tests.txt`

## 4. 训练、评测与编排

- [x] 4.1 实现统一 structured mask、train-only lambda calibration、F1--F5 cache 训练、combined validation selection 与 early stopping
- [x] 4.2 实现统一 Full/single-missing/S0--S5、weather/sector、LiDAR dependence、token shuffle、attention beam-specificity 和效率评测
- [x] 4.3 实现 GPU0--5 独立 launcher、PID/状态/日志/resolved config 管理与失败隔离
- [x] 4.4 运行 F0--F5 single-seed inner quick search，不运行 outer test、multi-seed 或下一轮端到端训练

## 5. 汇总与验证

- [x] 5.1 生成用户要求的 CSV/JSON/Markdown 产物、18 个研究问题答案和唯一推荐方向
- [x] 5.2 运行 `openspec validate add-beam-conditioned-feature-fusion-search --strict`、focused tests、`make verify-quick` 与 `make verify-compile`
- [x] 5.3 核对源码 diff 不包含 dataset、cache、checkpoint、日志或其他用户已有改动，并更新任务状态
