## 1. 身份与模型

- [x] 1.1 新增 G0--G5 常量、既有 cache/F1/F4 checkpoint 的 fail-closed 身份复现 gate 与 resolved config 生成。
- [x] 1.2 在既有 feature-fusion owner 中实现 prototype-compatible anchored score、正值 tau、非对角 Gram loss 与 focused model tests。
- [x] 1.3 实现冻结 F1 global branch、静态受限 beta、centered local logits 与 G4/G5 forward contract。

## 2. 训练与评测

- [x] 2.1 复用既有 inner train/validation mask、topology loss、checkpoint selection 与固定批次校准，实现 G2--G5 训练损失及 G5 固定 mask-group 聚合和 Full preserve loss。
- [x] 2.2 实现 G0/G1 checkpoint 复现、统一内层评测、error-distance、query/score/attention topology、模态 shuffle 和 global-local 替换诊断。
- [x] 2.3 输出本轮规定的 JSON、CSV、Markdown、参数和效率报告，并按六项 success gate 给出唯一推荐方向。

## 3. 编排与测试

- [x] 3.1 新增 GPU0--5 本地 launcher，保存 GPU 状态、PID、resolved config 与失败状态，且不自动重跑或启动后续训练。
- [x] 3.2 使用 `conda run -n kd_mm_beam pytest tests/test_topology_anchored_query_search.py -q` 运行 model/runner focused tests。
- [x] 3.3 使用 `openspec validate add-topology-anchored-prototype-query-search --strict`、`make verify-quick` 与 `make verify-compile` 验证，并更新任务状态。
