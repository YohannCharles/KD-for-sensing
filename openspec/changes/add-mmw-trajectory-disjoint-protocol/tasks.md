## 1. 协议核心

- [x] 1.1 在 `src/kd_sensing/data/mmw/trajectory_protocol.py` 实现来源清单、显式元数据优先和资源 connected-component trajectory group 重建
- [x] 1.2 实现固定 seed 2026 的 group-level 80/10/10 分配、15-group 的 12/2/1 整数规则和不可满足分层约束报告
- [x] 1.3 实现窗口有效性、三组两两资源零交集、完整 manifest/hash/统计及 fail-closed 校验
- [x] 1.4 实现历史 manifest 身份恢复、暴露汇总、claim eligibility 与 protocol comparison

## 2. 运行边界

- [x] 2.1 将 trajectory protocol 精确绑定接入 MMW protocol dispatcher、pooled domain 映射和 train-only normalization 校验
- [x] 2.2 实现 test 默认封存和显式授权映射，确保普通训练只构建 train/validation

## 3. 公平基线

- [x] 3.1 复用 Candidate12 encoder/fusion 实现 M0--M3 的限定 head/loss/random-balanced 差异和同 split hash 检查
- [x] 3.2 实现 validation-only 训练、checkpoint 选择、细分指标、聚合分析和完整运行产物写入
- [x] 3.3 添加 GPU 0--3 独立启动、PID/exit code 和十分钟只读监控脚本

## 4. 测试与运行

- [x] 4.1 添加 `tests/test_mmw_trajectory_split.py`，覆盖 group 完整性、资源隔离、test 封存、四模态与公平性
- [x] 4.2 使用 `conda run -n kd_mm_beam` 执行 focused tests、两步 smoke、checkpoint round-trip 和 metrics 写入
- [x] 4.3 生成本地正式协议与审计产物，确认 46,860 candidates、group 数、split 数和 claim eligibility
- [ ] 4.4 运行 `openspec validate add-mmw-trajectory-disjoint-protocol --strict`、`make verify-quick` 和 compile 检查（OpenSpec、architecture、compile 已通过；全仓 lint 仍被既有 `tools/run_router_observability.py` 未使用 `_amp` 导入阻塞）
- [x] 4.5 确认 GPU 0--3 无其他用户进程后启动 M0--M3，监控至完成并生成 validation-only 最终分析
- [x] 4.6 复用最佳 checkpoint 补齐 validation 的 15 个非空模态 mask，并按 0%/25%/50%/75% 缺失率汇总 Top-1 和 ADBA

## 5. M4 ABTC

- [x] 5.1 实现 train-only、epoch-varying 的 availability-level 与组合均衡 mask 分配
- [x] 5.2 实现 topology-smoothed consistency loss，并复用 M2 prototype/topology 监督
- [x] 5.3 将 M4 接入 trajectory runner、checkpoint contract、指标聚合与 15-mask 评估，不扩大 public CLI
- [x] 5.4 添加 sampler/loss/paired-forward focused tests，并执行 M4 两步 smoke 与 checkpoint round-trip
- [x] 5.5 运行 OpenSpec strict、focused tests、lint 与 compile 验证
- [x] 5.6 实现并验证 M4-a uniform、M4-b balanced-only、M4-c generic-KL 三个固定因果消融
- [x] 5.7 在 GPU 0--3 分别启动 M4-a/M4-b/M4-c/M4 正式训练并记录 PID
