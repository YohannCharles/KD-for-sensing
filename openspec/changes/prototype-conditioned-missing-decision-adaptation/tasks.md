## 1. 模型与原型状态

- [x] 1.1 在 `BeamPrototypeBank` 与 U0 输出中实现只读、无标签的 prototype state 导出，并覆盖其数值不改变 logits 的测试。
- [x] 1.2 实现冻结 U0 包装器、mask bias 与共享低秩 `MissingDecisionAdapter`，确保 Full 显式旁路、B 零初始化和 prototype state detach。
- [x] 1.3 使用 `conda run -n kd_mm_beam pytest` 添加并运行 Adapter 的冻结、掩码顺序、零初始化、Full 等价和 A7 置换隔离单元测试。

## 2. 受审计的 Stage A 工作流

- [ ] 2.1 实现 protocol/checkpoint fail-closed 审计、U0 结构审计、确定性 20-epoch mask schedule、split 内 A7 置换和 train-only condition normalizer。
- [ ] 2.2 实现只训练 Adapter 的 AdamW/cosine/warmup 训练、15-mask 逐样本 NPZ 评估、Full 等价审计、聚合/Retention/SPA/行为诊断与运行 manifest。
- [ ] 2.3 使用 `conda run -n kd_mm_beam pytest` 覆盖 clean protocol fail-closed、validation 状态不变、mask schedule 一致及内部指标检查。

## 3. 独立分析与启动表面

- [ ] 3.1 新增 `tools/analyze_prototype_decision_adapter.py`，从逐样本产物独立 NumPy 重算指标、bootstrap、精确 McNemar 和 Holm 校正。
- [ ] 3.2 新增 GPU 0--7 Stage A 启动脚本及仅生成不执行的 Stage B 多 seed 脚本，记录 GPU/PID/命令/日志/状态且不终止无关进程。
- [ ] 3.3 更新架构边界测试以显式允许受控实验脚本，确保仍无新增 public CLI 或 canonical recipe。

## 4. 验证与 Seed 1 筛选

- [ ] 4.1 使用 `conda run -n kd_mm_beam` 运行 OpenSpec、模型/协议/架构测试和编译检查。
- [ ] 4.2 在启动前验证 checkpoint SHA256、clean protocol/audit、GPU 状态与 Stage A 配置；不满足任一条件时 fail-closed。
- [ ] 4.3 在 GPU 0--7 执行 Seed 1 Stage A，运行独立重算与统计，并根据预注册门槛生成阶段性结论及 Stage B 建议。

## 5. Full-pool capacity protocol

- [x] 5.1 实现 46,860 候选窗口发现、trajectory 可靠性判定、共享时间轴 80/20 macro split、真实边界 purge 和 588 历史身份的 fail-closed 恢复。
- [x] 5.2 复用 Radar/BS-GPS 确定性增强，输出 protocol/cache manifest、每 domain/Beam 支持量并在增强后完成资源级零交叉审计。
- [x] 5.3 扩展受控实验 loader/config 审计以接受 `mmw_full_pool_development_v1`，不放宽既有 clean-inner 或 public CLI 边界。

## 6. Full-data 两阶段运行

- [x] 6.1 实现测时、动态 epoch 计算和唯一 Full-data U0 的 GPU4 Stage 1，从头训练并发布带 SHA256 的 `last.pth`。
- [x] 6.2 新增 Full-pool 编排脚本，在 checkpoint 哈希通过后按当前实验范围并行运行 Adapter，并每 600 秒记录运行状态。
- [x] 6.3 保存四个实验的 15-mask 逐样本结果、Full 等价证据和内部指标，使用独立脚本以 `1e-7` 容差重算聚合、Retention、SPA、MAE、Within-3 与 ADBA。
- [x] 6.4 运行 OpenSpec、focused tests、架构/CLI/compile 验证，完成 Full-data 与旧 3,600 结果的描述性对照和证据结论。
- [x] 6.5 修正测时 probe 被 resume 为正式 U0、reference clean U0 profile 漂移和 prototype 塌缩未 fail-closed 的问题，并添加聚焦测试。
- [ ] 6.6 保留无效 FP16 run 证据，按修正协议从头重跑唯一 U0 和 GPU1--7 上的 A0--A7，重新生成独立指标与最终结论。
- [x] 6.7 将 Full-pool 作业从不可靠 CUDA ordinal 改为物理 GPU UUID 绑定，拒绝 GPU0，并以聚焦测试和运行 manifest 验证 GPU1--7 映射。
- [x] 6.8 并行构建 pooled domain 与 train-only GPS moments，固定顺序归并并验证与单线程统计一致。
- [x] 6.9 将 Stage 2 扩展为 GPU1 上 A0→A7 队列及 GPU2--7 上 A1--A6，并验证八项独立返回码和 GPU0 零使用。
- [x] 6.10 将 prepared CSV 的重复资源检查改为逐单元向量化验证、按真实路径去重及总并发不超过 90 的 fail-closed 并行校验，并覆盖缺失资源、Radar `_DA` 和非法 Beam 回归测试。
- [x] 6.11 将 Full-pool split audit 从逐行全列扫描改为保持同一身份口径的向量化资源集合与 `itertuples` 强哈希/帧展开，并用手工期望集合验证零交叉输入不变。
- [x] 6.12 严格复用现有 Image/LiDAR 帧缓存，生成 protocol-bound GPS 坐标与 scaler artifact，完成数值/覆盖/吞吐审计，并仅在 1.5x 吞吐门槛失败时构建唯一帧 LMDB。
- [ ] 6.13 为 U0 与 A1--A7 实现预注册训练损失早停，记录实际 epoch/steps/stop reason，按用户授权在物理 GPU0/4/6 启动 Stage 2，并在 GPU7 空闲后无重复地迁移尚未启动的 A7、在 A5 完成释放 GPU6 后迁移尚未启动的 A6。
- [x] 6.14 实现固定 `lambda=0.5`、`sigma=2.0` 的 circular ADBA-surrogate loss profile、B1/B4/B6/B7 四 GPU 编排、独立 ADBA 重算与聚焦测试，且保持原 CE profile 数值不变。
- [ ] 6.15 在物理 GPU0/4/6/7 并行运行 B1/B4/B6/B7，完成 15-mask、Full 等价、B6-B7 对照和最终 ADBA 导向分析。
- [x] 6.16 实现 Global/Lookup/Factorized bias、允许 mask schedule、确定性分层 fold、weight-space probe、条件四卡编排与独立重算，并以聚焦测试验证零初始化、Full 旁路和 held-out 零曝光。
- [x] 6.17 复用 B1 并运行单 seed Global/Lookup；若 MLP All-14 ADBA 严格高于 Lookup，则运行 fold 0 的 MLP/Factorized unseen pilot，输出阶段门槛和 held-out 结果。
