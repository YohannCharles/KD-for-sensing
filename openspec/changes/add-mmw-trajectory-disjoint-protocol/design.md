## Context

当前 15 个 Town3 domain 的 canonical `h5p1_strict_v2/all_sequences.csv` 共 46,860 个窗口。CSV 没有 `run_id`、`episode_id` 等显式运行 ID；`contiguous_segment_id` 包含 CAV，当前数据中每个 CAV 一个连续段，共 48 段。相同 domain 内各 CAV 的时间范围重合，并引用同一 RSU Radar 与 BS-GPS 帧，因此不能把 48 个 CAV 段直接当作独立 trajectory。

现有 `full_pool_protocol.py` 已提供资源列增强、资源身份抽取、配置精确绑定和 fail-closed 审计模式；Candidate12 已提供四模态 encoder、固定 MLP fusion、prototype head、训练循环与 validation 指标。新实现应复用这些 owner，而不是创建另一套数据集或 public CLI。

## Goals / Non-Goals

**Goals:**

- 从完整候选窗口确定性重建不会跨 split 共享资源的 trajectory groups。
- 生成固定 seed 的 group-level 80/10/10 manifest、两两资源零交集审计、历史暴露和完整统计。
- 默认只让普通训练加载 train/validation，并让 test 保持 manifest 层封存。
- 在同一协议和训练预算上运行 M0--M3，并只汇总 validation 结果。
- 让协议生成、loader 绑定、模型公平性和两步 smoke 都有 focused tests。

**Non-Goals:**

- 不声称复现原论文除 trajectory-disjoint 以外的完整设置。
- 不改变历史窗口、预测目标、四模态输入、beam codebook、预处理、模型 encoder/fusion 或既有 public CLI。
- 不把 channel、path、beam gain/power、历史 beam index 或未来 GPS 送入模型。
- 不访问 test 预测，不运行 multi-seed，不根据 split 分布或模型成绩更换 seed。
- 不提交本地协议、缓存、日志或 checkpoint。

## Decisions

### 1. 单一协议 owner 与本地工具入口

新增 `src/kd_sensing/data/mmw/trajectory_protocol.py` 作为构建、校验、审计和 domain 映射的唯一 owner；`tools/run_mmw_trajectory_baselines.py` 负责本地 prepare/train/aggregate，shell 脚本只负责 GPU 编排和监控。`protocol.py` 仅分派协议类型，data factory 继续使用现有 pooled-domain 构建。

替代方案是复制 Full-pool 工具并让每个训练方法自带 split 逻辑；这会产生多个可漂移的协议实现，拒绝采用。

### 2. trajectory unit 使用显式元数据优先、资源图回退

每行先记录 source CSV、稳定 sample/target/row identity、dependency frames 和所有审计资源。若存在可表示一次场景运行的显式字段，基础 unit 使用该字段；当前数据没有这些字段，回退为 `weather + sensor_scenario + contiguous_segment_id` 基础节点。

对共享 Radar、BS-GPS、target/dependency frame、完整 CSV row，或同一 scenario execution 且时间范围重叠的节点做 union-find；connected component 是最终 group。资源身份都加 condition/domain 命名空间，避免不同天气中相同相对路径或帧号被误合并。最终 group id 由排序后的成员和场景身份 SHA256 导出，保证稳定。

替代方案是把 `contiguous_segment_id` 或 CAV 当作 group；它会把共享 RSU 帧放进不同 split，拒绝采用。

### 3. 固定数量与确定性分层

组数为 50 时固定 40/5/5；否则枚举满足 validation/test 至少一组且总数精确的整数三元组，选择与 0.8/0.1/0.1 平方误差最小者，平局按 validation 不少于 test、再按固定顺序决定。当前 15 组得到 12/2/1。

分配先以 `weather + scenario` 和窗口数构造稳定排序，再用 seed 2026 的哈希打散，在不拆 group 的前提下贪心最小化 split 对目标 group 数、天气、scenario 和窗口量的偏差。若某类别覆盖在数学上不可满足，只记录约束缺口；不得通过拆组、访问模型结果或换 seed 修补。

### 4. test manifest 可审计但 loader 默认不可见

协议目录会写 test CSV、sample ids、hash 和分布，因为资源零交集必须可复核；`protocol_dataset_domains()` 默认只返回 train/validation 路径。只有显式 `allow_test_evaluation=True` 且调用专用映射时才返回 test 路径。普通训练配置必须 `training.final_test.enabled=false`，并拒绝任何 test CSV 注入。

这一区分允许审计标签分布但不执行 test 推理；本 change 的运行器没有 test-evaluation action。

### 5. 历史暴露按可恢复身份保守判定

扫描用户指定输出根中可读的 manifest 和 split CSV，提取稳定 sample/target identity及其历史 role。无法从 aggregate-only 文件恢复逐样本身份时记录 unavailable，不猜测。任一新 test sample 出现在历史 train 或方法选择 validation 时，`claim_eligible=false`；当前 Full-pool 已覆盖绝大多数候选窗口，预期为 false。

### 6. 四基线仅在 head/loss/training mask 上不同

复用 Candidate12 encoder 与 fusion，增加可选 linear head，保留一个 64 prototype bank。M0 用 linear CE；M1 用 prototype CE 但 topology loss 权重为零；M2 使用当前 topology prototype loss；M3 与 M2 相同并按 seed 2026 对四个单模态做随机均衡训练。四者使用同一 split hash、normalization、batch size、epoch、optimizer 和 validation-loss checkpoint selection。

## Risks / Trade-offs

- [当前数据只有 15 个完整场景执行 group，test 仅 1 组，天气/domain 覆盖不能全面平衡] -> 保持完整 group，明确报告约束缺口，不拆组也不换 seed。
- [共享资源图可能因过宽 identity 合并无关天气或场景] -> 所有资源 identity 使用 condition/domain 命名空间，场景重叠边只在同一 scenario execution 内生效。
- [历史 manifest 格式不统一，逐样本暴露可能无法完全恢复] -> 报告扫描文件、可恢复条目和 unavailable 源；claim eligibility 采用保守 false。
- [四卡 20 epoch 运行时间较长] -> prepare 和 2-step smoke 先失败关闭；正式任务独立 PID/exit code，一个失败不终止其他任务。
- [现有两个 unrelated active change 已违反 current spec 的单 change 约束] -> 本 change 不改写或归档它们，并在最终状态明确记录该既有治理冲突。

## Migration Plan

1. 先合入协议 owner、delta specs 和 focused tests，不改 canonical YAML。
2. 在本地生成 `outputs/mmw_trajectory_split/protocol/`，严格校验 46,860 candidates、资源零交集和 test sealed。
3. 用协议 train split 重新拟合 normalization，完成两步 smoke 与 checkpoint round-trip。
4. GPU 0--3 独立启动 M0--M3，只读取 train/validation，持续写状态。
5. 汇总 validation 和协议统计；若需回滚，只停止使用本地新 manifest，源码不迁移或删除任何既有产物。

## Open Questions

- 当前 15-group 数据不足以同时让 validation/test 覆盖全部天气与 domain；本 change 记录这一事实。更细粒度但仍资源互斥的 trajectory 只有在上游提供真实 run/episode 元数据或独立 RSU 时间段时才能成立。

