## Why

冻结 Full-pool U0 已经能给出 Beam 分布，但它给不出**可靠性承诺**：现有一切结论都是 Top-1/Top-3 点估计，没有任何一条能回答「扫这几个波束，真值在里面的概率至少是多少」。这一缺口在模态缺失下最尖锐——同一个模型在 `full` 与 `radar_only` 上的可信度差一个量级，而部署侧只看到同一个 Top-K 列表。

已完成的零训练诊断（`outputs/conformal_beam_diagnostic/`，设定 N、9,180 条 inner validation、15 种 mask、未访问 outer test）给出三个已测事实：

- **单一边际阈值在退化 mask 上系统性欠覆盖**。名义 0.90 时，frame 级随机切分下 `radar_only` 只有 0.7386（−16.1 pp），9/15 个 mask 低于名义，跨 mask 覆盖跨度 0.2028；名义 0.70 时 `radar_only` 掉到 0.5035，跨度 0.2630。
- **按可用性模式分层是对的方向**。同一切分下跨度从 0.2028 收到 **0.0166**（α=0.1）、0.2630 收到 **0.0094**（α=0.3），且平均集合大小几乎不变（35.4 → 34.8）。
- **等预算的固定 Top-K 被压制**。与边际方案同样的平均开销下，最差 mask 覆盖为 0.8283 / 0.7309 / 0.5706 / 0.4519 / **0.1558**（α=0.05/0.1/0.2/0.3/0.5）。α=0.5 时 conformal 在每个 mask 上都给到约 0.49、平均只扫 7.3 个波束，而 top-7 塌到 0.156——同等扫描开销下三倍的可靠性差距。

但**按轨迹整块切分（无泄漏）后条件覆盖并未闭合**：mask 分层的跨度只从 0.1735 降到 0.0988，最差 mask `image_only` 为 0.8598，8/15 低于名义。追加已观测协变量并不能补上：`mask × weather` 更差（0.1202），`mask × domain` 在 α=0.3 下既更差（0.1616 vs 0.1340）又更贵（19.2 vs 16.2 波束），且两个 domain 的 calibration 轨迹数为 0，使 **31.8%** 的测试样本只能退回粗阈值。因此真正的问题不是「分层不够细」，而是**标定轨迹与部署轨迹之间存在分布漂移**。

这正是第二创新点的位置：**在原型几何上做可用性条件化、且对标定→部署轨迹漂移稳健的波束候选弧**。它与创新点一（`BeamPrototypeBank` 原型聚类）是同一套几何的第二次使用——nonconformity 分数就是 U0 `_head_logits` 对原型的余弦量，不是外挂的置信度。近邻工作 SCAN-BEST（arXiv 2503.13801，IEEE TCCN 2026）做的是 near-field 跨频段辅助的 conformal risk control，L-ARC（arXiv 2405.07976）做的是 RKHS 局部化风险，两者都**没有模态缺失轴**；本提案的条件轴与环形弧约束是二者都不覆盖的部分。

## What Changes

- 增加环形波束弧闭包：在已审计的 `ula_dft_phase_cycle_v1` 拓扑上，把无序 conformal 集合闭包成最短覆盖弧。弧是集合的超集，因此覆盖率单调不降，split-conformal 有效性直接继承，不需要重新推导保证。
- 增加五条阈值估计路线的单次预注册筛选：C0 边际 split CP、C1 mask 条件 Mondrian CP、C2 轨迹级 leave-one-trajectory-out cross-conformal、C3 轨迹级分布稳健膨胀、C4 原型空间局部化阈值函数，以及等容量的 C5 分层标签置换负对照。
- 把「标定/测试切分粒度」本身作为受控变量而非实现细节：轨迹整块切分是唯一主结果口径，frame 级随机切分只作为可交换性对照同时报告，禁止单独引用后者。
- 增加预注册门槛与判死规则：主门槛在设定 N、α=0.1、5 个预注册切分种子上判定；任一路线未同时通过有效性、条件性、代价与负对照门槛即判死，不调参、不加种子、不换切分、不访问 outer test。
- 增加本地运行器、弧几何与阈值估计器的聚焦测试；不新增 public CLI、canonical recipe，不重训 U0、encoder、router 或 prototype。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `u0-mainline`: 定义冻结 U0 之上只读、零训练的集合值输出合约——原型度量空间的 nonconformity 分数、可用性条件阈值、环形弧闭包，以及不得改变任何 U0 前向数值。
- `clean-data-integrity`: 定义标定集必须来自 inner validation 而非 train、轨迹整块切分的无泄漏要求、唯一可拟合对象（C4 阈值函数）只由标定侧轨迹拟合，且 `outer_test_accessed` 保持 false。
- `repo-boundaries`: 明确诊断与筛选产物落在 `outputs/conformal_beam_diagnostic/` 与 `outputs/conformal_beam_screen/`，实验脚本不扩展公共 CLI。

## Impact

新增 `src/kd_sensing/baselines/conformal_beam_arcs.py`（弧几何）与 `conformal_shift_robust.py`（C2--C5 阈值估计器），复用已落地的 `conformal_beam_sets.py` 原语、`router_observability.py` 的冻结 U0 表征缓存与 `full_pool_bt_scl.load_audited_topology`。新增 `tools/run_conformal_beam_screen.py` 与对应聚焦测试。不修改 U0 canonical recipe、encoder、fusion、prototype、classifier 或公共 CLI；不训练任何骨干。
