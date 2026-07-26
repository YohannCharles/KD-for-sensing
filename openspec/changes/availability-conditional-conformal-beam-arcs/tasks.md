## 1. 零训练诊断（已完成，作为动机证据封存）

- [x] 1.1 实现 split-conformal 原语 `conformal_beam_sets.py`：轨迹身份、轨迹整块切分、frame 级随机切分对照、有限样本 `ceil((n+1)(1-alpha))` 分位、分层阈值与有限回退。
- [x] 1.2 实现只读运行器 `tools/run_conformal_beam_diagnostic.py`：复用冻结 U0 表征缓存重放 15 种 mask，输出 marginal / mask / mask×weather / mask×domain 与等预算 fixed-top-K 对照。
- [x] 1.3 使用 `conda run -n kd_mm_beam pytest tests/test_conformal_beam_sets.py` 覆盖有限样本分位、无轨迹泄漏、对照确实泄漏、有限回退与分层优于合并。
- [x] 1.4 在设定 N、α ∈ {0.05, 0.1, 0.2, 0.3, 0.5} 下运行两种切分，产出 `outputs/conformal_beam_diagnostic/split_{track,random}_alpha*/`；`trained=false`、`outer_test_accessed=false`。

## 2. 环形弧闭包

- [ ] 2.1 新增 `src/kd_sensing/baselines/conformal_beam_arcs.py`：绑定 `ula_dft_phase_cycle_v1` 审计 manifest，实现包含给定集合的最短环形弧、弧长与弧覆盖；拓扑 id、波束数或 domain 审计不符时 fail closed。
- [ ] 2.2 实现弧的超集断言与弧长比统计，确保弧覆盖率恒不低于集合覆盖率。
- [ ] 2.3 使用 `conda run -n kd_mm_beam pytest tests/test_conformal_beam_arcs.py` 覆盖跨 0/63 回绕、单元素集合、全码本集合、超集性质与非法拓扑 fail-closed。

## 3. 漂移稳健阈值估计器

- [ ] 3.1 新增 `src/kd_sensing/baselines/conformal_shift_robust.py`：实现 C2 leave-one-trajectory-out cross-conformal、C3 轨迹级 β 分位稳健膨胀、C4 原型空间局部化阈值函数、C5 等容量分层标签置换。
- [ ] 3.2 保证 C4 是本筛选中唯一可拟合对象，其拟合只读取标定侧轨迹；测试侧轨迹在拟合期不可见。
- [ ] 3.3 保证 C5 只置换分层标签而不置换分数，且其估计器容量与 C3/C4 严格一致。
- [ ] 3.4 使用 `conda run -n kd_mm_beam pytest tests/test_conformal_shift_robust.py` 覆盖可交换数据上 C0--C4 均达名义水平、C2 不重用测试轨迹、C4 无测试侧泄漏、C5 容量相等与置换确定性。

## 4. 预注册筛选运行

- [ ] 4.1 新增 `tools/run_conformal_beam_screen.py`：复用 `preflight`、冻结 U0 SHA256 校验与表征缓存，按 5 个预注册切分种子 × 两种切分粒度 × 5 个 α 运行 C0--C5，输出逐 mask 长表、弧长表与回退比例。
- [ ] 4.2 实现轨迹级重抽区间（重抽单元为 `(domain, cav)` 轨迹，非帧），所有区间按轨迹给出。
- [ ] 4.3 实现 G1--G6 门槛判定表 `success_gates.csv`，并把判死规则写入报告；不得在门槛失败后调 α、调 β、加种子或换切分。
- [ ] 4.4 使用 `conda run -n kd_mm_beam pytest` 覆盖运行器的协议 fail-closed、`trained=false`、`outer_test_accessed=false` 与两种切分必须同时输出。

## 5. 验证与结论

- [ ] 5.1 运行 `openspec validate availability-conditional-conformal-beam-arcs --strict`。
- [ ] 5.2 使用 `conda run -n kd_mm_beam pytest tests/ -q` 与 `conda run -n kd_mm_beam python scripts/verify_compile.py` 完成全量回归。
- [ ] 5.3 产出 `outputs/conformal_beam_screen/conformal_beam_screen_report.md`：同时列出两种切分、全部 α、G1--G6 判定与显式结论；未通过时按判死规则直接产出负结果报告，不启动 multi-seed 骨干或 outer test。
