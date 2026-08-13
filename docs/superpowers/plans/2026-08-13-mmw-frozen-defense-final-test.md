# MMW Frozen Defense and Final Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `openspec-apply-change` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完成冻结 Prototype-only 主线的噪声、防守统计、复杂度测量，并在全部预检通过后一次性解封 MMW test。

**Architecture:** validation-only 分析复用已有 evidence、requested-only simulator 和 train-only likelihood；新工具只负责聚类 bootstrap、复杂度测量和不可覆盖 seal。final test 使用同一模型/evidence/probing实现的显式 test role，不复制策略逻辑，并在创建 test loader 前写访问审计标记。

**Tech Stack:** Python 3.11、PyTorch、NumPy、pandas、OpenSpec、pytest；所有 Python 命令使用 `conda run -n kd_mm_beam`。

## Global Constraints

- MMW protocol 固定为 `mmw_id_stratified_block_v1` seed 0，test 为 6,003 windows。
- 四方法为 Prototype-only、matched Hard、RMBP-MM-local、AMBER-Full-local，seed 固定 `{1,2,3}`。
- 噪声固定 `{0,3,6} dB`；0 dB 是确定性锚点，3/6 dB 各3个 replica。
- bootstrap 固定 10,000 次、seed `20260813`，分别报告 trajectory 与 domain cluster CI。
- test 只访问一次；访问后不得调参、换 seed、换 checkpoint 或修改策略语义。
- 不创建 commit，不覆盖已有 outputs，不把 outputs/checkpoint/log 纳入源码。

---

### Task 1: Measurement-error replay

**Files:**
- Modify: `tools/eval_topology_predictor.py`
- Test: `tests/test_beam_probe_diagnostic.py`

**Interfaces:**
- Consumes: current Prototype-only matrix/checkpoint and train-only likelihood.
- Produces: three per-seed robustness results and one frozen summary.

- [ ] **Step 1:** Expose `--samples-per-pattern` while keeping the existing default and fixed grid.
- [ ] **Step 2:** Run the focused diagnostic tests and CLI help check.
- [ ] **Step 3:** Replay all 5,931 validation samples per mask for each seed without GPU/model forward.
- [ ] **Step 4:** Validate child hashes/grid and create the three-seed summary.

### Task 2: Paired cluster bootstrap

**Files:**
- Create: `tools/analyze_mmw_cluster_bootstrap.py`
- Create: `tests/test_mmw_cluster_bootstrap.py`

**Interfaces:**
- Consumes: canonical Prototype/Hard/RMBP per-sample ledgers and matrix sample identities.
- Produces: paired deltas and 95% percentile CIs for trajectory/domain clustering.

- [ ] **Step 1:** Add a synthetic failing test for mismatched paired keys and GT.
- [ ] **Step 2:** Implement strict key alignment and cluster bootstrap with NumPy/pandas only.
- [ ] **Step 3:** Run focused tests.
- [ ] **Step 4:** Run B=10,000 on Direct, Posterior Top-3 and TBCP-3 for both method pairs.

### Task 3: Frozen complexity benchmark

**Files:**
- Create: `tools/benchmark_mmw_frozen_methods.py`
- Create: `tests/test_mmw_complexity_benchmark.py`

**Interfaces:**
- Consumes: four frozen validation-best configs/checkpoints and one validation sample.
- Produces: parameter count, profiler-covered FLOPs, CUDA median/p95 and RF protocol table.

- [ ] **Step 1:** Test percentile aggregation and the immutable RF-count table.
- [ ] **Step 2:** Implement strict loading, `FlopCounterMode`, CUDA-event timing, and JSON/Markdown output.
- [ ] **Step 3:** Run focused tests.
- [ ] **Step 4:** Benchmark on a free A40 with batch1, FP32, 20 warmups and 100 repeats.

### Task 4: Final-test seal and preflight

**Files:**
- Create: `tools/mmw_final_test_panel.py`
- Create: `tests/test_mmw_final_test_panel.py`

**Interfaces:**
- Consumes: exactly 12 frozen run manifests/checkpoints, validation evidence and one train-only likelihood.
- Produces: an immutable SHA-bound seal manifest without constructing test data.

- [ ] **Step 1:** Test rejection of missing runs, wrong checkpoint role/hash and overwrite attempts.
- [ ] **Step 2:** Implement preflight checks without calling any test-binding/data-loader API.
- [ ] **Step 3:** Run focused tests and OpenSpec strict validation.
- [ ] **Step 4:** Create the seal manifest and independently verify all recorded hashes.

### Task 5: One-time final MMW test

**Files:**
- Modify: `tools/mmw_final_test_panel.py`
- Modify only if required to avoid duplicated policy logic: `src/kd_sensing/eval/beam_probe_diagnostic.py`
- Test: `tests/test_mmw_final_test_panel.py`
- Test: `tests/test_beam_probe_diagnostic.py`

**Interfaces:**
- Consumes: the verified seal SHA and explicit test authorization.
- Produces: Direct/Posterior Top-3/TBCP-3 evidence for 12 runs and missing-count 0/1/2/3 summaries.

- [ ] **Step 1:** Test that test-role evaluation is impossible without an intact seal and unused access marker.
- [ ] **Step 2:** Write an atomic access marker before test-loader construction, then reuse the existing 15-mask and requested-only probing logic under explicit test role.
- [ ] **Step 3:** Run focused tests without real test access.
- [ ] **Step 4:** Execute the panel exactly once, preserve per-run failures, and refuse rerun.
- [ ] **Step 5:** Verify 12/12 outputs, sample identities, protocol lineage and summary metrics; mark the experiment permanently frozen.

## Self-Review

- Spec coverage: tasks 15.1--15.5 map one-to-one to Tasks 1--5 above.
- No placeholders: every task names its files, inputs, outputs and verification boundary.
- Type/interface consistency: the seal manifest is the only authorization input to final test; validation analyses never consume test data.
