## Context

当前仓库没有 active change，但 `add-predictive-jepa-robustness` 的归档结果仍以未提交状态存在：新 current spec、archive 目录、诊断 manifest 和派生训练配置已经出现在工作树里。上一轮只读检查显示：

- `openspec validate --specs --strict` 通过，说明 OpenSpec schema 层面没有破损。
- predictive JEPA 的模型、difficulty 和 benchmark focused tests 通过，说明新增运行路径的基础 shape/schema 测试是绿的。
- `tests/test_architecture_boundaries.py` 失败，集中在 lifecycle inventory、自动生成 `TBD` Purpose 和 retired-route wording guard 误报。
- `src/kd_sensing/diagnostics/jepa_gps_shortcut_benchmark.py` 已膨胀到多 suite、多 aggregation、多 artifact writer 的巨型文件，Predictive Robustness 继续加重了该热点。

因此本 change 不是继续发明模型，而是让刚进入 current 的 Predictive Robustness 具备可审计收口，并为后续真实 claim 和 runner 拆分铺好边界。

## Goals / Non-Goals

**Goals:**

- 让 predictive JEPA 归档后的 current spec、inventory、文档和架构边界测试重新一致。
- 保留 retired-route guard 的防回流价值，同时减少对合法 current JEPA 文案的误报。
- 明确 `P4` 训练 profile、P0-P5 benchmark suite、synthetic smoke manifest 和 strict real claim 的关系。
- 为 benchmark runner 建立窄模块拆分方向和测试迁移策略，优先保证输出 schema 和指标语义不变。
- 给 implementation 阶段列出足够小的验证闭环，避免把本地真实训练或产物纳入源码。

**Non-Goals:**

- 不改变 JEPA predictive hybrid 模型 forward、pooler 数值语义、difficulty operator 语义或 checkpoint 加载语义。
- 不启动真实 train-then-evaluate，不登记真实 DBA claim。
- 不把 `kd-sensing-jepa-gps-shortcut-benchmark` 拆成新的公共 CLI；公共入口保持不变。
- 不恢复或兼容退役 KD、HiST/Hist、Top8 selector standalone、GPS residual、camera residual、Raymobtime s008、CRAF/MARF/G2D 或 Multimodal-NF 路线。

## Decisions

1. 先修治理闭环，再拆 runner。

   当前最短红灯路径是 lifecycle/Purpose/writing guard。先补齐 `predictive-jepa-robustness` 的 inventory 分类和 Purpose，再调整 guard 规则或文案，使 `tests/test_architecture_boundaries.py` 变绿。Runner 拆分可在同一 change 后半段做低风险移动，但不得成为阻塞治理收口的前提。

   备选方案是直接大拆 `jepa_gps_shortcut_benchmark.py`。风险是大规模 move 会掩盖治理红灯，且容易改变输出 schema。

2. Guard 修复优先采用“明确合法语境”而不是简单放宽所有规则。

   Retired-route guard 应继续拦截旧 KD/HiST/Top8/residual 等 active wording。对合法 JEPA current 文案，优先通过以下方式处理：

   - 在 current spec 中把 GPS-query 表述限定为“现有 JEPA baseline compatibility”，避免被读作退役路线推荐。
   - 对 `blocked_condition_fields`、`forbidden_condition_fields` 这类安全诊断字段，增加测试级 allowlist 或上下文判断。
   - 保持 `condition_id_consumed: False` 这样的显式安全信号。

   备选方案是删除 `gps_condition` / `image_condition` 字段名或移除 GPS-query baseline 描述。这样会损害诊断可审计性和现有 baseline compatibility。

3. Predictive Robustness claim 分三层写清楚。

   ```text
   train config
        │ 可能只训练 P4 或其它 curriculum profile
        ▼
   benchmark manifest
        │ P0-P5 regional aggregate + strict comparability
        ▼
   claim registry
        │ 只有 real strict comparable run 才能升级
        ▼
   paper/report wording
   ```

   实现时，文档和 specs 必须明确：单个训练 profile 不等于完整 P-suite evaluation；synthetic metrics、mock weights、allow_missing_artifacts 或 partial model group 只能生成 schema evidence，不能生成真实 claim。

4. Benchmark runner 拆分以 suite-specific helper 为边界。

   首轮拆分建议只移动纯函数和 schema/aggregation helper，不改变 public CLI 和 result dict：

   ```text
   jepa_gps_shortcut_benchmark.py
     ├─ 保留 CLI-facing run_jepa_gps_shortcut_benchmark facade
     ├─ benchmark_manifest.py        # manifest load/validation/comparability
     ├─ benchmark_predictive.py      # predictive suite normalize/rows/summary
     ├─ benchmark_scenario_d.py      # Scenario D/CxD helpers
     ├─ benchmark_artifacts.py       # OutputRegistry / artifact plan
     └─ benchmark_figures.py         # optional plotting/skipped fallback
   ```

   如果 implementation 阶段判断拆分会导致过大 diff，可以先在 inventory 中登记新的热点预算和拆分方向，保证 governance 红灯先清掉。

5. 验证按层推进。

   - OpenSpec：先跑本 change strict validate，再跑 all specs strict validate。
   - Governance：跑 `tests/test_architecture_boundaries.py`。
   - Predictive focused：跑 `tests/test_config_load_characterization.py`、`tests/test_jepa_gps_shortcut_benchmark.py`，必要时追加 `tests/test_modality_difficulty.py` 和 `tests/test_gps_conditioned_jepa.py`。
   - 不要求全量 `pytest -q` 作为 proposal 阶段目标；implementation 完成后可视风险决定。

## Risks / Trade-offs

- [Risk] 静态 wording guard 过度放宽后漏掉退役路线回流。→ Mitigation：只为明确的 current JEPA compatibility 和 forbidden-field diagnostics 加例外，继续保留 retired token + active wording 组合检测。
- [Risk] Runner 拆分引入 import cycle 或改变输出 schema。→ Mitigation：先抽纯函数，保留原 facade；用现有 benchmark tests 比对 output_files、CSV/JSON key 和 claim status。
- [Risk] 文档把 P4 train profile 误写成 P0-P5 完整 claim。→ Mitigation：在 predictive spec、experiment matrix、protocols 和 claim registry 中同时标注 train/eval/benchmark 分工。
- [Risk] 未跟踪 archive/current spec 状态被误当成 active change。→ Mitigation：agent navigation 和 lifecycle spec 明确以 `openspec list --json`、current specs、inventory 和 git status 共同判断。
- [Risk] 为了让测试变绿而只改测试、不修文档事实。→ Mitigation：tasks 要求同时修改 inventory/Purpose/caveat 文档，再调整测试规则。

## Migration Plan

1. 补齐 predictive capability 的 current lifecycle inventory 和真实 Purpose。
2. 更新 current specs 和 docs 中的 claim/caveat 文案，区分 train profile、benchmark suite 和 real claim。
3. 调整 architecture boundary guard，使合法 JEPA compatibility 和 forbidden-field diagnostics 不再误报。
4. 运行 governance focused tests，确认归档收口红灯清除。
5. 视 diff 大小拆分 benchmark runner 的 predictive helper；若暂缓拆分，则更新 inventory 热点预算和后续拆分任务。
6. 运行 OpenSpec、architecture、config 和 benchmark focused tests。

Rollback 策略：若 runner 拆分引发非预期行为，保留治理修复，回退拆分到原 facade 内部实现，并把拆分作为后续独立 change；不得回退 lifecycle/Purpose/claim caveat 修复。

## Open Questions

- implementation 阶段是否将 runner 拆分纳入同一 change，还是仅登记预算并另起专门 refactor change。
- Guard 例外应放在测试 helper 的上下文判断中，还是通过修改 spec 文案完全避开误报。
- Predictive train config 是否应在本 change 中追加 P0-P5 eval profile 示例，还是只通过 benchmark manifest 强调完整 evaluation。
