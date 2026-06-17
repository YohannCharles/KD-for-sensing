## Why

`add-predictive-jepa-robustness` 已经归档并进入 current specs，但工作树仍暴露出 post-archive 收口漂移：新 capability 未登记 lifecycle、自动归档生成的 Purpose 仍为 `TBD`、架构边界测试把合法的 GPS-query/condition-id 防护文案误判为旧路线回流。与此同时，Predictive Robustness 又继续扩大 `jepa_gps_shortcut_benchmark.py` 的职责，若不先固化治理边界，后续真实 claim 和 runner 拆分都会变得难以审计。

## What Changes

- 收口 Predictive JEPA 归档后的 OpenSpec/文档治理：补齐 lifecycle inventory、真实 Purpose、claim 状态边界和 smoke/real-run caveat。
- 调整项目健康护栏的静态文案规则，使其继续拒绝退役路线 active wording，同时允许 current JEPA specs 中对 GPS-query baseline 兼容性和 condition-id 禁用字段的合法描述。
- 明确 Predictive Robustness 训练配置、benchmark manifest 和真实 claim 的边界：单个训练 profile 不等价于完整 P0-P5 benchmark；synthetic/smoke manifest 不得升级为真实性能 claim。
- 为 `jepa_gps_shortcut_benchmark.py` 建立可实施的拆分方案和热点预算：先不改 metric 语义，优先把 predictive/suite normalization、aggregation、artifact writer 等职责拆到窄模块或登记明确暂缓理由。
- 不新增训练入口、不启动真实训练、不提交 benchmark 输出、checkpoint、cache、CSV、PNG 或 logs。
- 不恢复 KD、HiST/Hist、Top8 selector standalone、GPS residual、camera residual 或其它退役研究线。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `project-health-guardrails`: 收紧归档后 current spec 的 Purpose/lifecycle 完整性检查，并避免旧路线 wording guard 对合法 JEPA compatibility/forbidden-field 描述产生误报。
- `spec-lifecycle-boundaries`: 明确归档后 capability 进入 current specs 时必须同步 lifecycle inventory、真实 Purpose 和 active/archive 状态解释。
- `predictive-jepa-robustness`: 明确 Predictive Robustness 的 post-archive 文档治理、训练 profile 与完整 P-suite benchmark 的边界，以及 mock/smoke claim 状态限制。
- `jepa-gps-shortcut-benchmark`: 增加 runner 职责拆分和 suite-specific 模块化边界，防止 predictive/CxD/Scenario D 逻辑继续堆叠在单一巨型 runner 中。

## Impact

- 受影响 OpenSpec/文档：
  - `docs/project_surface_inventory.md`
  - `openspec/specs/project-health-guardrails/spec.md`
  - `openspec/specs/spec-lifecycle-boundaries/spec.md`
  - `openspec/specs/predictive-jepa-robustness/spec.md`
  - `openspec/specs/jepa-gps-shortcut-benchmark/spec.md`
  - predictive JEPA 相关主线文档和 claim 账本的 caveat 文案
- 受影响测试：
  - `tests/test_architecture_boundaries.py`
  - 必要时追加文档/guard focused tests，覆盖合法 GPS-query compatibility wording、condition-id forbidden-field wording 和新 capability lifecycle。
- 可能受影响代码：
  - `src/kd_sensing/diagnostics/jepa_gps_shortcut_benchmark.py` 及新拆出的窄模块。
  - 不改变模型 forward、difficulty operator、training loop、evaluation metrics 或 checkpoint schema。
- 验证优先级：
  - `openspec validate stabilize-predictive-jepa-governance --strict`
  - `openspec validate --specs --strict`
  - `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`
  - `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py tests/test_jepa_gps_shortcut_benchmark.py -q`
