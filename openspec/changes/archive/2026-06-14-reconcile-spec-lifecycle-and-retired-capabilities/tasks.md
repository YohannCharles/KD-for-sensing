## 1. Lifecycle Inventory

- [x] 1.1 在 `docs/project_surface_inventory.md` 新增 OpenSpec capability lifecycle 分类说明，定义 `current`、`supporting` 和 `retired-tombstone`。
- [x] 1.2 枚举 `openspec/specs/*/spec.md` 并为每个 capability 记录唯一 lifecycle 分类。
- [x] 1.3 将 HiST/Hist、Raymobtime s008、GPS coarse anchor、GPS residual、camera residual、geometry residual、standalone Top8 selector、CRAF/MARF/G2D、Multimodal-NF 和旧 KD 等能力分类为 retired 或 supporting，并记录支撑代码例外。
- [x] 1.4 确认 current/supporting/retired 分类与 README 当前入口、docs workflow 和现有 migration guard 说明一致。

## 2. OpenSpec Wording Reconciliation

- [x] 2.1 清理 `openspec/specs/project-architecture/spec.md` 中把 HiST/Hist、Raymobtime s008 或其它退役路线描述为 active mainline、当前推荐入口或当前热点的旧 wording。
- [x] 2.2 将仍有效的通用 helper、BGAM TopK 支撑、metrics、cleanup 或 migration guard 语义改写为 supporting-only，不恢复旧 standalone workflow。
- [x] 2.3 更新退役墓碑 spec 的 Purpose 或首个 requirement，使每个 retired-tombstone capability 打开后能直接看出已退役或仅作防回流边界。
- [x] 2.4 检查 current specs 不再出现未加退役/历史限定的旧 CLI、旧配置、旧 workflow 推荐说法。

## 3. Navigation And Documentation

- [x] 3.1 更新 `docs/agent_navigation.md`，加入 lifecycle-first 读取规则、supporting 与 retired-tombstone 的解释规则。
- [x] 3.2 更新 `docs/agent_navigation.md` 中 active/archive/ignored cache 状态判断，明确 `.pytest_cache`、`__pycache__`、未跟踪 archive 和本地产物不能覆盖当前 specs。
- [x] 3.3 更新 `openspec/specs/ai-maintainer-navigation/spec.md`，使导航规范与新增 lifecycle 规则一致。
- [x] 3.4 如 README 或 docs workflow 中仍有旧 active wording，改为退役、历史、supporting 或当前 workflow 指向。

## 4. Health Guardrails

- [x] 4.1 扩展 `tests/test_architecture_boundaries.py`，检查 lifecycle inventory 覆盖全部 `openspec/specs/*/spec.md`，且 lifecycle 值只允许 `current`、`supporting`、`retired-tombstone`。
- [x] 4.2 增加 retired-tombstone wording 检查，拒绝墓碑 spec 缺少退役语义或无历史限定地恢复当前推荐入口 wording。
- [x] 4.3 增加 current docs/specs 旧 active wording 检查，覆盖 HiST/Hist、Raymobtime s008、Top8 selector standalone workflow、GPS residual、camera residual、CRAF/MARF/G2D、Multimodal-NF 和旧 KD。
- [x] 4.4 保持 tracked artifact 边界检查，不因 ignored `__pycache__`、`.pytest_cache`、`outputs/` 或 `logs/` 本地存在而失败。

## 5. Verification

- [x] 5.1 运行 `openspec validate reconcile-spec-lifecycle-and-retired-capabilities --strict`。
- [x] 5.2 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
- [x] 5.3 如改动 README/docs workflow 或配置引用扫描规则，追加运行 `conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_config_load_characterization.py -q`。
- [x] 5.4 复查 `git status --short --ignored`，确认新增源码变更不包含本地数据、outputs、logs、cache、checkpoint 或 Python bytecode。
- [x] 5.5 在最终说明中记录完成的验证命令、未运行项及剩余风险。
