## 1. 策略落地

- [x] 1.1 为 maintainer context index 的 entrypoint entries 增加 owner module、responsibility、output boundary 和可选 retired route guard。
- [x] 1.2 更新 `docs/project_surface_inventory.md`，说明 CLI/脚本职责边界。
- [x] 1.3 更新 `docs/agent_navigation.md` 中 CLI/scripts 路由，要求先查 owner metadata。

## 2. 健康检查

- [x] 2.1 更新架构边界测试，校验每个 entrypoint 有 owner metadata 和 output boundary。
- [x] 2.2 对 `thin_cli_alias` 增加轻量检查，拒绝明显训练 loop、optimizer step、模型 forward 或大段 dataset parsing marker。
- [x] 2.3 校验 owner module/script 路径存在，并保持 pyproject 与 index 双向一致。

## 3. 当前入口审计

- [x] 3.1 为现有 package CLI 补齐 owner metadata。
- [x] 3.2 为现有 `scripts/` 和 shell orchestration 补齐 output boundary。
- [x] 3.3 确认当前打开的 `beambench_check_dataset.py` 等薄 alias 只委托 owner module。

## 4. 验证

- [x] 4.1 运行 `openspec validate codify-thin-cli-alias-policy --strict`。
- [x] 4.2 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
- [x] 4.3 未触碰 CLI help，无需追加运行 `conda run -n kd_mm_beam pytest tests/test_cli_help.py -q`。
