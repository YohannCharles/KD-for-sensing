## 1. Verify 基座

- [x] 1.1 选择最小聚合形式：Makefile、justfile、脚本或文档命令块。
- [x] 1.2 增加 quick verify，聚合 `openspec validate --all --strict` 和 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
- [x] 1.3 增加 CLI/config verify，聚合 `conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_config_load_characterization.py -q`。
- [x] 1.4 增加 docs/spec verify 或记录现有 OpenSpec/document health 命令。

## 2. 环境复现

- [x] 2.1 增加最小 smoke/dev 环境声明或环境导出脚本。
- [x] 2.2 更新 `ENVIRONMENT.md`，区分 official BeamBench 环境、当前 `kd_mm_beam`、smoke/dev 环境和 GPU/full training 环境。
- [x] 2.3 确认环境文件不包含本地路径、凭证、真实数据、checkpoint 或日志。

## 3. CI / lint

- [x] 3.1 增加无数据 CI 或本地等价命令，默认只跑 quick verify。
- [x] 3.2 增加轻量 Python compile/lint 检查，优先覆盖 package CLI 和 tracked scripts。
- [x] 3.3 更新 README、AGENTS、`docs/agent_navigation.md` 和 inventory 的验证说明。

## 4. 验证

- [x] 4.1 运行 `openspec validate add-reproducible-verification-foundation --strict`。
- [x] 4.2 运行新增 quick verify。
- [x] 4.3 运行 `conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_config_load_characterization.py -q`。
