## 1. Renderer

- [x] 1.1 在 `kd_sensing.diagnostics` 中新增或扩展窄 HTML renderer，输入现有 dashboard summary dict，输出完整静态 HTML 字符串。
- [x] 1.2 为 HTML renderer 增加统一的 HTML escaping helper，覆盖路径、claim id、warning、hint、active change 名称和任意本地 artifact 文本。
- [x] 1.3 HTML 页面展示 metadata、run states、active changes、resources、claim readiness、paper readiness、warnings、candidate-only caveat 和 next actions。
- [x] 1.4 HTML renderer 在空 summary 或缺少可选字段时仍生成可读页面。

## 2. CLI

- [x] 2.1 扩展 `kd-sensing-research-dashboard`，增加 `--output-html` 或等价参数，复用同一次 summary 构建结果。
- [x] 2.2 CLI 写出 HTML 时创建必要父目录，并打印 `dashboard_html: <path>` 或等价提示。
- [x] 2.3 保持 CLI 只读：不得启动训练、服务、清理、后台进程，不得修改 claim registry、README、OpenSpec 或实验文档。
- [x] 2.4 不新增重复 console script；如需调整入口说明，只更新既有 `kd-sensing-research-dashboard` 文档。

## 3. 文档

- [x] 3.1 更新 README 或相关诊断文档，补充 HTML dashboard 命令示例和 ignored output 边界。
- [x] 3.2 更新 `docs/agent_context/claims.md` 或 `docs/agent_context/diagnostics.md`，说明 HTML dashboard 是 candidate-only/readiness 视图，不是正式 claim registry。
- [x] 3.3 如 inventory 或 maintainer context index 提到 dashboard 输出能力，同步加入 HTML 输出边界和 focused validation。

## 4. 测试

- [x] 4.1 扩展 `tests/test_research_claim_harvester.py` 或新增 focused test，验证 HTML section、candidate-only caveat、空 summary fallback 和 escaping。
- [x] 4.2 增加 CLI 测试，使用临时目录验证 `--output-html` 写出文件并且文本/JSON/HTML 同源。
- [x] 4.3 更新 `tests/test_cli_help.py`，确认 `kd-sensing-research-dashboard --help` 暴露 HTML 参数。
- [x] 4.4 测试必须使用 synthetic summary、tmp path 或 fixture，不读取真实 `dataset/`、checkpoint 或用户本地训练产物。

## 5. 验证

- [x] 5.1 运行 `openspec validate add-html-evidence-dashboard --strict`。
- [x] 5.2 运行 `openspec validate --all --strict`。
- [x] 5.3 运行 `conda run -n kd_mm_beam pytest tests/test_research_claim_harvester.py tests/test_cli_help.py -q`。
- [x] 5.4 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
