## Context

当前项目已经有 `kd-sensing-research-dashboard`，它能聚合 run index、claim candidates、active OpenSpec change、资源快照和 ledger，并输出文本摘要或 JSON。缺口在于研究协作时经常需要一个可直接打开、截图、归档和人工审阅的静态 HTML 页面；现有文本摘要不适合浏览较多候选，JSON 又不适合快速阅读。

该 dashboard 必须遵守现有诊断边界：只读本地 `outputs/`、`logs/`、OpenSpec 和 claim 相关文档；输出写入 ignored runtime root 或用户显式路径；不得启动训练、清理产物、移动 checkpoint 或修改正式 claim registry。

## Goals / Non-Goals

**Goals:**

- 在现有 dashboard summary 数据结构之上增加静态 HTML renderer。
- 通过既有 `kd-sensing-research-dashboard` CLI 暴露 `--output-html` 或等价参数。
- HTML 与 `--json` / `--output-json` 使用同一 summary，避免两套 dashboard 逻辑漂移。
- 页面离线可打开，展示 run states、active changes、claim readiness、paper export gate、candidate-only caveat 和 next actions。
- 增加 focused tests 覆盖 HTML escaping、空数据、candidate-only 标记和 CLI 输出。

**Non-Goals:**

- 不新增 Web 服务、前端框架、交互式服务器、Gradio 或浏览器自动刷新能力。
- 不把 HTML dashboard 作为正式论文 claim registry，也不自动更新 `docs/result_claims_registry.md`。
- 不生成或提交真实运行图表、checkpoint、CSV、ledger 或 dashboard HTML。
- 不改变 harvester 的 strict comparability gate、ledger schema 或 paper export 逻辑。

## Decisions

1. **使用静态 HTML renderer，而不是前端应用。**

   Renderer 接收现有 dashboard summary dict，返回完整 HTML 字符串。这样实现可以放在 `kd_sensing.diagnostics` 的窄 helper 中，测试不需要启动服务，也不引入新依赖。

2. **CLI 复用现有 `kd-sensing-research-dashboard`。**

   新增 `--output-html PATH`，并在命令结束时打印 `dashboard_html: <path>`。不新增 console script，避免项目入口继续膨胀。

3. **HTML 与 JSON 同源。**

   `run()` 先构建一次 summary，然后根据参数分别写 JSON、HTML、ledger 或文本摘要。HTML 不重新扫描 outputs/logs，避免计数和 hint 不一致。

4. **默认离线和安全转义。**

   HTML 必须使用标准库转义所有来自 run name、path、claim id、warning、hint 和 active change 的文本。页面可以包含内联 CSS，但不得加载外部 CDN、远程 JS 或本地 checkpoint 内容。

5. **输出边界保持 ignored。**

   推荐默认路径位于 `outputs/analysis/research_dashboard/dashboard.html` 或用户显式路径。实现必须只创建父目录和 HTML 文件，不移动或复制输入 artifact。

## Risks / Trade-offs

- **HTML 页面变成第二套业务逻辑** -> renderer 只消费 summary，不重新实现 harvester、run index 或 claim gate。
- **页面展示 candidate 时被误读成正式结论** -> 每个 candidate 区块和页面摘要都必须展示 candidate-only / draft / pending caveat。
- **路径或 warning 中包含特殊字符** -> 所有动态文本必须 HTML escaping，并用单测覆盖。
- **输出文件被误提交** -> 文档和 spec 明确 HTML 写入 ignored output root；测试使用临时目录。
