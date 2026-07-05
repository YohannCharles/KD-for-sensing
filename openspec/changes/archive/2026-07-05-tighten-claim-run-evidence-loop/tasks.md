## 1. Claim doctor

- [x] 1.1 定义 claim doctor 输入：claim registry、candidate ledger、run index、summary artifacts。
- [x] 1.2 实现缺失字段检查，覆盖 seed、split、metric profile、label space、checkpoint、difficulty digest 和 stress provenance。
- [x] 1.3 输出 upgradable candidates 和 next action hints，但不自动修改 docs。

## 2. Run card

- [x] 2.1 定义 run card JSON schema 和 Markdown 输出。
- [x] 2.2 从 run index、config、metrics、checkpoint sidecar 和 git 状态填充 run card。
- [x] 2.3 确保 run card 写入 ignored output root，且不包含真实 checkpoint 内容或凭证。

## 3. Paper export gate

- [x] 3.1 强化 paper export 主表过滤，硬排除 pending/mock/historical/upper-bound/not_comparable/unverified/candidate-only。
- [x] 3.2 增加 excluded report，列出排除 claim id、status 和 caveat。
- [x] 3.3 增加 focused tests 覆盖主表 gate 和 appendix/diagnostic 显式导出。

## 4. Dashboard

- [x] 4.1 扩展 `kd-sensing-research-dashboard` 或等价只读入口，输出 paper readiness。
- [x] 4.2 汇总 active changes、run states、claim counts、upgradable candidates 和 next actions。
- [x] 4.3 确认 dashboard 不自动写 current docs。

## 5. 验证

- [x] 5.1 运行 `openspec validate tighten-claim-run-evidence-loop --strict`。
- [x] 5.2 运行 `conda run -n kd_mm_beam pytest tests/test_research_claim_harvester.py tests/test_paper_artifact_export.py tests/test_run_index.py -q`。
- [x] 5.3 如修改 CLI，追加 `conda run -n kd_mm_beam pytest tests/test_cli_help.py -q`。
