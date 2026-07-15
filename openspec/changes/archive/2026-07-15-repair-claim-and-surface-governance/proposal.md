## Why

当前 OpenSpec strict 与 quick tests 虽然绿色，但 lifecycle 测试只验证 inventory 是 spec 集合的子集，未发现 3 个 current spec 和 7 个 MMW 脚本漏登记；claim registry 也缺少 exporter 与统计门禁要求的字段，并存在断裂 claim 外键。治理护栏因此会对真实漂移“假绿”，无法可靠保护后续重构。

## What Changes

- 将 current spec、script、root/current document 与 lifecycle inventory 的检查改为双向集合相等，分别报告 missing、extra、duplicate 和非法分类。
- 补齐当前 MMW/OpenSpec/script/document lifecycle；一次性实验脚本必须明确 owner、保留原因、推荐入口关系和删除条件，未跟踪源码也必须进入 on-disk 验收。
- 将 claim registry 改为 exporter 可验证的结构化列，记录 method、dataset/split、metric/value、status、provenance、caveat、seed_count、baseline、CI/mean-std、comparability 和 stress status；所有 catalog claim id 必须有唯一外键。
- paper exporter 从 denylist 改为 reviewed-status allowlist，缺必填字段、pending、not_comparable、mock/smoke 和 candidate-only 行只能进入 excluded/appendix。
- 清理 AGENTS、agent context、root 复现报告和环境文档中的已删除 CLI、配置与虚假 CI/current hardware 描述；README 增加最短安装与健康检查。
- 修复 Beam active delta 的归档覆盖风险并收口其最后验证任务；明确 final C2 是仓库/模型主线，MMW 是当前 supporting dataset campaign，不混用“主线”含义。
- 让 `verify-full` 真正执行全量 pytest；compile/surface 检查覆盖 owner roots 的 on-disk Python，增加最小 CI，不触碰本地数据或训练产物。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `openspec-document-health`：lifecycle inventory 必须双向完整覆盖 current specs、root/current docs 和 agent context。
- `project-entrypoint-lifecycle`：script inventory 必须覆盖 on-disk scripts 并包含完整生命周期字段。
- `mainline-experiment-documentation`：claim registry schema、外键和主线术语必须机器可校验。
- `paper-artifact-export`：主表使用 reviewed allowlist 和必填 schema gate。
- `project-health-guardrails`：full/compile/CI 验证必须覆盖真实源码表面，不能因未跟踪文件假绿。

## Impact

- 影响 architecture/lifecycle tests、inventory、claim/docs、paper exporter、README/AGENTS/agent context、Makefile、verify scripts 和 CI 配置。
- 不恢复任何退役 CLI、wrapper、YAML 或 package facade；失效 root 文档优先删除或降级为 concise historical note。
- 不从 ignored `outputs/`、`logs/` 或本地 PDF 推断 claim，也不提交真实 metrics、checkpoint、图表或运行产物。
