## Context

当前架构测试从 `docs/project_surface_inventory.md` 读取 lifecycle，但只断言 inventory 中的 capability 存在于 current specs，方向相反使新增 spec 漏登记仍通过。script 检查只核对 console scripts，`verify_compile.py` 又只读取 `git ls-files`，因此 on-disk 未跟踪实验脚本同时绕过分类与编译。claim registry 的 Markdown 表只有五列，paper exporter 用排除状态 denylist；空状态和字段不全的行可以进入主表，catalog 也没有 claim foreign-key guard。

## Goals / Non-Goals

**Goals:**

- current spec、script、root/current document、console entrypoint 与 inventory 双向一致。
- claim schema、claim id 外键和 paper main-table gate 可机器验证。
- current 文档只引用真实入口，并清楚区分仓库主线与 supporting dataset campaign。
- `verify-full`、compile 和 CI 名称与实际覆盖一致。
- active Beam change 在不覆盖 current temporal requirements 的前提下收口。

**Non-Goals:**

- 不重新实现或恢复历史 CLI、YAML、wrapper 或报告 workflow。
- 不把 ignored 输出、PDF、checkpoint、metrics 或图表纳入源码。
- 不把一次性 MMW 脚本自动升级成 public API；只做 lifecycle 判定。

## Decisions

### 1. Inventory 是集合事实源，测试做双向差集

保留现有 Markdown inventory 的结构化表，不新增第二份 YAML 镜像。解析器分别构造 actual 与 declared 集合，断言相等并输出 `missing`、`extra`、`duplicate`、`invalid_lifecycle`。current specs 从 `openspec/specs/*/spec.md` 读取；scripts 从 owner root 的 on-disk `.py/.sh` 读取，使未跟踪新增文件不能绕过验收。

### 2. Script lifecycle 每行携带处置条件

每个保留 script 至少记录 path、lifecycle、owner、保留原因、recommended/public relation、output boundary、focused validation 和 deletion condition。规则化 family 可用明确 glob/prefix 行，但匹配必须无歧义且实际集合仍完全覆盖。一次性 campaign 完成后删除脚本，结论迁入 history/claim，不新建 wrapper。

### 3. Claim registry 继续使用 Markdown，但 schema 固定

为保持人工审阅友好，registry 仍是 Markdown 表；列名固定并由 exporter 与 architecture test 共同解析。pending/not_comparable 行的 `value`、CI 可为空，但必须给 blocker/upgrade gate；可进入主表的 reviewed 行必须包含 method、dataset/split、metric/value、seed_count、baseline、统计摘要、comparability、stress、provenance 和 caveat。

### 4. Paper main table 只接受 allowlist

主表允许状态固定为 `official_reproduction`、`local_strict_validation` 和满足完整统计字段的 `local_experimental`。其它状态无论拼写、大小写或空值都进入 excluded report；`candidate_only=true` 永远不进入主表。相比继续扩充 denylist，allowlist 对未来未知状态默认安全。

### 5. 文档按 current、historical、delete 三类处理

README/AGENTS/current context 只保留真实命令。仍有结论价值的 root 报告在顶部标为 dated historical 并移除推荐命令，或把唯一结论迁入 `mainline_experiment_history.md` 后删除。环境硬件记录改为 dated capture，不声称代表当前机器。

### 6. 验证命令名必须兑现覆盖

`verify-full` 在 quick/CLI/compile 后执行 `conda run -n kd_mm_beam pytest -q`。compile 扫描受控 on-disk owner roots而非只扫描 tracked 文件，同时排除 dataset/outputs/logs/cache。CI 只复用这些入口，不复制另一套命令。

### 7. 主线术语拆成两层

`final C2 / U-MaskBeamJEPA` 保持仓库默认模型/研究主线；MMW/CSI 是 current supporting dataset workflow，其中 MMW 可被称为“当前数据实验 campaign”，不得称为已经替换仓库默认主线，除非另开 transition change。

## Risks / Trade-offs

- [完整 inventory 会增加维护成本] -> 允许无歧义 family pattern，但所有实际路径必须恰好匹配一次。
- [严格 claim schema 使当前主表为空] -> 这是正确结果；pending evidence 留在 registry/excluded report，不用假完整行填充。
- [on-disk compile 会发现用户临时脚本] -> 只扫描声明 owner roots；仓库根临时文件由 surface guard 单独拒绝。
- [CI 安装 Torch 较慢] -> 先使用单一最小 job；不增加 lint/type/coverage 矩阵。

## Migration Plan

1. 先补失败测试，登记当前 spec/script/document 集合与 claim 外键。
2. 更新 claim registry/exporter，再清理 stale docs 和入口说明。
3. 修复 Beam delta 并完成其最后验证；不自动归档仍有运行任务的 MMW change。
4. 更新 verify/CI，运行 full regression 和 OpenSpec strict。

## Open Questions

- 无。未完成的 BPA/CMA 真实训练仍按其 active change 管理，不在治理 change 中伪完成。
