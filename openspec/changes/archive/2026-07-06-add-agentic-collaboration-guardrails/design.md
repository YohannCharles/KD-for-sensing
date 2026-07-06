## Context

热门 coding agents 已经从“本地改文件”扩展到 Issue/PR、自动 review、后台任务、Slack/Linear/Jira 和 CI。这个仓库目前更强在本地规则和 OpenSpec，但协作入口较薄：CI 只跑 quick verify，PR/Issue 模板没有把 OpenSpec/change/验证/产物边界结构化，脏工作树和 completed change closeout 主要靠人工判断。需要补一层协作护栏，让 agent work 更容易被审查和回滚。

## Goals / Non-Goals

**Goals:**

- 让每个 PR/Issue 能显式说明 change id、范围、验证、产物边界和 claim 状态。
- 增加分层 CI/doctor/security 检查，但保持默认 CI 无数据、无训练。
- 防止 secrets、checkpoint、dataset、outputs、系统启动文件和退役入口回流。
- 在 agent 开始或结束任务时报告工作树/change 状态，帮助收口。
- 允许 AI review 作为额外信号，不替代人类 review 和现有 tests。

**Non-Goals:**

- 不要求云 CI 跑 GPU/full training。
- 不自动 archive complete changes。
- 不自动删除/移动本地产物。
- 不把 GitHub/Codex/Copilot 作为唯一工作流前提。

## Design

### 1. Issue and PR templates

模板字段应结构化但不繁琐：

- OpenSpec change id 或“窄修复无需 change”的说明。
- 触碰范围：model/data/config/cli/diagnostics/docs/claims/outputs。
- 验证命令和未运行原因。
- 是否读取真实 dataset 或写 ignored outputs。
- 是否影响 claim、paper table 或 dashboard candidate。
- 是否新增入口、脚本、config、checkpoint、依赖或环境变量。
- 回滚/停止条件。

### 2. Layered CI

建议 CI 分层：

- PR/push quick：OpenSpec strict + architecture boundaries。
- PR optional/manual：CLI/config smoke、compile、doctor。
- Scheduled/nightly：doctor all scopes、retired route scan、unclassified config/hotspot report、secret/artifact scan。
- Manual heavy：用户显式触发，仍默认不跑真实训练，除非给出环境和数据前提。

CI 命令必须能在无真实 dataset 和无 GPU 环境下运行。

### 3. Security and artifact scans

安全扫描重点匹配本项目风险：

- 禁止提交 secret/token/private key。
- 禁止修改系统启动/认证文件或把训练命令写入 `/root/.container_env` 等路径。
- 禁止 tracked checkpoint、TensorBoard event、outputs、logs、cache、dataset 真实内容。
- 检查 shell runner 是否包含危险 destructive 命令、裸 `rm -rf`、无确认删除、后台长跑污染系统配置。
- 依赖审计可先作为 warning/manual，不阻塞研究本地环境。

### 4. Dirty worktree and change closeout preflight

preflight/doctor 不应清理或重写，只报告：

- 当前 `openspec list --json` active changes。
- active change 是否 complete。
- 是否存在 untracked archive 或 deleted active + new archive 的成对状态。
- git dirty/untracked 文件分类：source/docs/specs/tests/scripts vs ignored artifacts。
- 是否有未分类 config/script/hotspot warning。

报告应给出下一步：archive、记录 deferral、继续实施、忽略 unrelated user changes 或先提交。

### 5. Agent review boundary

AI review 可以检查 regressions、missing tests、security、artifact boundary、claim caveat 和 OpenSpec drift。它不得：

- 自动 merge。
- 自动升级 claim。
- 自动清理 outputs。
- 自动恢复退役入口。
- 在无用户许可时修改系统/凭证文件。

## Risks / Trade-offs

- [Risk] CI 变慢，影响快速迭代。  
  Mitigation: 保留 quick 默认层，其它层 manual/nightly 或 path-filter。
- [Risk] 安全扫描误报科研数据路径。  
  Mitigation: 初期以 warning/manual 为主，明确 allowlist 与 protected paths。
- [Risk] 模板太重导致不填写。  
  Mitigation: 模板短字段，详细说明链接到 docs。
- [Risk] preflight 被误解为会自动收口。  
  Mitigation: 明确只读报告，不 archive、不删除、不 reset。

## Migration Plan

1. 新增 PR/Issue 模板和 agent review prompt 建议。
2. 增加 CI 分层 workflow 或现有 workflow matrix。
3. 增加 security/artifact scan focused helper。
4. 扩展 doctor/preflight 输出脏工作树和 OpenSpec closeout 状态。
5. 更新 README/AGENTS/navigation/inventory。

## Open Questions

- security scan 是否先纳入 CI required check，还是先作为 scheduled warning。
- 是否需要 path-filter，把 docs-only PR 与 source/config PR 的 CI 分层区分开。
