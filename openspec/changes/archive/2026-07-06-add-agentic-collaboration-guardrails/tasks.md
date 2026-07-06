## 1. PR / Issue 协作入口

- [x] 1.1 新增或更新 `.github` Issue/PR 模板，覆盖 OpenSpec change、范围、验证、产物边界、claim 状态和回滚条件。
- [x] 1.2 增加 agent task/review prompt 建议，说明禁止触碰路径、允许读取产物和 expected validation。
- [x] 1.3 文档说明模板不替代 OpenSpec、tests 或 human review。

## 2. 分层 CI

- [x] 2.1 保留 PR/push quick CI：OpenSpec strict + architecture boundaries。
- [x] 2.2 增加 manual/scheduled 或 path-filtered workflow：CLI/config smoke、compile、surface doctor。
- [x] 2.3 记录 CI 不读取真实 dataset、不启动训练、不加载 checkpoint 的边界。

## 3. 安全和产物扫描

- [x] 3.1 增加 secrets/system-config/runtime-artifact scan helper 或测试。
- [x] 3.2 覆盖 `/root/.container_env`、系统 profile、SSH/env 凭证、checkpoint、outputs、logs、cache、dataset 真实内容。
- [x] 3.3 增加 shell runner 危险命令检查，保留 manifest cleanup 的显式确认例外。

## 4. Preflight / closeout

- [x] 4.1 扩展 project surface doctor 或新增只读 preflight，报告 active/complete/archive/untracked change 状态。
- [x] 4.2 报告 dirty tracked/untracked 文件分类，不自动清理、不 reset、不 archive。
- [x] 4.3 在 agent navigation 或 AGENTS 中说明 closeout 风险和推荐收口步骤。

## 5. AI review 集成

- [x] 5.1 记录 Codex/GitHub Copilot 或等价 AI review 的推荐提示和范围。
- [x] 5.2 明确 AI review 只作为附加信号，不自动 merge、不自动升级 claim。
- [x] 5.3 如引入 GitHub Action 或 review workflow，确保无真实数据/训练依赖。

## 6. 验证

- [x] 6.1 运行 `openspec validate add-agentic-collaboration-guardrails --strict`。
- [x] 6.2 运行 `openspec validate --all --strict`。
- [x] 6.3 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
  - 2026-07-05 已运行；失败点来自当前工作树既有未跟踪 `openspec/specs/html-evidence-dashboard/spec.md` 缺少 inventory lifecycle 与有效 Purpose，不属于本 change，未擅自修改。
  - 2026-07-06 复跑通过：30 passed。
- [x] 6.4 如实现 CI/security/preflight helper，运行对应 focused tests。
