## 1. 工具适配入口

- [x] 1.1 选择首批需要落库的适配面：至少评估 Claude、GitHub Copilot、Cursor、Kiro 和 Replit/Lovable/Bolt 风格项目知识入口。
- [x] 1.2 新增薄适配文件或模板，统一引用 `AGENTS.md`、`docs/agent_navigation.md` 和 scoped context。
- [x] 1.3 确认适配文件不复制完整任务路由、完整 retired token 清单或完整 OpenSpec requirement。

## 2. 当前研究简报

- [x] 2.1 新增 `docs/current_research_brief.md` 或等价短文档。
- [x] 2.2 记录当前主线、冻结方法、不要追的路线、claim 升级条件和下一步高价值实验。
- [x] 2.3 在 README、agent navigation 或 inventory 中登记简报职责，说明它不替代 claim registry 和 experiment protocols。

## 3. 记忆候选与复盘

- [x] 3.1 新增 agent mistake/retrospective ledger 或等价记录位置。
- [x] 3.2 定义字段：错误模式、触发场景、正确规则、建议沉淀位置、验证命令、人工确认状态。
- [x] 3.3 更新文档边界，禁止 hook 自动重写 README、OpenSpec、AGENTS 或正式 claim 文档。

## 4. 只读角色 agent / skills

- [x] 4.1 定义首批只读角色：claim auditor、experiment triage、surface doctor reviewer、literature scout。
- [x] 4.2 确认角色说明只返回建议，不直接写文件、不启动训练、不清理产物。
- [x] 4.3 如实现为 skills 或工具专属 agent 文件，补充 discovery/trigger 描述和 OpenSpec 边界。

## 5. 验证

- [x] 5.1 增加 focused 文档健康或架构边界检查，覆盖适配文件引用、`kd_mm_beam`、退役入口和只读角色边界。
- [x] 5.2 运行 `openspec validate add-agent-context-portability --strict`。
- [x] 5.3 运行 `openspec validate --all --strict`。
- [x] 5.4 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
