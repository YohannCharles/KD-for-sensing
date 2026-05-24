## Context

项目已经把训练、评估和预处理统一到配置驱动入口，并且运行目录中保存了完整解析配置、启动摘要、训练日志、metrics、checkpoint sidecar 和 TensorBoard event。实际使用中，后台并行训练还依赖 tmux/tee 日志、`ps` 和 `nvidia-smi` 才能判断状态；当进程被系统 `Killed` 或等待前置 checkpoint 时，run 目录往往只留下部分 artifact。

运行索引需要服务两个人群：一是正在并行跑矩阵的研究者，需要快速看哪些 run 还活着、哪些已经失败；二是后续复现实验的人，需要从本地 ignored 产物中恢复配置、指标、checkpoint 和失败原因。

## Goals / Non-Goals

**Goals:**
- 提供只读 run index，能扫描本地 `outputs/` 和 `logs/` 生成结构化状态。
- 支持状态推断、失败原因摘要、资源快照、指标摘要和 checkpoint 摘要。
- 给训练/评估入口增加轻量状态 sidecar，提升后续索引准确性。
- 保持所有项目 Python 命令使用 `conda run -n kd_mm_beam`。

**Non-Goals:**
- 不实现长期运行的 daemon、数据库、Web 服务或任务调度器。
- 不终止、暂停、恢复或重启用户训练进程。
- 不把 `outputs/`、`logs/`、checkpoint 或 TensorBoard 产物纳入源码变更。
- 不改变现有训练、评估、预处理的算法语义。

## Decisions

1. 采用“扫描器 + 可选状态 sidecar”的双路径。
   - 扫描器读取已有 artifact 和日志，能立即覆盖历史 run。
   - 新训练/评估写 `run_status.json` 或等价 runtime status 字段，提高未来 run 的状态准确性。
   - 备选方案是只依赖训练入口写状态，但无法解释已有 partial/killed run。

2. 状态推断使用分层规则。
   - 优先级：当前进程匹配 > 明确 status sidecar > metrics/train_log/checkpoint 完整性 > 日志失败模式 > artifact 完整性。
   - `Killed`、`Traceback`、`ERROR conda.cli.main_run`、等待 checkpoint shell 等日志模式只作为原因证据，不直接修改 run 目录。
   - 对无法确认的目录输出 `unknown`，同时列出缺失 artifact。

3. CLI 先提供 JSON/表格/CSV，不引入 UI。
   - 推荐入口：`kd-sensing-runs status --outputs outputs --logs logs`。
   - JSON 作为测试和后续 viewer/报告集成的稳定接口，表格用于日常终端查看。

4. 资源快照是可选增强。
   - 如果系统有 `nvidia-smi`，索引器记录 GPU index、显存、利用率和进程 PID 关联。
   - 系统内存、swap 和进程 RSS 通过标准 Linux 命令或 Python `/proc` 读取；不可用时安全降级。
   - 不把资源快照作为状态判断的唯一依据。

## Risks / Trade-offs

- [Risk] 日志格式来自 tmux、tee、conda 和 shell，可能不稳定。→ Mitigation：日志解析只输出 best-effort reason，并用测试覆盖常见 `Killed`、Traceback、waiting pattern。
- [Risk] `ps`/`nvidia-smi` 只能看到当前机器，不适合跨机器历史。→ Mitigation：资源快照字段可为空，历史状态仍可由 artifact 推断。
- [Risk] 新 sidecar 写入训练异常路径可能遗漏 `SIGKILL`。→ Mitigation：索引器仍必须通过日志和 partial artifact 分类 `killed` 或 `stale`。
- [Risk] 运行索引可能变成调度系统。→ Mitigation：本 change 明确非目标，不提供 kill/resume/retry 命令。

## Migration Plan

1. 先实现扫描器和纯函数状态推断，覆盖已有 run。
2. 新增 CLI/脚本入口和文档，默认只读。
3. 在训练/评估入口开始和正常结束时写状态 sidecar；异常路径尽量捕获 Python exception 并写失败状态。
4. 增加 tests，确认旧 run 目录没有 sidecar 时仍能分类。

回滚策略：删除 CLI 入口和 sidecar 写入调用即可；已有 run artifact 不需要迁移。

## Open Questions

- CLI 是否作为正式 console script `kd-sensing-runs`，还是先放在 `tools/analysis/run_index.py`？
- 是否需要把状态汇总接入 TensorBoard text 或 Gradio viewer？本 change 暂不做。
