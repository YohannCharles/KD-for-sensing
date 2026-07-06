## Context

热门 app-builder 的强项是“描述 -> 预览 -> 修复 -> 发布”的短反馈循环。这个仓库是研究/训练项目，不能照搬自动部署和自动数据库迁移，但可以借鉴其反馈结构：每个研究 workflow 都应有一个低成本、可重复、可离线查看的证据闭环。当前已有 research dashboard、paper export、project surface doctor、run index 和多类 summary 脚本，但缺少统一的 happy path、预算记录和预览 QA。

## Goals / Non-Goals

**Goals:**

- 给当前主线/诊断提供一个安全 happy path。
- 默认执行无训练、无真实数据写入、无 checkpoint 写入的检查/汇总。
- 对生成的 HTML、表格、figure data 和 conclusion draft 做结构化 QA。
- 长训练前显式记录预算、输出边界和停止条件。
- 降低 console script / editable install / `python -m` 差异带来的 agent 失败。

**Non-Goals:**

- 不让一键入口自动启动长时间训练。
- 不把 ignored outputs 提交到源码。
- 不把 dashboard candidate 自动写入正式 claim registry。
- 不替代已有 focused tests、paper export 或 run index。

## Design

### 1. Happy path runner

实现阶段应选择一个薄入口，可能是 Make target、package CLI 或已登记 local/manual runner。它的职责是编排已有只读或低成本命令：

- OpenSpec / architecture quick verify。
- project surface doctor。
- run index / research dashboard summary。
- paper export dry summary 或 table consistency check。
- 当前主线 checklist / missing evidence summary。

默认模式不启动真实训练，不读取未显式指定的真实 dataset，不加载 checkpoint。若用户指定真实 outputs 或 checkpoint，只允许只读扫描或显式 fresh eval 子命令。

### 2. Preview QA

静态预览 QA 不依赖浏览器服务。它检查：

- HTML 是否离线可读、无远程 CDN/脚本、动态文本 escaping。
- CSV/table 是否包含必需列、claim caveat、candidate-only 标记和 comparable 字段。
- figure data 是否非空、关键 series 存在、方法顺序稳定。
- conclusion draft 是否保留 pending/incomplete caveat，未把 candidate 写成 reviewed claim。

若未来需要截图，可用可选工具或 synthetic HTML fixture；默认 focused tests 应只依赖标准库/pytest 和临时目录。

### 3. Experiment budget manifest

长跑前的 budget manifest 至少包含：

- workflow/change id。
- config 或 manifest path。
- dataset family 和是否读取真实 dataset。
- GPU/CPU 需求、预计时长、并行度。
- 输出 root、checkpoint 计划、cache 计划。
- fresh eval / paper export 是否会运行。
- 停止条件、失败后处理和不提交产物声明。

该 manifest 可作为 dry-run 输出或 tracked 模板；真实运行生成的实例应写入 ignored `outputs/` 或用户显式路径。

### 4. Run recipe and environment fallback

实现应解决 console script 不在 PATH 时的低摩擦失败：

- README/导航保留 console script 主路径。
- smoke 验证可记录 `conda run -n kd_mm_beam python -m kd_sensing.cli.<owner>` fallback。
- CI/smoke 环境和 GPU/full training 环境明确分层。
- 环境 recipe 不包含本地路径、凭证、真实数据、checkpoint 或平台内部配置文件修改。

### 5. Documentation shape

README 只保留最短 happy path 和链接。详细矩阵、预算字段、预览 QA 字段放到 `docs/` 或 OpenSpec。避免把研究手册塞回 README。

## Risks / Trade-offs

- [Risk] 一键入口被误解为“自动训练全部实验”。  
  Mitigation: 默认只做 dry-run/summary/QA；真实训练必须显式 opt-in。
- [Risk] 预览 QA 变成脆弱快照测试。  
  Mitigation: 检查结构字段和 caveat，不检查像素级样式。
- [Risk] budget manifest 增加使用成本。  
  Mitigation: 对短 smoke 可自动生成默认 budget；长跑才要求显式字段。
- [Risk] 与已有 HTML evidence dashboard change 重叠。  
  Mitigation: 本 change 聚焦 run loop 和 QA，可复用已有 dashboard 输出，不重新定义 dashboard renderer。

## Migration Plan

1. 选定一个当前主线/诊断 happy path，先只编排已有无副作用命令。
2. 增加 budget manifest schema/template 和 dry-run 输出。
3. 增加静态 evidence QA helper 和 synthetic fixtures。
4. 更新 README/docs/navigation/inventory 的最短入口。
5. 增加 focused validation。

## Open Questions

- Happy path 应优先做成 package CLI 还是 Make target。
- 是否将 Playwright/浏览器截图列为 optional local validation，还是先只做 HTML/CSV/figure-data 结构检查。
