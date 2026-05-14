## Context

当前项目已经有 canonical DeepSense6G 场景选择、模态契约、窄模块 builder、Gradio viewer manifest、canonical fusion 配置矩阵和严格 checkpoint 诊断。但旧兼容入口仍在源码和规格中被要求保留：场景命名 dataset 文件同时承载主实现和场景 alias，builder/transform 聚合 facade 暴露，fusion 配置和模型类名仍保留旧 alias，artifact registry 仍有 checkpoint 目录 fallback。

这些入口已经不再是新增功能所需的主路径。继续保留会让测试和文档持续验证历史行为，降低后续架构收敛速度。

## Goals / Non-Goals

**Goals:**

- 将 DeepSense6G dataset 主实现迁移到场景中立模块，并删除 `scene-specific dataset class alias` 等兼容类与 `scenario9|scenario31|scenario32` dataset type。
- 删除已由窄模块替代的兼容 facade 和旧配置/模型 alias。
- 将 OpenSpec、README、扩展指南和测试从“旧入口继续兼容”改为“旧入口被拒绝或不再出现”。
- 增加静态引用检查，防止新代码重新依赖 legacy facade、旧 dataset type 或 legacy fusion 路径。

**Non-Goals:**

- 不迁移或删除本地 `outputs/`、`logs/`、`dataset/` 和历史 checkpoint 产物。
- 不提供旧 checkpoint 到新模型结构的自动转换。
- 不删除当前仍是 canonical 实验路径的模型注册名，例如 `fusion_teacher`、`fusion_student` 本身；只删除同义 alias、旧配置路径和兼容包装层。
- 不改变 beam prediction label、future slot、cache policy 和 normalization artifact 的核心语义，除非它们只服务旧兼容入口。

## Decisions

1. DeepSense6G dataset 使用场景中立模块承载主实现。
   - 决策：新增或迁移到 `kd_sensing.data.datasets.deepsense6g`，注册名只保留 `deepsense6g`；`data.dataset.scene` 继续选择 `9/31/32` 等场景。
   - 理由：`scenario9.py` 已经不再只服务 Scenario 9，文件名和类名会误导后续开发。
   - 替代方案：保留 `scenario9.py` 作为 thin import wrapper。拒绝，因为用户目标是删除兼容冗余。

2. 旧入口统一失败，不做静默映射。
   - 决策：旧 dataset type、旧 facade import、legacy fusion config alias 和旧模型类名 alias 要么不存在，要么在配置解析阶段抛出包含迁移路径的错误。
   - 理由：静默映射仍然是兼容层，会继续扩大测试矩阵。
   - 替代方案：保留 warning 后转发到 canonical 入口。拒绝，因为 warning 很容易被忽略，且仍需维护旧行为。

3. 先改规格和测试，再删代码。
   - 决策：实施时先更新 OpenSpec active change、README/docs 和测试期望，明确哪些 legacy 行为不再成立，再迁移源码。
   - 理由：当前测试大量直接导入 `scene-specific dataset class alias` 和 `engine.builders`；如果先删代码，失败面会混杂真实迁移错误和旧契约错误。

4. artifact registry 只支持 canonical registry 或显式路径。
   - 决策：删除 checkpoint 目录 fallback；缺少 registry checkpoint 时直接报错并列出 canonical 候选和显式配置方式。
   - 理由：fallback 的目录推断会掩盖场景、run name 和模型结构不匹配。

5. 诊断工作流只保留当前 viewer/export 路线。
   - 决策：删除旧静态可视化兼容命令或让其不再作为安装入口；文档只推荐 `kd-sensing-export-viewer-manifest` 与 Gradio viewer。
   - 理由：兼容命令已不再是主工作流，继续保留会重复 parser、入口和测试。

## Risks / Trade-offs

- [Risk] 旧实验脚本或用户私有代码依赖 `scene-specific dataset class alias`、`engine.builders` 或 legacy config 路径。→ Mitigation：错误信息和文档给出 canonical 替代路径；变更明确标为 breaking。
- [Risk] 删除 checkpoint 目录 fallback 后，KD 配置在没有 registry checkpoint 时更早失败。→ Mitigation：错误信息列出需要训练/归档的 teacher slug，并允许显式 checkpoint 路径覆盖。
- [Risk] 一次性删除多个兼容层会造成测试失败面较大。→ Mitigation：按 dataset、facade、config/model alias、artifact registry、diagnostics 分阶段提交，并在每阶段运行聚焦测试。
- [Risk] active OpenSpec 变更仍声明旧兼容行为。→ Mitigation：实施任务包含更新 active change delta specs/design/tasks，并用 `rg` 扫描 active change 残留。

## Migration Plan

1. 更新 OpenSpec delta、README 和扩展指南，确认所有旧兼容要求被删除或改为迁移提示。
2. 迁移 dataset 主实现到 `deepsense6g.py`，更新 registry/import 位置和测试 helper。
3. 删除兼容 facade 与 legacy alias，更新内部导入到窄模块。
4. 删除 legacy fusion 配置路径、旧模型类名 alias 和相关测试期望。
5. 删除 legacy artifact fallback 和旧诊断入口，更新错误信息。
6. 运行引用扫描与聚焦测试，最后在 `kd_mm_beam` 中运行全量测试。

Rollback 只能通过恢复本变更代码和规格；不提供 runtime feature flag，因为保留 flag 本身会继续形成兼容冗余。

## Reference Handling

- Active OpenSpec 删除说明：改为描述类别和迁移路径，不再把旧入口完整字面量作为推荐命令或运行入口。
- Dataset 引用：迁移到 `deepsense6g` registry 和 `data.dataset.scene`，旧 dataset type 改为拒绝测试。
- Builder/transform 引用：测试和内部代码改为窄模块导入，删除聚合 facade 文件。
- Fusion 引用：删除旧配置别名文件和旧类名 alias，测试改为验证 canonical 注册名与旧入口拒绝。
- Checkpoint 引用：删除目录 fallback 和对应 metadata 字段，只保留 registry 或显式 checkpoint 路径。
- Diagnostics 引用：删除旧 console/script 入口，文档只推荐 manifest export 和 Gradio viewer。

## Open Questions

- 是否要同时删除 `All_models/` 中已跟踪的原代码权重文件，还是仅删除运行时 fallback 并把文件标记为历史资料？本提案默认先不删除已跟踪权重文件，避免把源码清理和大文件归档混在一起。
- 是否保留 `data.dataset.scene` 的字符串别名 `scenario9`、`scenario31`、`scenario32`？本提案默认保留场景别名，因为它们属于 canonical scene 解析，不等同于旧 dataset type。
