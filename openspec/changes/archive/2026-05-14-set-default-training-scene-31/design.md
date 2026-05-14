## Context

项目当前通过 `DeepSenseScene` 描述符、默认配置、YAML 配置和场景化输出目录共同决定 DeepSense6G 训练使用哪个场景。代码里已经存在 Scenario 31 描述符，但默认常量、默认配置、训练配置、KD checkpoint 路径、teacher registry 路径、README 和测试仍以 Scenario 32 为默认。

这次变更跨越配置解析、训练输出、KD 默认依赖路径、文档和测试。目标是让未显式覆盖场景的训练自然落到 Scenario 31，同时保留显式运行 Scenario 9 或 Scenario 32 的能力。

## Goals / Non-Goals

**Goals:**

- 将默认 DeepSense6G 训练场景统一切换为 Scenario 31。
- 让默认数据根目录、运行 metadata、输出分组、resume 路径、最佳 checkpoint registry 和 teacher reliability registry 与 `scene31` 一致。
- 更新训练相关 YAML、canonical 配置、脚本默认路径、README 和测试断言。
- 保留 `data.dataset.scene=9`、`31`、`32` 以及对应 `scene*`、`scenario*` 别名的显式覆盖能力。

**Non-Goals:**

- 不迁移或删除已有 `outputs/scene32` 训练产物。
- 不移除 Scenario 32 支持，也不改变显式 Scenario 32 实验的语义。
- 不为 Scenario 31 重新标定 teacher prior 数值；缺少实测 prior 时不凭空复用 Scenario 32 的可靠性假设。
- 不改变数据集文件格式、split 策略或模型结构。

## Decisions

1. 使用场景描述符作为唯一事实来源。

   将 `DEFAULT_DEEPSENSE_SCENE_ID` 与 `DEFAULT_CONFIG` 中的默认 scene 字段改为 31，并确保 `resolve_deepsense_scene`、metadata 注入、默认 `data_root` 和输出目录都从同一描述符派生。相比只批量替换 YAML，这能避免代码默认值和配置文件默认值再次分叉。

2. 训练默认路径统一切到 `scene31`，显式场景路径保持可覆盖。

   默认 KD `weights_dir`、fusion `registry_path`、Stage 2/3 默认 checkpoint 路径和 teacher registry 脚本默认输出应改为 `outputs/scene31/...`。用户显式提供绝对路径、显式 `data.dataset.scene=32` 或自定义输出目录时，解析逻辑仍按用户输入执行。

3. 配置和文档按用途更新，而不是删除所有 `scene32` 文本。

   训练默认配置、README 默认说明和默认命令示例应切到 `scene31`。保留用于“显式 Scenario 32”、跨场景比较、历史产物说明或测试夹具的 `scene32` 引用，避免把非默认场景支持误删。

4. 验证以配置解析和轻量训练入口 smoke test 为主。

   实现后优先运行 `conda run -n kd_mm_beam pytest` 中覆盖配置解析、scene metadata、checkpoint registry 和 dry-run 训练的测试。完整真实训练依赖本地数据和算力，不作为本变更的必要验证。

## Risks / Trade-offs

- Scenario 31 本地数据或 split CSV 不存在 -> 通过配置解析测试验证默认路径，真实训练前由用户准备 `dataset/scenario31`。
- `scene32` 字符串替换过度影响显式 Scenario 32 示例 -> 按训练默认路径、分析夹具、跨场景诊断分别审查，保留非默认用途引用。
- Scene 32 专属 manual prior 被误用于 Scene 31 -> 不复用 Scene 32 专属 prior；需要 Scene 31 prior 时应由独立 registry 或显式配置提供。
- 大量 YAML 和测试断言更新可能遗漏 -> 用 `rg "scene32|scenario32|scene: 32"` 复查剩余引用，并逐项判断是否属于显式 Scenario 32 或历史引用。

## Migration Plan

- 更新默认场景常量、默认配置和训练相关 YAML。
- 更新默认 checkpoint/registry 路径与 README。
- 更新 OpenSpec delta 对应测试。
- 保留旧 `outputs/scene32`，需要回滚时可将默认常量和默认配置路径恢复为 32；显式 `data.dataset.scene=32` 始终可继续运行。

## Open Questions

- Scenario 31 是否已经有完整的单模态 teacher checkpoint 和 teacher reliability registry。如果没有，KD/fusion 默认配置切到 Scene 31 后需要先训练或生成这些默认依赖。
