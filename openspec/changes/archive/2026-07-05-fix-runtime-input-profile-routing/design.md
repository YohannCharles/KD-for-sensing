## Context

`prepare_task_inputs` 是训练、验证和评估共享的输入准备入口。Fusion 分支已经通过 `input_profiles` 字典按 modality 传入 profile；单模态分支也应遵循同一规则。由于该函数位于 shared runtime，错误 profile 不一定在 smoke test 中立刻暴露，但会让配置语义和实际 tensor preparation 逐渐漂移。

## Goals / Non-Goals

**Goals:**

- 让每个单模态任务只读取 `model_cfg.input_profiles[<task>]`。
- 保持 fusion、label、model output adaptation 和 evaluation pass 行为不变。
- 用 synthetic batch focused tests 锁住 profile 路由，不依赖真实数据。

**Non-Goals:**

- 不新增 input profile 类型。
- 不重构 batch preparation、difficulty pipeline 或 modality registry。
- 不改变任何 config 默认 profile 或模型 architecture。

## Decisions

1. 直接修复 shared runtime，而不是在 config validation 中补偿。
   - 理由：问题发生在 runtime 调用点，修复位置小且可通过 focused test 直接覆盖。
   - 备选：在 config loader 中重写错位字段；这会掩盖 runtime bug，也难以覆盖手写 `model_cfg`。

2. focused tests 使用 monkeypatch 或轻量 synthetic tensor 观察 profile 参数。
   - 理由：不需要真实 dataset，也不需要启动训练。
   - 备选：跑完整 CLI smoke；成本更高，且难以定位 profile 透传错误。

3. 健康护栏只记录验证命令和禁止错位的契约，不新增长期 CLI。
   - 理由：该问题属于 runtime contract，不需要新的用户入口。

## Risks / Trade-offs

- [Risk] 某些历史配置无 `input_profiles` 字段。→ Mitigation: 继续允许缺省 profile，修复只影响存在字段时的 key 选择。
- [Risk] 测试通过 monkeypatch 过度依赖实现细节。→ Mitigation: 测试只断言 public helper 收到的 profile 值，不绑定内部行号。
- [Risk] 后续新增 modality 忘记添加 profile test。→ Mitigation: 在 health guardrail/spec 中要求新增单模态 runtime profile 分支时同步 focused test。

## Migration Plan

- 直接修复 runtime key 选择。
- 运行 focused test 和架构边界测试。
- 如发现某 config 依赖旧错位行为，应视为配置 bug 并修正为同名 modality profile。

## Open Questions

无。
