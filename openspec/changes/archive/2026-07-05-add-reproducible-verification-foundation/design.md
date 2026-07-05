## Context

当前验证命令分散在 README、AGENTS、agent navigation、inventory 和 OpenSpec specs 中。虽然这些命令准确，但 agent 每次需要重新判断最小集合。项目也记录了 `kd_mm_beam` 环境事实，但没有 tracked 的环境构建入口或 CI baseline。

## Goals / Non-Goals

**Goals:**

- 提供一个可复制的最小 verify 层，适合本地和 CI 运行。
- 记录可重建的环境依赖边界，避免把当前机器状态当成唯一事实。
- 增加轻量 lint/compile，优先发现语法、入口和文档引用漂移。

**Non-Goals:**

- 不要求 CI 跑 GPU 训练或读取真实 `dataset/`。
- 不锁死研究训练环境的所有 CUDA/driver 组合。
- 不替代 focused tests；verify 只是常用入口。

## Decisions

1. verify 分层：`verify:quick`、`verify:cli-config`、`verify:docs`、`verify:full`。
   - 理由：不同改动需要不同成本的验证，符合现有 focused validation 规则。
   - 备选：只有一个全量命令；会让小改动成本过高，也不适合 CI。

2. 环境声明采用最小 tracked 文件加文档说明。
   - 理由：科研 GPU 环境常随服务器变化，完全 lock 可能过度约束；但最小依赖必须可见。
   - 备选：只保留 `ENVIRONMENT.md`；无法让新机器自动构建 smoke 环境。

3. CI 默认只跑无数据、无训练、无 checkpoint 的检查。
   - 理由：GitHub/远程 CI 通常没有项目真实数据和 GPU。
   - 备选：CI 跑全量 pytest；风险是因本地数据或 CUDA 差异产生伪失败。

## Risks / Trade-offs

- [Risk] 新增 CI 在缺 conda 环境时不可运行。→ Mitigation: CI 可使用 pip editable + dev extras 的 smoke path，本地仍记录 `conda run -n kd_mm_beam`。
- [Risk] 环境文件与真实训练服务器漂移。→ Mitigation: 文档明确 CPU/smoke 与 GPU/full training 的边界。
- [Risk] verify 命令成为第二套测试权威。→ Mitigation: verify 只聚合已有 OpenSpec/focused tests，权威仍是 specs、tests 和 docs。

## Migration Plan

- 先新增 verify 聚合和文档引用。
- 再添加环境文件或环境导出脚本。
- 最后接入 CI，只跑无副作用 smoke。

## Open Questions

- CI 使用 conda/mamba 还是 pip editable smoke 环境，需要在实现时根据可用 runner 决定。
