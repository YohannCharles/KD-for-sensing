## MODIFIED Requirements

### Requirement: current OpenSpec 必须保持显式且隔离

`openspec/specs/` MUST 保留声明中的 current capabilities。完成或停止的 change MUST 从工作树移除；历史由 Git 和仓库外快照追溯，不得保留 `openspec/changes/archive/`。PCPF-T 实施期间 MUST 是唯一 active change。

#### Scenario: 枚举 current OpenSpec

- **WHEN** 维护者查看 `openspec/specs/` 和 `openspec/changes/`
- **THEN** 只能看到声明的 current specs 与至多一个 PCPF-T active change
- **AND** PCPF-T change MUST 可单独执行 strict validation

## ADDED Requirements

### Requirement: PCPF-T 只能扩大本地研究面

PCPF-T MAY 增加窄模型/loss owner、registry entry、focused tests、`tools/configs/pcpf/` 和本地 runner/evaluator。README 和维护导航 MUST 将其描述为 active research mainline，同时继续将 U0、AMBER-Full、RMBP-MM、DeepSense6G 与正式 MMW protocol 标记为保留基础。它 MUST 不增加 public CLI、canonical `configs/mmw/` recipe、兼容聚合层或绕过 `src/kd_sensing` 的运行入口。

历史 sparse CSI 扩展 MAY 复用 `src/kd_sensing/channel/` 中的 simulator、codebook 和 cache owner，但 MUST 通过 PCPF 本地 opt-in dataset sidecar 接入；不得把第五模态加入全局 modality registry、不得复制 TSPC trainer/runner，也不得把本地 channel/cache 产物纳入源码变更。

#### Scenario: 检查 PCPF-T source surface

- **WHEN** 维护者审计 PCPF-T 变更
- **THEN** public CLI 数量和 canonical MMW recipe MUST 保持不变
- **AND** PCPF-T 运行 MUST 通过共享 trainer、config loader、registry 和数据协议

### Requirement: 历史失败分支必须按 owner 成组退出

非 PCPF 本地实验模型、runner、config、脚本、诊断与测试 MUST 成组移除。clean/trajectory 数据协议、PCPF sparse-CSI 直接依赖、dataset、`outputs/`、`cache/`、日志和 checkpoint MUST 不因源码收敛被删除或迁移。

#### Scenario: 收敛失败实验

- **WHEN** 维护者归档历史 change 并删除失败 owner
- **THEN** 当前 source tree MUST 不残留孤立 import、runner、config 或测试
- **AND** PCPF-T 与保留 stable baseline 的 focused import/test MUST 继续通过


### Requirement: PCPF-T 产物必须保持本地边界

Resolved config、risk statistics、gate report、diagnostics、checkpoint 和 smoke 输出 MUST 写入 `outputs/pcpf_temporal_risk/` 或调用方指定的 ignored output 目录，不得作为源码、OpenSpec 证据或 tracked fixture 提交。

#### Scenario: 执行 PCPF-T smoke 或评估

- **WHEN** runner 生成任何运行产物
- **THEN** tracked source tree MUST 不出现 checkpoint、日志、cache、resolved config 或评估 JSON
