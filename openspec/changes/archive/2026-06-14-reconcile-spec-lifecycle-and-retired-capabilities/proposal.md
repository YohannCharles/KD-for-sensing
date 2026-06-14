## Why

当前 `openspec/specs/` 同时包含当前能力、支撑能力和退役墓碑能力，且部分旧 `project-architecture` 文案仍把 HiST-Beam、Raymobtime 等退役路线描述为 active mainline 或当前热点。语法校验和架构测试已经通过，但这些语义冲突会让 AI agent 和维护者误判当前支持面，尤其是在无 active change、但工作树存在已归档未收口内容或本地缓存噪声时。

## What Changes

- 新增 OpenSpec spec 生命周期分类契约，明确 `current`、`supporting`、`retired-tombstone` 等状态的语义、标记位置和允许/禁止 wording。
- 清理 `project-architecture` 中已经退役路线的 active wording，使 HiST/Hist、Raymobtime s008、Top8 selector、GPS residual、camera residual、CRAF/MARF/G2D/Multimodal-NF 等只作为退役边界或 migration guard 出现。
- 更新 AI/maintainer navigation，使 agent 在读取当前 specs 时优先识别 capability lifecycle，不因 spec 文件名或 archive 目录存在而误判当前入口。
- 扩展项目健康护栏，检查当前 specs 的 lifecycle marker、退役墓碑 spec 的 wording、当前 spec 中旧 active mainline 说法，以及 ignored runtime/cache 状态不应被当作源码需求。
- 更新 project surface inventory 和相关文档分类，记录退役墓碑 spec 的用途，避免后续把它们重新描述成当前推荐入口。
- 不新增训练、评估、预处理、诊断 CLI，不恢复任何退役实现或兼容 wrapper，不修改模型数值语义、数据 split、配置解析 runtime 或本地产物清理行为。

## Capabilities

### New Capabilities

- `spec-lifecycle-boundaries`: 定义 OpenSpec capability 生命周期分类、退役墓碑 spec 的允许语义、当前/supporting 能力的 wording 约束，以及 AI/维护者读取 current specs 时的判定规则。

### Modified Capabilities

- `project-architecture`: 清理退役路线仍被描述为 active mainline、当前热点或当前推荐入口的要求，改为只保留当前架构边界、退役禁止回流和必要支撑代码边界。
- `ai-maintainer-navigation`: 增加 spec lifecycle 的读取规则、墓碑 spec 识别规则、已归档未收口 change 与 ignored cache 噪声的判断规则。
- `project-health-guardrails`: 增加生命周期标记、退役墓碑 wording、当前 spec active wording 漂移和本地缓存误读相关的健康检查要求。

## Impact

- 主要影响 OpenSpec artifacts、README/docs 中的文档生命周期说明、`docs/project_surface_inventory.md` 和架构边界测试。
- 可能触碰 `tests/test_architecture_boundaries.py` 中的文档/规格扫描规则，但只用于静态检查源码、文档和 OpenSpec，不读取真实 `dataset/`，不写入 `outputs/`、`logs/`、cache 或 checkpoint。
- 不改变 `src/kd_sensing` runtime 行为、训练/评估 CLI、pyproject console scripts、配置加载最终语义、checkpoint schema 或已有本地产物。
