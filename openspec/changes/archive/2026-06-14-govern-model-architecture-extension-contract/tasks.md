## 1. OpenSpec 与生命周期收口

- [x] 1.1 运行 `openspec validate govern-model-architecture-extension-contract --strict`，修正 proposal/design/specs 的格式或语义问题。
- [x] 1.2 将 `model-architecture-extension-contract` 加入 `docs/project_surface_inventory.md` 的 OpenSpec capability lifecycle 分类，并标记为 `current`。
- [x] 1.3 确认 modified capabilities 的 delta specs 与 proposal 中列出的 capability 完全一致，不遗漏 `modular-sequence-model`、`component-registry`、`ai-maintainer-navigation`、`project-health-guardrails` 和 `project-architecture`。
- [x] 1.4 在 active Scenario D change 未归档的前提下，确认本 change 不覆盖或误改 `add-scenario-d-image-observability-benchmark` 的任务状态和产物边界。

## 2. 扩展文档与 AI 导航

- [x] 2.1 更新 `docs/extension_guide.md` 的 Add a Model 小节，将首个新增 baseline 示例改为 `modular_sequence` 配置或子组件 registry 示例。
- [x] 2.2 在 `docs/extension_guide.md` 中新增 whole-model exception 小节，说明直接 `@MODELS.register(...)` 需要 OpenSpec 理由、forward/output metadata 和 focused tests。
- [x] 2.3 更新 `docs/agent_navigation.md` 的模型 / forward / registry 路由，要求先读 `model-architecture-extension-contract`、`modular-sequence-model` 和 `component-registry`。
- [x] 2.4 更新 `docs/project_surface_inventory.md` 的模型扩展和热点说明，区分 config-only baseline、component baseline、whole-model exception 和 workflow/paper reproduction。
- [x] 2.5 如 README 或主线文档索引提到模型扩展指南，确认其指向更新后的 `docs/extension_guide.md`，且不把整模型注册作为默认建议。

## 3. 模型 metadata 与可组合扩展

- [x] 3.1 审计 `ModularSequenceModel.training_strategy_metadata()`，补齐 projectors、heads、representation core 配置名、conditioned encoder 和 reliability metadata 消费字段。
- [x] 3.2 为提供 `training_strategy_metadata()` 的 encoder/core/head 添加或更新 focused tests，确认 metadata 会被 `ModularSequenceModel` 聚合。
- [x] 3.3 为 whole-model exception 定义最小 metadata 检查 helper 或测试约定，至少覆盖模型注册名、启用模态、架构类别、checkpoint/freeze 策略和 reliability metadata 消费状态。
- [x] 3.4 评估 `ObservabilityAwareFusion` 的长期接入方式：注册为 `REPRESENTATION_CORES`、新增窄 adapter registry，或保留显式 helper，并在设计或文档中记录最终选择。
- [x] 3.5 若实现可组合 adaptive fusion 入口，添加 synthetic forward tests，覆盖 opt-in reliability metadata、普通 baseline 忽略 metadata 和 diagnostics 输出。

## 4. 架构护栏与 focused tests

- [x] 4.1 扩展 `tests/test_architecture_boundaries.py`，检查新增 `@MODELS.register(...)` 整模型注册名必须在 current specs、active change artifact、inventory 或明确 allowlist 中出现。
- [x] 4.2 增加文档健康检查，拒绝 `docs/extension_guide.md` 把直接整模型注册写成普通 baseline 的首选示例。
- [x] 4.3 增加或更新模型 focused tests，验证新增/既有模块化组件 metadata 聚合、registry build、forward 输出和 `adapt_model_output` 兼容。
- [x] 4.4 增加 batch/runtime 回流检查，防止普通 baseline 新增专用 `prepare_*`、`forward_task_model` 或 validation loop 分支来绕开共享 runtime。
- [x] 4.5 确保新增架构护栏只扫描已跟踪源码、配置、文档、OpenSpec artifact 和测试文件，不读取真实 `dataset/`、`outputs/`、checkpoint、cache 或日志。

## 5. 配置与 workflow 边界

- [x] 5.1 审计当前 root/canonical fusion config 和 Vision-Position/JEPA 相关 config，确认普通 baseline 示例仍优先走 `modular_sequence` 或当前明确允许的 existing model。
- [x] 5.2 对 workflow/paper reproduction baseline 保留例外说明，确认 BeamBench AE+GPS 等 workflow 仍标记为 paper/workflow baseline，而不是普通模块化 baseline。
- [x] 5.3 如新增 CLI 或脚本入口，必须同步 `pyproject.toml`、README/docs、`docs/project_surface_inventory.md` 脚本 allowlist 和架构边界测试；若本 change 不新增入口，则明确保持无新增入口。
- [x] 5.4 确认本 change 不恢复旧 KD/HiST/residual/Top8/CRAF/MARF/G2D/Multimodal-NF 路线、不新增兼容聚合层、不修改 runtime output root。

## 6. Validation

- [x] 6.1 运行 `openspec validate govern-model-architecture-extension-contract --strict`。
- [x] 6.2 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
- [x] 6.3 运行模型/registry focused tests，例如 `conda run -n kd_mm_beam pytest tests/test_resnet18_image_architecture.py tests/test_student_configs.py -q`，并按实际触碰范围追加 JEPA、Scenario D 或 observability-aware fusion tests。
- [x] 6.4 如触碰 config normalization 或 canonical recipe，运行 `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py -q`。
- [x] 6.5 在最终说明中记录未运行的长耗时真实训练/benchmark、未新增入口、未读取真实数据和剩余风险。
