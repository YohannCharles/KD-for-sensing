## Context

当前项目已经有四类 baseline 路径：普通 `modular_sequence` 配置、可组合组件、whole-model exception、workflow/paper reproduction。问题不在于文件名是否包含 baseline，而在于 `src/kd_sensing/models/` 和 `src/kd_sensing/baselines/` 的职责容易被误读：

- `models/` 是 registry-backed 模型能力和可组合子组件的 owner，必须能被共享 batch/runtime/summary 处理。
- `baselines/` 是论文复现、外部源码审计、多阶段训练、feature cache、Table 报告等 workflow owner。
- `configs/fusion/` 保存本地可训练 baseline/control 的运行配置。
- `configs/baselines/` 保存外部复现、审计或官方 artifact manifest 类配置。

已有架构测试已经删除 BeamBench 大聚合 facade，并要求直接导入具体 owner；这说明项目方向是收紧 owner 边界，而不是新建一个统一 baseline 大桶。

## Goals / Non-Goals

**Goals:**

- 让文档、package marker 和测试一致表达 baseline/model 边界。
- 防止后续在 `baselines/` 中新增 `@MODELS.register(...)` 或从 `models/` 反向依赖 workflow package。
- 保持当前训练入口、registry 名称、配置路径和 CLI 行为不变。

**Non-Goals:**

- 不把已有 whole-model exception 移到 `baselines/`。
- 不恢复已删除的 baseline facade、旧入口或兼容聚合层。
- 不重写 BeamBench、TII、RMBP-MM、AMR-Net、U-MaskBeamJEPA 的训练逻辑。

## Decisions

1. **按行为而不是名称整理。**
   - 决策：保留 `models/` 作为可训练模型能力 owner，保留 `baselines/` 作为 workflow owner。
   - 备选：把所有名字叫 baseline 的模型搬进 `baselines/`。不采用，因为会混淆 registry build、模型架构摘要、通用 runtime 和 workflow CLI。

2. **用测试守住最容易回流的错误。**
   - 决策：架构边界测试扫描 `src/kd_sensing/baselines/**/*.py`，禁止 registry 注册；扫描 `src/kd_sensing/models/**/*.py`，禁止导入 `kd_sensing.baselines`。
   - 备选：只写文档说明。文档容易漂移，不足以防止后续回流。

3. **只做必要文档和 marker 修改。**
   - 决策：修正 `src/kd_sensing/baselines/__init__.py` 的误导性 docstring，并在维护文档中添加放置规则表。
   - 备选：新增新的目录或大量 README。当前已有 `docs/model_architecture_inventory.md`、`docs/project_surface_inventory.md`、`docs/extension_guide.md`，继续复用即可。

## Risks / Trade-offs

- 放置规则仍依赖人工判断复杂模型属于 whole-model exception 还是 workflow。→ 通过 OpenSpec design 理由和 focused tests 审核新增 `MODELS` 注册。
- 架构测试是静态扫描，不能证明 workflow 本身设计合理。→ 只把它用于阻止明确错误的依赖方向；复杂 workflow 仍走 OpenSpec。
- 不移动现有文件可能看起来“不够大步”。→ 这保留了当前 runtime/registry 兼容性，把大步整理放在规则和守门上，避免新遗留问题。

## Migration Plan

1. 添加 OpenSpec delta requirements。
2. 更新 `baselines` package marker 和维护文档。
3. 增加架构边界测试。
4. 运行 OpenSpec validate 和架构边界测试。
5. 若未来确实要迁移某个整模型，单独按 whole-model retirement / component migration change 执行。

## Open Questions

无。当前规则足以处理已知 `models/` 与 `baselines/` 混用问题。
