# 模型任务上下文

用于新增或修改模型、forward、registry、baseline 组件、representation core、head 或模型配置入口。先判断任务类型，再读取对应权威文件。

## 先读

- `openspec/specs/model-architecture-extension-contract/spec.md`
- `openspec/specs/modular-sequence-model/spec.md`
- `openspec/specs/component-registry/spec.md`
- `docs/project_surface_inventory.md` 中的模型扩展路径分类、源码热点模块和 retired route guard
- `docs/model_architecture_inventory.md` 和 README 当前模型说明

## Owner

- 普通模型组件：`src/kd_sensing/models/`、`src/kd_sensing/registries.py`
- shared forward 消费：`src/kd_sensing/engine/batch.py`、`src/kd_sensing/engine/runtime.py`
- workflow 或 paper reproduction：`src/kd_sensing/baselines/`
- 配置选择面：`configs/` 和 `src/kd_sensing/config/`

## 决策

| 类型 | 默认路径 | Caveat |
| --- | --- | --- |
| config-only baseline | 只改 YAML、overlay、virtual recipe 或 hyperparameter | 首选复用 `model.primary.type: modular_sequence` |
| component baseline | 新增或替换 encoder、projector、representation core 或 head | 通过 registry 和 `model.primary` 配置选择 |
| whole-model exception | 新增完整 `@MODELS.register(...)` | 必须有 current spec 或 active change 说明，并覆盖 registry build、synthetic forward、metadata tests |
| workflow reproduction | 官方协议、多阶段训练、feature cache 或特殊报告 | 放在 `src/kd_sensing/baselines/<family>/` 或包内 CLI，不复制通用训练循环 |

## Retired guard

旧 KD、HiST/Hist、Top8 selector、GPS residual、camera residual、BGAM、viewer manifest、AMR-Net_gps_image 和 JEPA-MSAC 只能作为退役、防回流或 migration guard 说明出现，不得恢复为 registry、CLI、实体 YAML 或兼容 wrapper。

## 验证

- `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`
- `conda run -n kd_mm_beam pytest tests/test_component_registry.py -q`
- 触碰 forward、runtime metadata 或 objective 时追加对应 focused tests，例如 `tests/test_prediction_objectives.py`、`tests/test_evaluation_pass.py`
