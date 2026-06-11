## 1. 范围审计

- [x] 1.1 生成 Raymobtime s008 删除候选 manifest，覆盖源码、配置、文档、测试、`dataset/Raymobtime/s008`、`outputs/raymobtime_s008`、logs/cache/checkpoint/diagnostic 路径，并记录路径、类型、大小、匹配原因和删除状态。
- [x] 1.2 扫描 `raymobtime_s008`、`Raymobtime`、`s008`、`coord`、`ray` 和 Raymobtime 专用模型/预处理器引用，区分当前支持面、历史 archive 和非目标文本。
- [x] 1.3 确认 `coord/ray` 没有非 Raymobtime 的当前工作流依赖；如发现真实依赖，先更新本 change 的 design/spec 再实施删除。

## 2. 源码删除

- [x] 2.1 删除 `kd_sensing.data.datasets.raymobtime_s008` 及 dataset lazy export/registry 引用。
- [x] 2.2 删除 `kd_sensing.preprocessing.raymobtime_s008*` 相关预处理器、helper、cache/index/beam/ray/path 模块和注册导入。
- [x] 2.3 删除 `kd_sensing.models.raymobtime_s008`、Raymobtime selection 模型注册、Raymobtime LiDAR 3D CNN encoder 和相关 `__all__` 暴露。
- [x] 2.4 删除或改写 Raymobtime 配置规则、validation 分支、migration 提示和 `engine.run_metadata` 中的 Raymobtime metadata 特判。
- [x] 2.5 从 dataset layout/descriptor、modalities、batch 准备和 profile 解析中移除 Raymobtime s008、`coord`、`ray` 和 Raymobtime occupancy/ray profile 支持。
- [x] 2.6 更新包级 lazy imports、registries 和安装元数据，确保当前包导入不要求任何 Raymobtime s008 模块存在。

## 3. 配置、文档和测试

- [x] 3.1 删除 `configs/raymobtime/` 和 `configs/preprocess/raymobtime_s008_*.yaml`，或确保旧路径被退役 guard 拒绝且不作为推荐入口。
- [x] 3.2 删除 `docs/Raymobtime_s008_selection.md`，并更新 README、`docs/experiment_matrix.md`、`docs/research_notes.md`、`docs/project_surface_inventory.md` 中当前支持和健康检查表述。
- [x] 3.3 删除 Raymobtime focused tests，新增或更新退役行为测试，覆盖旧 `raymobtime_s008` dataset/preprocessor/model/config 快速失败。
- [x] 3.4 更新配置测试、评估 smoke、架构边界测试和引用扫描，确保 Raymobtime s008 不再作为当前支持 workflow。

## 4. 数据和产物清理

- [x] 4.1 根据 manifest 清理 `dataset/Raymobtime/s008`，跳过不存在路径、symlink 外部目标和非 Raymobtime 路径。
- [x] 4.2 根据 manifest 清理 Raymobtime s008 专属 `outputs/`、`logs/`、cache、checkpoint、audit 和 diagnostic 产物。
- [x] 4.3 保留 manifest 作为审计产物，记录实际删除、跳过和拒绝删除的候选。

## 5. 验证

- [x] 5.1 运行 `openspec validate remove-raymobtime-s008 --strict`，修复 spec/design/tasks 问题。
- [x] 5.2 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
- [x] 5.3 运行 CLI help smoke：`conda run -n kd_mm_beam kd-sensing-export-viewer-manifest --help`、`conda run -n kd_mm_beam kd-sensing-visualize-modalities --help`。
- [x] 5.4 运行退役 guard 和配置相关 focused tests，确认旧 Raymobtime s008 配置快速失败且错误清晰。
- [x] 5.5 运行最终回归 `conda run -n kd_mm_beam pytest -q`，若环境或数据缺失导致无法完成，记录具体原因和替代验证结果。
