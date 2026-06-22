## 1. 基线与候选证据

- [ ] 1.1 记录实现前 `git status --short`，确认本 change 不包含 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 或 `All_models/` 产物改动。
- [ ] 1.2 用 CodeGraph 或 `rg` 确认候选调用方和公开入口风险：退役整模型类、旧 alias、thin CLI wrapper、`parse_simple_yaml`、canonical recipe 小层、dataset descriptor、objective metadata、run-index/cleanup legacy 分支、TinyViT 注册重复和内部 `__all__`。
- [ ] 1.3 读取 `docs/project_surface_inventory.md` 中 entrypoint、config lifecycle、hotspot/right-size 和 retired-route 段落，列出本 change 最小触达文件。
- [ ] 1.4 运行 `openspec validate prune-remaining-overengineered-surface --strict`，先修正 proposal/design/spec/tasks 格式问题。

## 2. 退役整模型类与 registry 表面

- [ ] 2.1 从 `src/kd_sensing/models/gps.py`、`image.py`、`radar.py`、`lidar.py`、`mmwave.py` 删除已退役整模型类和旧 strong/lightweight/teacher/student alias，保留仍被当前模块消费的 feature extractor。
- [ ] 2.2 从 `src/kd_sensing/models/fusion/networks.py` 删除已退役 `FusionTeacherModalityNet`、`FusionStudentModalityNet` 和旧 alias；保留当前 fusion owner 需要的共享 helper 或迁入真实 consumer。
- [ ] 2.3 更新测试：删除只证明退役类 direct-forward 的断言，改为覆盖 `modular_sequence`、当前 feature extractor、registry unknown-name 和 current config build。
- [ ] 2.4 审核 `register_removed()` 调用，删除只服务历史 fixture 的 removed guard，保留高频迁移和防静默误跑 guard。
- [ ] 2.5 运行 `conda run -n kd_mm_beam pytest tests/test_component_registry.py tests/test_student_configs.py tests/test_gps_modality.py tests/test_lidar_modality.py tests/test_mmwave_modality.py -q`。

## 3. 配置解析与 canonical recipe 收缩

- [ ] 3.1 删除 `src/kd_sensing/config/parsing.py` 中手写 `parse_simple_yaml` 和 optional-yaml fallback，使用 `pyyaml.safe_load` 作为唯一 YAML 解析路径。
- [ ] 3.2 删除 `src/kd_sensing/config/io.py` 中 optional-yaml dump 分支，确保 `dump_config()` 使用 `yaml.safe_dump`。
- [ ] 3.3 合并或内联 `src/kd_sensing/config/canonical_recipes/` 中只包装常量表的 dataclass/recipe 小层，保持 `build_virtual_config()` 关键语义不变。
- [ ] 3.4 收缩 `src/kd_sensing/config/migration_guards.py` 中低价值 retired-route scan，保留 current config load 必需拒绝项。
- [ ] 3.5 运行 `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py tests/test_architecture_boundaries.py -q`。

## 4. Dataset runtime 与 objective metadata 小层合并

- [ ] 4.1 简化 `src/kd_sensing/data/dataset_descriptors.py`，复用 `modalities.py` 的 profile/sample key/fusion input key 合约，或记录保留轻量 descriptor 查询的理由。
- [ ] 4.2 将 target-shot split 或 runtime row 消费改为 `Mapping[str, Any]`、flat dict sample 或 owner-local dataclass；删除无多调用方的独立 runtime row framework。
- [ ] 4.3 合并 `src/kd_sensing/engine/objectives/registry.py`、`history.py`、`metadata.py` 中只包装常量表的拆分，保持 objective metadata public helper 行为。
- [ ] 4.4 更新相关 imports、tests 和 docs，不新增兼容 wrapper。
- [ ] 4.5 运行 `conda run -n kd_mm_beam pytest tests/test_prediction_objectives.py tests/test_training_io_workflow.py tests/test_config_load_characterization.py -q -k "objective or target_shot or dataset_descriptor or input_profiles"`。

## 5. Run index 与 runtime cleanup right-size

- [ ] 5.1 收缩 `src/kd_sensing/diagnostics/run_index.py` 中只服务历史目录命名的默认 discovery 分支，保持 current canonical layout、状态分类、filters、JSON/CSV/table 输出和只读契约。
- [ ] 5.2 收缩 `src/kd_sensing/diagnostics/runtime_artifact_cleanup.py` 中低价值历史研究线规则，保留 dry-run manifest、保护边界、显式确认、路径重验证和 legacy archive 基础分类。
- [ ] 5.3 更新 run-index 和 cleanup tests，使其覆盖 current layout、安全保护和执行确认，而不是保活每个历史目录命名规则。
- [ ] 5.4 运行 `conda run -n kd_mm_beam pytest tests/test_run_index.py tests/test_runtime_artifact_cleanup.py -q`。

## 6. Facade、`__all__` 和 thin wrapper 收缩

- [ ] 6.1 删除 3 行 CLI wrapper，并将 `pyproject.toml` console script 直接指向真实 owner `main`。
- [ ] 6.2 将内部源码和测试从 package-level re-export、old facade、aggregate import 迁到真实 owner module。
- [ ] 6.3 删除内部模块中无 public facade 价值的 `__all__` 镜像；保留必要稳定 public API 的最小导出。
- [ ] 6.4 确认 `models/fusion/__init__.py`、`data/__init__.py`、`datasets/__init__.py` 等 package init 不导入重依赖、不注册默认组件、不承载旧 alias。
- [ ] 6.5 运行 `conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_architecture_boundaries.py -q`。

## 7. TinyViT 注册去重

- [ ] 7.1 将 `src/kd_sensing/models/tinyvit.py` 中四个 TinyViT 注册名改为 preset 表循环，保持 registry name、variant、pretrained、pretrained_source 和 metadata。
- [ ] 7.2 确认 unknown TinyViT 名称仍使用 registry 错误风格，scratch/22k 构建行为不变。
- [ ] 7.3 运行 `conda run -n kd_mm_beam pytest tests/test_tinyvit_image_encoder.py -q`。

## 8. 文档、OpenSpec 和健康护栏同步

- [ ] 8.1 更新 `docs/project_surface_inventory.md`，记录删除/合并/保留候选、breaking import changes、保留理由和后续触发条件。
- [ ] 8.2 更新 `docs/agent_navigation.md` 和 `docs/maintainer_context_index.yaml`，确保维护索引只保留无法从 pyproject、真实路径、OpenSpec 或 inventory 推导的最小事实。
- [ ] 8.3 更新 README 或相关 current docs，删除已删 facade、thin wrapper、退役整模型类和手写 YAML fallback 的推荐引用。
- [ ] 8.4 更新本 change 的 delta specs 或任务清单，确保实现中新增/删除的范围与 OpenSpec 一致。
- [ ] 8.5 运行 `openspec validate prune-remaining-overengineered-surface --strict`。

## 9. 最终验证与收口

- [ ] 9.1 运行架构、配置、registry 和 CLI smoke：`conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py tests/test_config_load_characterization.py tests/test_component_registry.py tests/test_cli_help.py -q`。
- [ ] 9.2 运行受影响 focused suite：`conda run -n kd_mm_beam pytest tests/test_runtime_artifact_cleanup.py tests/test_run_index.py tests/test_tinyvit_image_encoder.py tests/test_prediction_objectives.py -q`。
- [ ] 9.3 如多个 wave 都触碰核心 runtime，运行最终回归 `conda run -n kd_mm_beam pytest -q`。
- [ ] 9.4 再次检查 `git status --short`，确认没有本地数据、输出、日志、cache、checkpoint、历史权重或临时验证产物进入源码变更。
- [ ] 9.5 最终说明列出 breaking import changes、删除项、保留项、验证结果、未运行验证和剩余风险。
