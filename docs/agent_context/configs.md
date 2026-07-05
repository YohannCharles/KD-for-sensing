# 配置任务上下文

用于 YAML、canonical config、virtual config、overlay、migration guard、config load characterization 和实验配置族改动。

## 先读

- `openspec/specs/canonical-config-resolution/spec.md`
- `openspec/specs/project-entrypoint-lifecycle/spec.md`
- `openspec/specs/distillation-free-project-surface/spec.md`
- README 的配置和实验矩阵章节
- `docs/project_surface_inventory.md` 的配置生命周期分类
- `docs/agent_context/atlas.md` 的 config atlas

## Owner

- Config loader 和 migration guard：`src/kd_sensing/config/`
- Current tracked YAML：`configs/`
- Console script 暴露：`pyproject.toml`
- Scene31 local/manual 生成：`scripts/generate_scene31_next_round.py`、`scripts/generate_experiment_grid.py`

## 边界

- 实体 YAML 优先；virtual config 只服务 current recipe，不接管 retired KD、HiST、residual、BGAM 或 viewer route。
- `configs/scene31/` 的 generated YAML 默认不提交；源码长期保留 manifest、base config、generator 和必要 local/manual overlay。
- 新 config family 改变训练流程、数据契约或公共入口时，应先走 OpenSpec change。
- 配置路径可保留熟悉文件名，但语义必须由当前 `model.primary` 和 canonical recipe 表达。

## 验证

- `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py -q`
- `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`
- CLI/config 聚合：`make verify-cli-config`
