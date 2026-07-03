## 1. 配置矩阵

- [x] 1.1 新增 next-round 配置/manifest 生成逻辑，复用现有 Scene31 overlay 结构。
- [x] 1.2 生成 `configs/scene31/next_round/` 下 P0/P1 es40 YAML 与 manifest。

## 2. 运行与评估脚本

- [x] 2.1 新增 `scripts/run_scene31_next_round.sh`，支持 P0/P1/all、GPU、dry-run、skip/overwrite、训练后 fresh eval 和失败列表。
- [x] 2.2 扩展 fresh eval 配置查找，使 next-round YAML 可被现有 re-eval 路径读取。
- [x] 2.3 新增 `scripts/summarize_scene31_next_round.py`，输出 per-run、mean±std、delta 和 filtered Markdown/CSV。

## 3. Sanity Check 与验证

- [x] 3.1 增加 focused sanity test，校验 run name 与 seed/epoch/λ/sampler/condBTAPA 字段一致。
- [x] 3.2 运行 `conda run -n kd_mm_beam pytest <focused tests> -q`。
- [x] 3.3 运行 `openspec validate add-scene31-next-round-experiments --strict`。
