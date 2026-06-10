## 1. 消除确定性漂移

- [x] 1.1 运行并记录当前快速红点：`conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`、`find configs/fusion -maxdepth 1 -name '*.yaml'` 和已知过期引用扫描。
- [x] 1.2 修复 `openspec/specs/gps-conditioned-jepa-pretraining/spec.md` 的脚手架 Purpose，使当前 spec hygiene 不再命中 `TBD`。
- [x] 1.3 修复 `scripts/run_csi_hardening_matrix.sh` 中不存在的 hardening matrix 配置引用，并用只读路径检查确认脚本默认配置存在。
- [x] 1.4 更新 `docs/project_surface_inventory.md` 中 `configs/fusion/` 数量、脚本入口和已退役/保留边界，使 inventory 与真实仓库一致。

## 2. 批量收缩 Fusion 配置面

- [x] 2.1 审计 `configs/fusion/*.yaml`，将每个配置分类为 canonical 保留、当前文档推荐、实验特化迁移、复现归档或删除。
- [x] 2.2 批量迁移或删除实验特化、重复低内存、best/last 对照和一次性矩阵配置，使 `configs/fusion/` 根目录回到架构 guardrail 允许范围。
- [x] 2.3 同步更新 README、docs、OpenSpec、scripts 和 tests 中指向被迁移或删除配置的引用，确保不再声明旧路径为当前支持入口。
- [x] 2.4 更新 `tests/test_architecture_boundaries.py` 中与配置根目录、inventory 和 allowlist 对应的 guardrail，使测试继续限制根目录增长而不是简单放宽。

## 3. 删除确认冗余源码与低风险本地产物

- [x] 3.1 使用 CodeGraph 或等价结构检查确认无调用候选，包括 `src/kd_sensing/evaluation/flops.py`、`src/kd_sensing/evaluation/latency.py`、LOSO loader helper 和 transform cache re-export 的真实引用面。
- [x] 3.2 对不属于 console script、公开导出、注册入口、README/docs/OpenSpec 声明或测试依赖的候选，直接删除源码或函数，并同步移除相关测试/文档引用。
- [x] 3.3 对无法排除外部依赖的候选保留源码，但在 `docs/project_surface_inventory.md` 记录保留原因和后续单独退役条件。
- [x] 3.4 清理 ignored 的低风险本地产物：`__pycache__`、`.pyc`、`.pytest_cache`、`src/kd_sensing.egg-info`、空的 `tools/analysis/__pycache__` 和明确临时备份；不自动删除 `outputs/`、`logs/`、cache、checkpoint、dataset 或历史权重。

## 4. 验收与收尾

- [x] 4.1 运行 `openspec validate cleanup-project-surface-drift --strict`。
- [x] 4.2 运行 `openspec validate --all --strict`。
- [x] 4.3 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
- [x] 4.4 如修改 CLI 或脚本入口，运行对应无副作用 help/smoke，例如 `conda run -n kd_mm_beam kd-sensing-export-viewer-manifest --help` 和 `conda run -n kd_mm_beam kd-sensing-visualize-modalities --help`。
- [x] 4.5 汇总已删除/迁移/保留项，说明未自动删除的实验输出候选和后续 archive 条件。
