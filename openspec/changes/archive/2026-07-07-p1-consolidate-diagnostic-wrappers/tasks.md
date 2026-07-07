## 1. Baseline 与入口映射

- [x] 1.1 运行 `git status --short`，确认本 change 不包含本地数据、outputs、logs、cache、checkpoint 或历史权重。
- [x] 1.2 枚举 predictive visualization、MMW Town GPS v2 plot/compare、training profiling/recommendation 和 `scripts/mmw/` wrapper 的 docs/spec/tests/console script 引用。
- [x] 1.3 为每个 wrapper 记录 consolidated owner、mode 名称、输出保持要求和 focused validation。

Baseline mapping:

- predictive explanatory visualization: old `kd-sensing-predictive-gps-query-visualizations` / `kd_sensing.cli.predictive_gps_query_visualizations` -> `kd-sensing-jepa-gps-shortcut-benchmark --predictive-explanatory-figures`; keep tables/figures/manifest under an explicit explanatory output directory; validate with `tests/test_jepa_gps_shortcut_benchmark.py` and CLI help.
- MMW Town GPS v2 plot/compare: old `kd-sensing-plot-mmw-town-gps-v2` and `kd-sensing-compare-mmw-town-gps-v2` -> `kd-sensing-mmw-town-gps-v2 --mode plot|compare`; keep figure names, `comparison_with_previous.csv`, and `comparison_report.md`; validate with `tests/test_mmw_town_gps_adapter_v2.py` and CLI help.
- throughput profile/recommend: old `scripts/profile_training_io.py` and `scripts/recommend_parallel_training.py` -> `kd-sensing-training-throughput --mode profile|recommend`; keep profiling JSON/CSV fields and recommendation keys; validate with training IO and recommendation focused tests.
- MMW data wrappers: old `scripts/mmw/prepare_town10_skybridge.py` and `scripts/mmw/build_sequence_splits_from_manifest.py` -> `kd-sensing-preprocess --action mmw_town10_skybridge|mmw_sequence_splits_from_manifest`; keep prepared artifact layout and split metadata fields; validate with `tests/test_mmw_town10_preparation.py`, CLI help, and scripts surface doctor.

## 2. Predictive GPS query visualization 合并

- [x] 2.1 将 explanatory visualization 输出接入 predictive JEPA robustness diagnostics bundle 或 owner CLI mode。
- [x] 2.2 删除独立 visualization CLI/console script，或将 module 降为 internal helper。
- [x] 2.3 更新 predictive docs、claim notes 和 tests，明确解释性图不是独立 claim gate。

## 3. MMW Town GPS v2 plot/compare 合并

- [x] 3.1 在 MMW Town GPS v2 owner CLI 中提供 plot 和 compare mode，覆盖旧 CLI 的输入、输出和 help 语义。
- [x] 3.2 删除旧 plot/compare CLI 或 console script，不新增 fallback alias。
- [x] 3.3 更新 `pyproject.toml`、docs、OpenSpec current specs、CLI help tests 和 MMW focused smoke tests。

## 4. Training throughput profiling 合并

- [x] 4.1 将 IO profiling 和 parallel recommendation 收敛到一个 profiling owner 或 package CLI mode。
- [x] 4.2 删除 `scripts/recommend_parallel_training.py` 中重复解析逻辑；必要时让 recommendation 读取 profiling output。
- [x] 4.3 更新 throughput docs/spec/tests，保持 profiling 字段和推荐规则可验证。

## 5. scripts/mmw 薄 wrapper 归位

- [x] 5.1 将 Town10 skybridge preparation 和 manifest split build wrapper 的行为迁到 package preprocess/data owner 或 documented command recipe。
- [x] 5.2 删除只转发参数的 `scripts/mmw/` wrapper；若保留，补 retained-with-reason。
- [x] 5.3 更新 inventory 和 scripts surface doctor allowlist。

## 6. 验证与收尾

- [x] 6.1 运行 `openspec validate p1-consolidate-diagnostic-wrappers --strict`。
- [x] 6.2 运行 `openspec validate --all --strict`。
- [x] 6.3 运行 CLI help、predictive diagnostics、MMW Town GPS v2 和 throughput focused validation。
- [x] 6.4 运行 scripts/hotspots surface doctor。
- [x] 6.5 最终说明列出删除入口、替代命令、保留理由和未运行验证。
