## 1. 表面积与入口生命周期

- [x] 1.1 更新 `docs/project_surface_inventory.md`，为 `scripts/mmw/build_sequence_splits_from_manifest.py` 和 `scripts/run_mmw_sunny_modal15_l5p6_h246.sh` 记录 lifecycle、职责、推荐入口关系、输出产物边界和删除/收敛条件。
- [x] 1.2 更新 `tests/test_architecture_boundaries.py` 的脚本入口 allowlist，使 allowlist 与 inventory 保持一致，并拒绝未登记的 Python 或 shell 入口。
- [x] 1.3 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`，确认当前表面积 guardrail 恢复通过。

## 2. Target Sensitive Guard 与 Eligibility Metadata

- [x] 2.1 在 target adaptation loss/runtime 路径中集中实现 sensitive field policy，区分 source、target labeled subset、target unlabeled subset 和 target_test。
- [x] 2.2 为 labeled target radio/path auxiliary supervision 增加显式 opt-in 配置；未 opt-in 时访问 target radio/path sensitive 字段作为训练监督必须失败。
- [x] 2.3 在 adaptation run metadata、adapt log 或 summary 输入中写出 `used_target_*_for_training`、`sensitive_field_policy`、`main_conclusion_eligible` 和 `eligibility_reasons`。
- [x] 2.4 为 label_budget=0、unlabeled target、labeled target radio/path opt-in 和未 opt-in 失败路径添加 focused tests，并使用 `conda run -n kd_mm_beam pytest ... -q` 运行相关测试。

## 3. Sensor-assisted Summary 与 Quick Conclusion

- [x] 3.1 更新 MMW sensor-assisted summary，使每个 run record 包含 enabled sensing modalities、excluded sensitive fields、sensitive usage flags、`main_conclusion_eligible` 和 `eligibility_reasons`。
- [x] 3.2 更新 HiST-Beam quick validation conclusion，排除 `main_conclusion_eligible=false`、target leakage、未授权 target sensitive supervision、prototype no-op 和关键 baseline 缺失的 run。
- [x] 3.3 在 conclusion artifact 中写出 eligible/excluded/inconclusive 计数、exclusion reason histogram 和来源 summary/run artifact 路径。
- [x] 3.4 添加 focused tests，覆盖 ineligible run 不参与胜负判断、baseline 被排除导致比较 `inconclusive`、prototype no-op 不作为有效 prototype 证据。

## 4. Metric Horizon 聚合一致性

- [x] 4.1 将 selected `metric_horizons` 解析和 Top-K/DBA 聚合集中到共享 helper，普通 validation、force-mask subset validation 和 standalone evaluate 共用该 helper。
- [x] 4.2 修正 subset top-level `top1`、`top3`、`top5` 和 DBA/ADBA scalar，确保不再回退到 first valid slot 口径。
- [x] 4.3 在 metrics/report metadata 中记录实际使用的 `metric_horizons` 及默认来源。
- [x] 4.4 添加 focused tests，覆盖 `metric_horizons=[2,4,6]` 时普通 validation 与 subset validation 的 top-level scalar 使用同一聚合口径，并运行 `conda run -n kd_mm_beam pytest tests/test_metric_horizon_selection.py -q`。

## 5. MMW 数据准备公共入口

- [x] 5.1 将 MMW split/radar CSV materialization 提取到公开 package utility、preprocessor 或 CLI 可调用入口，输出 metadata 记录输入 manifest、split 配置、seq_len、num_pred、condition、scenario、样本数和输出路径。
- [x] 5.2 更新 HiST-Beam MMW LOSO preflight，使其只读取已准备 artifact、调用公开 utility 或报告缺失 artifact，不再导入 dataset 私有 `_ensure_*` helper。
- [x] 5.3 缺失 prepared artifact 时，preflight 错误信息必须包含公开准备入口、关键参数和目标输出路径。
- [x] 5.4 添加 focused tests 覆盖公开 utility 生成/校验 metadata、preflight 不导入私有 helper、缺失 artifact 提示可执行修复路径。

## 6. LOSO Executor 拆分

- [x] 6.1 为 `hist_beam_loso_execution` 增加 characterization tests，覆盖 run metadata、summary JSON、quick validation conclusion 和 checkpoint reuse metadata 的关键字段。
- [x] 6.2 提取 preflight 逻辑到窄模块，并保持现有公开 CLI/import facade 兼容。
- [x] 6.3 提取 stage orchestration、summary/conclusion 和 matrix metadata 逻辑到窄模块，新增内部代码优先依赖窄模块。
- [x] 6.4 运行相关 focused tests，确认拆分后公开 CLI 参数、run artifact 字段和 checkpoint reuse 语义保持兼容。

## 7. 验证与 OpenSpec

- [x] 7.1 运行 `openspec validate stabilize-sensor-assisted-workflow --strict`。
- [x] 7.2 运行 `openspec status --change stabilize-sensor-assisted-workflow`，确认 change apply-ready。
- [x] 7.3 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py tests/test_hist_beam_loso.py tests/test_mmw_town10_preparation.py tests/test_metric_horizon_selection.py -q`。
- [x] 7.4 若实现触及 CLI 或公共工作流，补跑相关 `conda run -n kd_mm_beam <command> --help` 检查，并在实现记录中说明未运行的长耗时训练/全量回归。
