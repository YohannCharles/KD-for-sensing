## 1. Eligibility 与防泄漏审计

- [x] 1.1 定位当前 v8 quick validation `eligible_run_count=0` 的生成路径，梳理 run-level eligibility、summary eligibility 和 `run_marked_ineligible` 的字段来源。
- [x] 1.2 为 run metadata 增加或修正 `eligibility_status`、`eligibility_reasons`、`used_target_oracle_fields`、`target_oracle_usage_stage`、split diagnostics path 和 oracle usage summary。
- [x] 1.3 修正 eligibility checker：无 oracle 的 sensor-assisted run 不得因数据集中存在 path/radio/channel 文件而被排除；实际使用禁用 target oracle 的 run 必须排除。
- [x] 1.4 区分 target_test label 的 `evaluation_only` 用途与 adaptation/selection 用途，确保最终评价指标、histogram、KL 和 confusion 不被误判为泄漏。
- [x] 1.5 增加 quick validation summary 测试，覆盖 eligible run、禁用 oracle run、unknown oracle usage run 和 `eligible_run_count=0` reason 输出。

## 2. Collapse 诊断产物

- [x] 2.1 实现 histogram KL 工具，计算 `KL(pred_hist || support_prior)`、`KL(true_hist || support_prior)`、`KL(pred_hist || true_hist)`，并处理零概率安全项。
- [x] 2.2 在 v8/v9 evaluation 后写出 `collapse_diagnostics.json` 或等价 metrics 字段，包含 support prior、true/pred histogram、top beams、unique predicted beams 和 KL。
- [x] 2.3 增加 branch-wise eval：输出 `target_logits_only`、`prior_only`、`target_logits_plus_prior` 的 Top-K、within3、MAE 和 prediction histogram。
- [x] 2.4 记录 `beta_prior_initial`、`beta_prior_final`、`beta_prior_effective`、prior top beams、target logits top beams 和 final logits top beams。
- [x] 2.5 输出 per-true-beam confusion 摘要，并确保该 artifact 只在 evaluation 完成后生成，不参与 adaptation、prior 初始化、threshold selection 或 early stopping。
- [x] 2.6 添加 collapse diagnostics 单元或集成测试，覆盖字段完整性、KL 数值稳定性和 target_test evaluation-only 边界。

## 3. V9 配置与模型分支

- [x] 3.1 在 HiST-Beam variant/mode 解析中加入 `v9_input_conditioned_target_adaptation` 或等价 v9 mode，默认不影响 v0-v8。
- [x] 3.2 扩展配置结构，加入 `hist_beam.v9` 参数：`use_target_prior`、`beta_prior_max`、`learnable_beta_prior`、`prior_dropout`、`use_prototype_logits`、`prototype_type`、`prototype_tau`、`eta_prototype`、`sector_size`、`prototype_feature_source`、`use_widened_prior_marginal_kl`、`widened_prior_sigma`、`widened_prior_temperature` 和 loss 权重。
- [x] 3.3 实现 v9 forward 分支，输出 `target_logits`、`target_prior_bias`、`prototype_logits`、`logits_final`、`features` 和 `hist_beam` metadata，并保持 source logits 只作为诊断输出。
- [x] 3.4 实现 beta cap 参数化和 fixed beta ablation，确保 optimizer 参数集合与 `learnable_beta_prior` 一致。
- [x] 3.5 实现训练期 prior dropout，evaluation 默认关闭随机 dropout，并将 prior dropout 状态写入 diagnostics。
- [x] 3.6 添加 v9 forward/config 测试，覆盖默认 final logits 不含 source logits、beta cap、fixed beta、prior dropout train/eval 行为和旧 v8 兼容。

## 4. Target Support Prototype Logits

- [x] 4.1 实现 target support feature 提取与 prototype 构造流程，只消费 target_adapt labeled support 样本与允许输入字段。
- [x] 4.2 实现 beam-level prototype logits，记录每个 beam 的 support count，并对缺失 prototype 做 mask、平滑 fallback 或明确 unavailable reason。
- [x] 4.3 实现 sector-level prototype logits，支持 `sector_size=2/3`，记录 sector-to-beam 映射方式和每个 sector support count。
- [x] 4.4 将 prototype logits 按 `eta_prototype` 合入 final logits，并支持 `prototype only`、`A3 + beam prototype`、`A3 + sector prototype` ablation。
- [x] 4.5 将 prototype Top-K、within3、MAE、prediction histogram、prototype type、temperature、support coverage 和 unavailable reason 写入 metrics 或 collapse diagnostics。
- [x] 4.6 添加 prototype logits 测试，覆盖 beam/sector prototype、缺失类别、temperature/eta、无 target_test 泄漏和 metrics 字段。

## 5. Anti-Collapse Loss 与无标签边界

- [x] 5.1 实现 widened target prior 构造，支持更大的 Gaussian sigma 或 temperature，并记录 widened prior top beams 与 support prior 差异摘要。
- [x] 5.2 实现 prediction marginal KL loss，使用 batch mean predicted probability 匹配 widened target prior，并使用非 KD 命名记录 diagnostics。
- [x] 5.3 若启用 target_adapt unlabeled 样本，确保 dataloader/loss 只读取 target_adapt 未标注输入，不读取 target_test 或禁用 target-side oracle 字段。
- [x] 5.4 为 Group C 增加 protocol gating：metadata 无法证明无 target_test/禁用 oracle 消费时，run 必须 disabled、debug 或 ineligible。
- [x] 5.5 添加 anti-collapse loss 测试，覆盖 widened prior 构造、KL loss 权重、uniform target 非默认、Group C gating 和无标签防泄漏。

## 6. V9 Quick Validation 配置与汇总

- [x] 6.1 新增 v9 quick validation override 或配置示例，包含 Group A：A3-base、A3-no-prior、A3-fixed-beta、A3-prior-dropout。
- [x] 6.2 新增 Group B prototype ablation 配置：beam prototype only、sector prototype only、A3+beam prototype、A3+sector prototype。
- [x] 6.3 新增可选 Group C 配置：A3+widened prior KL、A3+prototype+widened prior KL，并默认受 protocol gating 控制。
- [x] 6.4 更新 LOSO summary 和 `quick_validation_conclusion.json`，汇总 eligibility、Top-K、within3、MAE、unique predicted beams、histogram KL、beta diagnostics 和 prototype diagnostics。
- [x] 6.5 确保 summary 将 ineligible/debug run 从主结论中排除，但保留 artifact path 和 reason 便于诊断。

## 7. 验证

- [x] 7.1 运行 `openspec validate add-v9-input-conditioned-target-adaptation --strict`。
- [x] 7.2 运行 eligibility 与 summary 相关定向测试，命令必须使用 `conda run -n kd_mm_beam pytest <test-files> -q`。
- [x] 7.3 运行 v9 model/prior/prototype/loss/collapse diagnostics 相关定向测试，命令必须使用 `conda run -n kd_mm_beam pytest <test-files> -q`。
- [x] 7.4 运行架构边界快速检查：`conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
- [x] 7.5 运行相关 quick validation CLI 或 dry-run/help 检查，命令必须使用 `conda run -n kd_mm_beam ...` 并确认 v9 配置不影响旧 v8 默认行为。
- [x] 7.6 最终实现完成后运行 `conda run -n kd_mm_beam pytest -q`。
