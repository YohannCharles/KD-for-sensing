## 1. Residual 工具与单元测试

- [x] 1.1 新增 circular residual label 工具，支持 `delta=(future_beam-last_beam) mod num_classes`、多 horizon `[B,H]` 和非法 label 清晰报错
- [x] 1.2 新增 residual logits 到 absolute beam logits 的环形重建工具，保持样本级 `last_beam` 和 horizon 维度
- [x] 1.3 新增 residual top-k 重建测试，覆盖 wrap-around、排序保持、Top-1/Top-3/Top-5 与 absolute beam 对齐
- [x] 1.4 使用 `conda run -n kd_mm_beam pytest <focused residual tests> -q` 验证 residual 工具

## 2. Batch 输入与 profile 边界

- [x] 2.1 在配置解析中新增 `hist_beam.history_anchor.enabled/mode/num_delta_classes/embedding_dim/lambda_absolute_aux` 或等价字段，默认关闭
- [x] 2.2 扩展 batch preparation，在 history anchor 显式启用时构造 `input_beam_batch`、`last_beam` 和 residual labels
- [x] 2.3 保证默认 sensor-assisted/P3/HiST-Beam 配置不传入 `input_beam_batch`，并补 default regression 测试
- [x] 2.4 在 run metadata 中记录 `history_anchor_enabled`、`history_anchor_mode`、`uses_input_beam_as_model_input` 和 enabled sensing modalities
- [x] 2.5 使用 `conda run -n kd_mm_beam pytest <focused batch/profile tests> -q` 验证输入边界

## 3. HiST-Beam 模型与 loss

- [x] 3.1 扩展 `HistBeamConfig` 和 `HistBeamFusionNet.forward()`，在 opt-in 模式下接收 beam-history embedding 或等价 conditioning
- [x] 3.2 新增 residual/delta head，并输出 `residual_logits`、`beam_logits` 或可重建 absolute logits、shared/private diagnostics
- [x] 3.3 接入 source training residual CE，并支持可配置 reconstructed absolute auxiliary CE
- [x] 3.4 保留 radio/path/hierarchical/orthogonality loss 与 residual loss 的组合语义，确保未启用 history anchor 时旧 loss 不变
- [x] 3.5 新增 model forward 和 one-batch loss smoke，使用 `conda run -n kd_mm_beam pytest <focused hist-beam model tests> -q` 验证

## 4. Evaluation、baseline 与 summary

- [x] 4.1 在 evaluation pass 中对 residual logits 执行 absolute beam 重建，并复用现有 Top-K、NRP 和 dB loss 计算
- [x] 4.2 扩展 predictions artifact，写出 `last_beam`、true/pred residual、top-k residual 和 reconstructed absolute top-k
- [x] 4.3 增加 Markov delta baseline，记录使用的 split、样本数、smoothing 和 Top-K 指标
- [x] 4.4 增加 source prior collapse 诊断，输出 source train、target test 和 prediction beam histogram 或 artifact path
- [x] 4.5 扩展 summary，分离默认 sensor-assisted 主结论与 history-anchored 内部比较
- [x] 4.6 使用 `conda run -n kd_mm_beam pytest <focused evaluation/summary tests> -q` 验证评估产物

## 5. Target adaptation 与防泄漏

- [x] 5.1 实现 history-anchored residual target adaptation 的默认冻结策略，只训练 private adapter、calibration head、bias、temperature、LayerNorm affine 或等价低参数模块
- [x] 5.2 接入 labeled target_adapt residual supervised loss，确保 unlabeled target_adapt 不读取 future beam label
- [x] 5.3 记录 trainable parameter count、trainable ratio、adaptation time 和 sensitive usage flags
- [x] 5.4 扩展 split/metadata 校验，覆盖 `input_beam` 历史窗口、future label、target_adapt/target_test guard band 和 strict eligibility
- [x] 5.5 使用 `conda run -n kd_mm_beam pytest <focused leakage/adaptation tests> -q` 验证防泄漏和适配参数范围

## 6. 配置、脚本与最小实验矩阵

- [x] 6.1 新增 history-anchored quick validation YAML 或脚本，默认一个 source 场景泛化到其它两个 target 场景、两个 seed、`label_budget=10`
- [x] 6.2 在矩阵中覆盖 absolute source-only、history absolute classifier、residual-only、residual+private calibration 和 last/Markov baseline
- [x] 6.3 确保新脚本不覆盖现有 `run_mmw_sunny_modal15_l5p3_h123.sh` 或 P3/V8 输出目录语义
- [x] 6.4 使用 `conda run -n kd_mm_beam python scripts/train.py --help` 或等价入口 smoke 验证配置可解析
- [x] 6.5 给出首轮推荐命令，但不在实现任务中默认启动长训练

## 7. 回归与 OpenSpec 校验

- [x] 7.1 使用 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` 验证架构边界
- [x] 7.2 使用 `conda run -n kd_mm_beam pytest tests/test_raymobtime_s008_selection.py tests/test_modality_visual_diagnostics.py -q` 运行项目推荐快速检查中仍适用的测试
- [x] 7.3 使用 `conda run -n kd_mm_beam pytest <all new history-anchor tests> -q` 运行新增测试集合
- [x] 7.4 使用 `openspec validate add-history-anchored-residual-beam --strict` 校验 change
- [x] 7.5 实现完成后更新 `tasks.md` checkbox，并在进入训练前确认默认 sensor-assisted 与 history-anchored run 的 summary filter 可区分
