## Context

当前仓库已经具备 MMW Town10 preparation、HiST-Beam LOSO runner、shared/private 模型、adapter/prototype adaptation、hierarchical beam label、radio-semantic label/prototype 和 throughput 优化能力。V6 radio-semantic 方法使用 beam power profile 派生 `radio_semantic_label`，适合作为可审计 baseline，但它的知识单元仍然接近 codebook 响应分布，而不是 Sionna channel 中更物理的 path-level propagation pattern。

P3-HiST-Beam 需要在不重写现有 workflow 的前提下，最小侵入地新增 path-level physical propagation prototype：source 侧用 path descriptor/label 监督 shared branch，source training 后保存 shared path prototype；target adaptation 侧用 source shared prototype 给 target 样本分配 path class，再维护 target-private prototype bank 进行场景内 private clustering。target label budget 为 0 时，target beam、beam power、CSI/channel、path params、path descriptor、path label 和 radio label 均不得作为训练监督。

现有实际代码路径以 `src/kd_sensing` 包为主，例如 `src/kd_sensing/data/datasets/mmw.py`、`src/kd_sensing/data/mmw/radio_semantic.py`、`src/kd_sensing/models/fusion/hist_beam.py`、`src/kd_sensing/engine/hist_beam_training.py`、`src/kd_sensing/engine/hist_beam_adaptation.py` 和 `src/kd_sensing/engine/hist_beam_prototypes.py`。用户请求中的 `datasets/multimodal_wireless_dataset.py`、`utils/prototypes.py`、`utils/leakage_guard.py` 可映射到这些包内窄模块或新建等价模块。

## Goals / Non-Goals

**Goals:**

- 自动巡检 MMW root 下 CARLA sensor、Sionna channel/path 和 metadata，报告 path-level 参数可用性，并允许通过 `data.field_map` 适配不同字段名。
- 扩展 MMW dataset flat sample，使其可选返回 `path_params`、`path_descriptor`、`path_semantic_label` 和 `path_valid`，同时保持现有 sensing modality 和 target 字段兼容。
- 新增 `PathFeatureBuilder`，从 complex path gain、delay、AoD/AoA、mask 和可选 pose 构造稳定 descriptor，缺失时能返回 unavailable 并 fallback 到 radio/coarse。
- 新增 `PathSemanticLabelBuilder`，支持 source-only KMeans path label、rule path pattern、radio_power baseline 和 coarse baseline，并保存可复用 scaler/KMeans artifact。
- 扩展 HiST-Beam forward、source loss、prototype artifact、target adaptation、inference 和 metrics，支持 V8 path-level prototype 与 path condition on/off。
- 强化 leakage guard，使 `label_budget=0` 和 unlabeled target training 的禁止字段访问变成可测试的 hard error。
- 保留 V0/V3/V4/V5/V6/V7 既有实验路径、配置名和指标语义，不删除 radio-semantic baseline 或 hierarchical loss 选项。

**Non-Goals:**

- 不把 CSI/channel/path_params 变成模型输入模态；它们只服务 label construction、few-shot labeled target supervision、offline evaluation 或 diagnostics。
- 不实现 `p(beam)=p(path_class)*p(offset|path_class)` 这类 prototype 直接预测 beam 的层次概率模型。
- 不默认保存或使用 source private prototype 对齐 target private；旧字段可保留以兼容，但默认 `use_source_private_proto=false`。
- 不要求所有 MMW 数据都有 path-level 字段；缺失时必须可诊断并按配置 fallback。
- 不提交本地 Sionna path 数据、KMeans 运行产物、训练输出、cache 或 checkpoint。

## Decisions

1. **把 path-level physical propagation prototype 作为新增能力，而不是替换 radio-semantic 模块。**

   V8 新增 `path_semantic` 配置域、path descriptor/label、path head 和 path prototype；V6 radio-semantic 保留为 beam-power baseline 和 fallback。这样可以横向比较 V5 coarse、V6 radio 和 V8 path，且不会破坏已完成的 radio-semantic smoke 和 LOSO 配置。

   备选方案是直接改写 radio-semantic builder 为 path builder，但会让历史 V6 结果不可复现，也会混淆 beam_power 与 path 参数的 leakage 边界。

2. **path 参数通过 field map 解析，descriptor 使用统一内部键。**

   数据巡检和 dataset 解析允许 `data.field_map.path_gain`、`delay`、`aod_azimuth`、`aod_zenith`、`aoa_azimuth`、`aoa_zenith`、`mask`、`tx_pose`、`rx_pose` 等配置项映射到实际 `.npy/.npz/.mat/.h5/yaml/json` 字段。进入 `PathFeatureBuilder` 前统一为内部键，例如 `a`、`tau`、`aod_azimuth`、`aoa_azimuth`、`valid_mask`。

   备选方案是写死 Sionna 字段名，但 Multimodal-Wireless 版本、导出格式和 path 文件命名可能不一致，会让巡检和训练脆弱。

3. **descriptor 是低维物理摘要，复杂 path tensor 不进入模型。**

   `PathFeatureBuilder` 对 complex gain 沿非 path 维聚合功率，得到 `q_p` 后计算 total power、dominant ratio、top3 mass、entropy、effective path count、excess delay、RMS delay spread、dominant AoD/AoA sin/cos、circular angular spread、可选 zenith spread 和 los-like score。模型只看 sensing inputs；path descriptor 只作为 source/few-shot labeled target 的辅助监督和 evaluation diagnostics。

   备选方案是直接将 path gain/delay/angle 序列送入模型，但这违反 input modality 边界，也会在 target 无标签 adaptation 中引入未来不可用的 channel 监督。

4. **KMeans path label 只在 source train fit，所有其它 split 只 transform。**

   `kmeans_path_descriptor` 默认 `num_path_classes=24`，在 source train descriptor 上 fit `StandardScaler` 和 `KMeans`，保存 mean/std、centers、mode、descriptor_dim、source domain、seed 和 unavailable 统计。source val/labeled target/target test 只能加载 artifact transform，不得重新 fit。缺失 descriptor 时按 `fallback_if_missing` 走 `radio_power` 或 `coarse`。

   备选方案是在每个 target 上重新 fit KMeans，但会泄漏 target distribution，并让 prototype class identity 在 source/target 间失去可比性。

5. **path prototype 绑定 shared representation，target-private bank 只做 target 内部 clustering。**

   source prototype artifact 新增 `mu_path_c: [K_path, shared_dim]`、`count_path` 和可选 `mu_path_descriptor`。target adaptation 计算 `softmax(cosine(c, mu_path_c)/proto_tau)` 得到 path assignment；高置信样本用 adapted private representation EMA 更新 `nu_path_s`，warmup 后加入 private clustering loss。该 loss 不对齐 source private prototype。

   备选方案是保存 source private path prototype 并约束 target private 对齐它，但 scene-private branch 正是用来吸收 town/scenario/weather/local geometry/codebook mapping 差异，强行对齐会削弱 adaptation。

6. **beam head 可以显式 path-conditioned，但 prototype 不直接输出 beam。**

   Source 阶段如果 `use_path_condition_in_beam_head=true`，用 `softmax(path_logits/tau)` 加权 path embedding；target 阶段优先用 `mu_path_c` 的 cosine assignment 加权 path embedding，没有 prototype 时 fallback 到 `path_logits`。关闭 condition 时 beam head 继续只读 `concat(c, s_star)`。无论哪种路径，最终 beam 都由 BeamHead 输出。

   备选方案是让 path prototype 映射到 beam distribution，但这会把 path class 当作 beam hierarchy parent，违背用户要求，也难以处理同一 propagation pattern 在不同 town/codebook 下的 private correction。

7. **训练访问通过 batch view/leakage guard 约束，而不是只靠约定。**

   target adaptation dataloader 或 training loop 应为 batch 附带 split、label_budget、is_labeled、is_unlabeled 等 metadata。loss 读取敏感字段前必须通过 guard；当 `label_budget=0` 或 unlabeled target training 时，读取 beam、beam_power、CSI/channel、path_params、path_descriptor、path_semantic_label、radio_semantic_label 作为训练监督直接 raise。评估和 diagnostics 可用 target test labels，但必须处于 evaluation context。

   备选方案是在 dataset 中简单删除字段，但 few-shot labeled target、offline evaluation 和 diagnostics 仍需要这些字段，统一 guard 更可审计。

8. **先 P0-P4 smoke，再 P5-P8 training/adaptation 闭环。**

   第一阶段保证巡检、descriptor、KMeans label、dataset output 和 model forward 可运行且不影响 V0/V3/V5/V6。第二阶段接 source loss、path prototype、target adaptation、inference 和 diagnostics。这样能在真实 MMW path 字段尚不完全确定时，先通过 fixture/smoke 固化 contract。

## Risks / Trade-offs

- [Risk] Sionna path 文件格式和字段命名与预期不同。→ Mitigation：先实现 `inspect_dataset.py` 字段探测和 `data.field_map`，descriptor builder 接受统一内部键并报告 unavailable reason。
- [Risk] KMeans path label 在 source 数据不足或 class imbalance 时不稳定。→ Mitigation：保存 scaler/KMeans artifact、class counts 和 coverage；提供 rule/radio/coarse fallback；summary 输出 source-target path histogram。
- [Risk] path descriptor 维度随 zenith/pose availability 变化。→ Mitigation：`descriptor_dim=auto` 在 artifact 中固化，transform 时校验维度；缺失可选字段使用 mask/unavailable reason，不静默改变维度。
- [Risk] path-conditioned beam head 可能不提升 Top-K。→ Mitigation：提供 `exp_v8_path_condition_off.yaml` 和 `exp_v8_path_condition_on.yaml`，将 condition 作为显式消融；prototype anchor 仍可独立用于 adaptation。
- [Risk] leakage guard 过严影响 few-shot target supervision。→ Mitigation：区分 labeled target subset 与 unlabeled target subset，只有配置 `allow_labeled_target_path_supervision=true` 时才允许 labeled subset 使用 path label/descriptor。
- [Risk] 新增 path head/regression head 改变旧 checkpoint 兼容。→ Mitigation：新增模块按配置启用；加载旧 V0/V3/V5/V6 checkpoint 时允许 missing/unexpected path keys 的受控兼容，并记录 variant metadata。
- [Risk] path artifact 保存大对象或本地数据。→ Mitigation：只保存 scaler/KMeans/prototype 统计和配置，不保存原始 path_params、CSI/channel 或样本级大数组。

## Migration Plan

1. 新增 dataset inspection 与 field map，不改变训练默认路径。
2. 新增 path semantics builder、artifact schema 和 fixture tests；在缺失 path 数据时保持 radio/coarse fallback。
3. 扩展 MMW dataset flat sample 与 collate/mask，确保 path/CSI/channel 不进入 enabled modalities。
4. 扩展 HiST-Beam model forward，使旧 variant 默认输出和 BeamHead 输入语义保持兼容。
5. 接入 source path loss 与 prototype generation；source-only baseline 仍按需跳过无关 prototype。
6. 接入 target adaptation `proto_type=path`、target-private bank 和 leakage guard hard error。
7. 增加 V8 YAML、LOSO summary 和 metrics diagnostics，完成 P0-P8 smoke/one-epoch 验证。
8. 用 leave-one-town-out、leave-one-scenario-out 和 weather-shift 矩阵验证 V8 相对 V6 的效果；实验输出留在本地产物目录。

## Implementation Outcome

已在 `outputs/p3_v8_matrix/` 完成当前可用 MMW sunny/Town10 `scenario_loso` 矩阵，共 216 个 completed runs，覆盖 V3 source-only、V4 adapter、V5 coarse prototype、V6 radio prototype、V8 path prototype、path condition off、adapter radio off 和 full fine-tuning baseline，budgets 为 0/5/10，seeds 为 0/1/2。

写实结论如下：

- V8 path prototype 没有在 top1 上超过 V6 radio-semantic。完整矩阵中 V8 overall top1=26.43%，V6 radio top1=28.35%；V8 top3=60.81%、top5=75.72%，高于 V6 的 top3=57.26%、top5=71.08%，说明 path prototype 当前更改善候选 beam 集合，而不是 single-best beam ranking。
- V8 path condition on/off 消融成立。`v8_path_proto` 相比 `adapter_path_proto` top1 +4.34pp、top3 +10.92pp、beam power loss -0.35 dB，说明将 path assignment embedding 送入 BeamHead 在当前矩阵中有效。
- V8 接近或略优于 full fine-tuning baseline，且参数效率明显更高。V8 trainable ratio 约 0.28%，full fine-tuning 为 100%；V8 top1=26.43%，full fine-tuning top1=25.24%。
- `label_budget=0` 和部分 few-shot adaptation 存在负迁移。V8 adapted-source top1 delta 在 budget0 为 -1.55pp、budget5 为 -0.43pp、budget10 为 +1.32pp；full fine-tuning 在 budget0/budget5 分别为 -4.03pp/-4.17pp，表现出过拟合或不稳定风险。
- path semantic head 仍未学稳。V8 path semantic accuracy 约 4.43%，接近 24 类随机水平；path prototype coverage 为 0.667，存在空类。后续应优先降低 path class 数、改进 descriptor clustering 或把 path prototype 用于 top-k reranking。
- 当前数据只支持 sunny/Town10 scenario_loso 结论。`dataset/MMW/data_availability.json` 没有多 town 或多 weather ready domains，因此本 change 不声称已验证 leave-one-town-out 或 weather-shift。
- `last_beam` diagnostic baseline top1 约 94.8%、top3 约 99.2%，说明 `num_pred=1` 时 beam temporal persistence 很强。后续若目标是 sensor-assisted beam prediction，应明确是否允许历史 beam 作为 baseline，并优先切换到 `gps+image+lidar+radar` 输入，避免 `mmwave/beam_power` 与 beam label 形成过强近路。

## Open Questions

- 实际 Multimodal-Wireless / Sionna path 文件中 path 维是否固定在第 0 维，还是需要 field metadata 指定 path axis？建议首版用 shape heuristic 加 `data.field_map.path_axis` 覆盖。
- `los_like_score` 是否需要归一化到固定范围？建议首版保留原始比值并由 StandardScaler 处理，diagnostics 再报告分布。
- target labeled subset 的 path supervision 默认关闭是否足够保守？建议默认关闭，只有研究消融显式开启。
