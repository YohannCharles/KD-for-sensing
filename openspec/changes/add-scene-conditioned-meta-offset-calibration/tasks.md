## 1. 契约与测试脚手架

- [ ] 1.1 新增 focused test 文件骨架：synthetic dataset、episode sampler、model shape、offset heads、hypernetwork、meta episode、loss、config matrix、registry/output metadata 和 leakage guard。
- [ ] 1.2 在测试 helper 中加入最小 synthetic config factory，默认不读取真实 `dataset/`、不写入 checkpoint，测试命令使用 `conda run -n kd_mm_beam pytest <focused-tests> -q`。
- [ ] 1.3 为 `scene_conditioned_meta_offset` 注册名、whole-model exception metadata 和 `adapt_model_output` 兼容写 failing tests。
- [ ] 1.4 为 `target_state`、`object_tokens`、`scene_params` 不进入 canonical modality list 写 modality/config guard tests。
- [ ] 1.5 为 scene meta-offset base recipe 默认选择 `overlap_k16_s8_stage1` 写 failing test，覆盖 tokenizer `overlap_patch`、kernel 16、stride 8、max tokens 729 和 GPS-query pooler metadata。

## 2. Synthetic 数据与 episode runtime

- [ ] 2.1 实现 synthetic scenario-hyperbeam descriptor/sample index/helper，输出 scene/town/scenario/weather/domain、target_state、scene_params、beam label、可选 beam_power/angle/aux targets 和 metadata。
- [ ] 2.2 实现 synthetic modality generator，支持 image、radar、gps、lidar、mmwave、object_tokens 的轻量张量和 missing/unavailable 标记。
- [ ] 2.3 实现 head-specific shift modes：geo_only、image_only、fusion_only、radio_only、object_only、beam_only 和 all_heads。
- [ ] 2.4 实现 episode sampler 与 support/query collate，支持 K=0/1/5/10/20、labeled/unlabeled support、domain key、seed 和 sample-id disjoint 校验。
- [ ] 2.5 接入 target sensitive field guard，覆盖 label_budget=0、target_unlabeled、support/query subset 和 metadata 记录。
- [ ] 2.6 运行 `conda run -n kd_mm_beam pytest tests/test_scenario_meta_offset_dataset.py tests/test_scenario_meta_offset_episode.py -q`。

## 3. 模型子组件

- [ ] 3.1 实现 scene params encoder、support set mean/transformer encoder、support label embedding 和 scene-id baseline encoder。
- [ ] 3.2 实现 ImageOffsetHead 的 FiLM、adapter、LoRA-style 小参数接口和 feature-path debug hook。
- [ ] 3.3 实现 FusionOffsetHead 的 softmax/sigmoid gate、missing modality mask 处理和 gate statistics。
- [ ] 3.4 实现 GeoOffsetHead 的 angle_shift、mlp_shift 和 both 模式，输出 `delta_angle` 与 `geo_logit_shift`。
- [ ] 3.5 实现 AlignOffsetHead、RadioOffsetHead、ObjectOffsetHead 和 BeamLogitOffsetHead 的最小可运行版本及 shape/debug metadata。
- [ ] 3.6 实现 HierarchicalHyperNetwork，生成按 geo/image/fusion/align/radio/object/beam 分组的小参数字典并拒绝完整 backbone 权重生成。
- [ ] 3.7 运行 `conda run -n kd_mm_beam pytest tests/test_scenario_meta_offset_heads.py tests/test_scenario_meta_offset_hypernetwork.py -q`。

## 4. 整模型注册与 forward/output 契约

- [ ] 4.1 实现 `scene_conditioned_meta_offset` 模型注册入口，复用现有 batch runtime 输入并支持 `support_batch=None`。
- [ ] 4.2 实现以 `overlap_k16_s8_stage1` 为默认基底的 canonical predictor 与 target-conditioned fusion 最小路径，返回 `canonical_logits`、`logits`、`offsets`、`scene_embedding`、`debug` 和可选 auxiliary outputs。
- [ ] 4.3 实现 offset 开关、`ablate_offsets` 或等价评估 helper，支持 canonical-only、单头、子集和 all-heads forward。
- [ ] 4.4 实现 `training_strategy_metadata()`，记录注册名、架构类别、canonical base variant、visual tokenizer type、kernel/stride、pooler type、启用模态、scene/support 来源、offset heads、hypernetwork mode、meta method、adapt_modules 和 reliability/sensitive metadata 消费。
- [ ] 4.5 运行 `conda run -n kd_mm_beam pytest tests/test_scenario_meta_offset_model_shapes.py tests/test_architecture_boundaries.py -q`。

## 5. Loss、metrics 与 meta adaptation

- [ ] 5.1 实现 CE、ordinal beam distance、beam power KL/MSE、angle auxiliary、LOS/path auxiliary、offset regularization、smoothness 和 gate regularization loss helper。
- [ ] 5.2 将 scene meta-offset loss 分量接入 objective/runtime metadata，确保各分量和加权总 loss 写入 history。
- [ ] 5.3 实现 meta adaptation helper，支持 none、maml、fomaml、anil、hyper、hyper_maml 和 `meta.adapt_modules` 白名单。
- [ ] 5.4 实现 inner-loop fast weights 或等价参数更新，确保 FOMAML/ANIL 可只更新 offset heads、adapters 或 beam head。
- [ ] 5.5 为 target oracle 禁止输入、label_budget=0、不允许 target_test 调参和 auxiliary target policy 写测试。
- [ ] 5.6 运行 `conda run -n kd_mm_beam pytest tests/test_scenario_meta_offset_losses.py tests/test_scenario_meta_offset_meta_episode.py tests/test_scenario_meta_offset_leakage.py -q`。

## 6. 配置 recipe、矩阵生成与 CLI

- [ ] 6.1 新增 scene meta-offset base/smoke/example 配置，默认使用 `overlap_k16_s8_stage1` canonical base、synthetic dataset、短 epoch、小 batch 和 ignored output boundary。
- [ ] 6.2 新增 config recipe/overlay table，覆盖 global、scene info、single/multi offset、adapter、radio、fusion、object、meta、few-shot、generalization、missing modality 和 loss ablation family。
- [ ] 6.3 实现 matrix generator，默认把 generated configs/manifest 写入 `outputs/analysis/scenario_meta_offset/config_matrix/` 或用户指定目录。
- [ ] 6.4 新增包内 thin CLI 或复用现有 CLI 参数，提供 sanity、matrix generation、train/eval/adapt smoke 命令入口，并更新 `pyproject.toml` console script only if 维护上下文索引登记允许。
- [ ] 6.5 写 config load/matrix tests，确认未声明缺失 YAML 不会被自动 recipe 接管，退役路线请求会失败。
- [ ] 6.6 运行 `conda run -n kd_mm_beam pytest tests/test_scenario_meta_offset_config.py tests/test_cli_help.py tests/test_config_load_characterization.py -q`。

## 7. Evaluation、reporting 与 sanity workflow

- [ ] 7.1 实现 evaluation helper，输出 top-1/3/5、mean/median beam distance、DBA、per-scene/town/weather、missing-modality metrics 和 offset statistics。
- [ ] 7.2 实现 few-shot adaptation curve evaluator，覆盖 K=0/1/5/10/20 并记录 support label usage、seed、split artifact provenance。
- [ ] 7.3 实现 offset contribution report，支持 canonical-only、canonical+single offset、canonical+subset 和 all-heads 指标。
- [ ] 7.4 实现 synthetic sanity workflow，跑通以 `overlap_k16_s8_stage1` 为默认基底的 global、hyper_all_heads、maml_offset_heads_only 和 hyper_maml，并检查 loss 下降、shape 正确和 offset heads 被调用。
- [ ] 7.5 确认 metrics、CSV/JSON summary、plots、resolved config 和 checkpoints 默认写入 ignored `outputs/` 或用户指定本地产物目录。
- [ ] 7.6 运行 `conda run -n kd_mm_beam pytest tests/test_scenario_meta_offset_eval.py tests/test_scenario_meta_offset_sanity.py -q`。

## 8. 文档与治理同步

- [ ] 8.1 更新 README 短索引，加入 scene meta-offset smoke/matrix/adapt 命令链接，保持 README 不承载完整治理数据库。
- [ ] 8.2 新增或更新 docs 说明任务定义、数据字段、防泄漏、多 offset head、meta 方法、实验矩阵和输出指标。
- [ ] 8.3 更新 `docs/maintainer_context_index.yaml` 的 task route、entrypoint owner metadata、config/output boundary 和必要 validation commands。
- [ ] 8.4 更新 `docs/project_surface_inventory.md` lifecycle/入口/配置说明，明确该能力是 current opt-in，不恢复退役路线。
- [ ] 8.5 如新增 mainline/paper-style workflow 名称，同步 `docs/mainline_model_catalog.md`、`docs/experiment_protocols.md`、`docs/result_claims_registry.md` 和 `docs/experiment_matrix.md` 的最小事实行。

## 9. 最终验证

- [ ] 9.1 运行 `openspec validate add-scene-conditioned-meta-offset-calibration --strict`。
- [ ] 9.2 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
- [ ] 9.3 运行 scene meta-offset focused suite：`conda run -n kd_mm_beam pytest tests/test_scenario_meta_offset_*.py -q`。
- [ ] 9.4 运行配置与 CLI smoke：`conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py tests/test_cli_help.py -q`。
- [ ] 9.5 运行 sanity CLI 或等价 smoke，确认不读取真实 `dataset/` 且输出只落 ignored runtime artifact boundary。
- [ ] 9.6 整理最终实现摘要、验证命令结果、未完成真实数据长训练 caveat 和后续真实 DeepSense6G/MMW adapter 接入计划。
