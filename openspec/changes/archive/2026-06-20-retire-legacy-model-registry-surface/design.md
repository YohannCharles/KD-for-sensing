## Context

项目当前已经形成清晰的模型扩展方向：普通 baseline 通过 `modular_sequence` 组合 `ENCODERS`、`PROJECTORS`、`REPRESENTATION_CORES` 和 `HEADS`；完整 `MODELS.register(...)` 只用于 workflow/paper reproduction 或无法组件化表达的例外。然而 registry 里仍同时存在三类旧 surface：

1. 历史别名：如 `modular_sequence_model`、`gps_only_neural_baseline`、`jepa_token_transformer`、`safe_residual_reranker`。
2. 旧整模型 baseline：如 `radar_strong`、`gps_lightweight`、`mmwave_strong`、`fusion_lightweight`。
3. feature extractor 作为 `MODELS` 暴露：如 `radar_feature_extractor`、`lidar_feature_extractor`、`mmwave_feature_extractor`。

这些名称让 `MODELS.list()`、`docs/model_architecture_inventory.md` 和架构摘要输出显得比实际 current 推荐面更庞杂。与此同时，部分 specs 仍保留早期 teacher/student wording，实际代码已经把 `*_teacher` / `*_student` 旧名注册为 removed guard。这次 change 的目标是把实现、配置、文档和 specs 一次性收敛到同一个模型生命周期。

## Goals / Non-Goals

**Goals:**

- 一次性完成两批退役：
  - 第一批：无 current config 依赖或明显重复的别名/feature-extractor `MODELS` 注册/实验性 encoder。
  - 第二批：迁移 root/canonical config 后退役旧 strong/lightweight 整模型。
- 保持所有当前 root/canonical configs 可加载、可构建，并继续复用训练/验证/评估共享 runtime。
- 为退役名提供可诊断 `register_removed(...)` 错误，错误信息指向明确迁移目标。
- 收敛 OpenSpec specs、maintainer index、architecture boundary tests、extension guide 和 model architecture inventory。
- 不读取真实 `dataset/`，不训练，不生成 checkpoint/cache/log。

**Non-Goals:**

- 不删除当前论文/workflow 例外：`bev_fusion_2604`、`jepa_msac`、`gps_conditioned_jepa`。
- 不退役当前主力 image encoder：`resnet18_imagenet_rgb`、TinyViT 四个 encoder、`jepa_context_image`、`camera_ae_frozen`。
- 不退役 CSI、geometry prior、safe residual beam reranker 的 canonical 名称。
- 不重构训练循环、dataset contract、batch runtime 或 `ModelOutput` 适配逻辑。
- 不改变已退役 KD/HiST/Top8/residual 等研究线边界；本 change 只收模型 registry surface。

## Decisions

### Decision 1: 先迁移配置，再退役注册名

第二批旧整模型仍被部分 root config 使用，不能直接移除。实现顺序必须是：

1. 将 `configs/radar/*`、`configs/gps/*`、`configs/mmwave/*` 和 `configs/fusion/radar_gps_supervised.yaml` 迁到 `modular_sequence`。
2. 用 config-load 和 synthetic build/forward tests 确认行为契约仍成立。
3. 再把旧注册名从 `MODELS` 正常注册改成 `MODELS.register_removed(...)`。

Rationale: 这样本仓库 current config 不会在同一个 commit 中短暂不可用，失败时也能明确定位是配置迁移还是 registry guard 问题。

Alternative: 先保留旧注册名但在文档中标 deprecated。这个方案不会减少 `MODELS.list()` 的噪音，也无法防止新 config 继续引用旧整模型。

### Decision 2: 普通单模态 baseline 的保留语义转移到配置名，而不是模型注册名

`configs/radar/strong.yaml`、`configs/gps/lightweight.yaml` 等文件名和 run name 可以继续存在，但 `model.primary.type` 应改为 `modular_sequence`。strong/lightweight 的差异由 encoder/core/head 配置、学习率、dropout、层数或 width 参数表达，而不是由独立整模型类表达。

Rationale: 用户仍能从熟悉的 config 路径启动实验；registry surface 则只保留可组合组件和明确例外。

Alternative: 删除 strong/lightweight config。这个方案破坏 quickstart 和历史训练入口，不符合当前 README/config lifecycle。

### Decision 3: Feature extractor 不再作为完整 `MODELS` 注册

`RadarFeatureExtractor`、`LidarFeatureExtractor`、`MmWaveFeatureExtractor` 可以作为 Python 类或 encoder 实现细节保留，但 `radar_feature_extractor`、`lidar_feature_extractor`、`mmwave_feature_extractor` 不应出现在 `MODELS.list()` 中。需要组件化使用时：

- radar 使用 `ENCODERS.radar_cnn`
- lidar 使用 `ENCODERS.lidar_cnn`
- mmWave 使用 `ENCODERS.mmwave_mlp`

Rationale: feature extractor 不是完整可训练 beam prediction 模型，放在 `MODELS` 中会误导架构清单。

Alternative: 同时注册到 `MODELS` 和 `ENCODERS`。这保留了歧义，不利于后续退役治理。

### Decision 4: removed guard 是显式兼容策略

退役名不作为 unknown component 处理，而是登记到对应 registry 的 `_removed` 表，并给出迁移说明。示例：

- `radar_strong`: 使用 `model.primary.type: modular_sequence`、`encoders.radar.type: radar_cnn`、`representation_core.type: single_gru`。
- `fusion_lightweight`: 使用 `modular_sequence`、`radar_cnn + gps_mlp`、`early_concat_gru`。
- `safe_residual_reranker`: 使用 `safe_residual_beam_reranker`。

Rationale: 旧 config 或本地 notebook 失败时，用户能得到直接可执行的迁移方向。

Alternative: 直接删除注册，无 removed guard。错误只会显示 unknown 名称，定位成本更高。

### Decision 5: `cls_token_transformer_fusion` 和 `token_transformer_fusion` 暂不纳入本次退役

`cls_token_transformer_fusion` 仍有 current spec 和 config 依赖；`token_transformer_fusion` 仍有实体 config 和 tests。本 change 只处理前两批已明确候选。若后续要把 token transformer 整模型也迁到 `modular_sequence + token_transformer` core，应另起 change，避免本次范围过宽。

Rationale: 一次性退役两批已经跨越 registry、configs、specs、tests 和 docs。再加入 token transformer 整模型会扩大验证矩阵。

Alternative: 全部 fusion 整模型一次删完。风险是误伤当前 all-modality/CLS-token 工作流。

### Decision 6: 规格清理和实现同步完成

现有部分 specs 仍以 `*_teacher` / `*_student` 为规范名，而代码已经注册 removed guard。本 change 必须同步更新 specs，否则后续 agent 会继续依据过期 spec 恢复旧入口。

Rationale: 这次退役目标是“减少杂乱”，而不是只做代码删除。规格不收口，杂乱会在下一次维护中回流。

Alternative: 只改代码和 docs。短期容易，但会留下需求漂移。

## Risks / Trade-offs

- [Risk] 迁移后的 modular config 与旧整模型数值结构不完全等价。→ Mitigation: 本 change 目标是注册 surface 收口，不承诺 checkpoint 兼容；config 和 docs 必须标记旧整模型已退役，必要时保留历史结果说明而非继续可训练入口。
- [Risk] 本地未提交 checkpoint 或 notebook 仍引用旧注册名。→ Mitigation: removed guard 提供迁移目标；本地历史复现如需旧类，可从 git history 或 archive 读取，不作为 current 入口。
- [Risk] 同时修改多个 specs 可能造成 archive 冲突。→ Mitigation: spec delta 只删除/替换与旧 registry 名直接冲突的需求，并新增模块化 canonical requirement。
- [Risk] architecture boundary test 当前工作树已有无关治理漂移。→ Mitigation: 本 change 实现时记录无关失败；若 failure 来自本 change 的 allowlist/docs，则必须修复。
- [Risk] `point_cloud_mlp` 未来可能有点云实验价值。→ Mitigation: 如果实现阶段确认仍想保留，必须将其标为 experimental 并从 current inventory 推荐表移出；否则退役为 removed encoder guard。

## Migration Plan

1. 建立退役清单 fixture：列出每个旧名、registry、迁移目标和退役原因。
2. 更新 specs 和 docs，先明确目标契约。
3. 迁移 root/canonical configs：
   - Radar strong/lightweight/supervised -> `modular_sequence + radar_cnn + single_gru + beam_head`
   - GPS strong/lightweight/supervised/ablation -> `modular_sequence + gps_mlp + single_gru + beam_head`
   - mmWave strong/lightweight/supervised -> `modular_sequence + mmwave_mlp + single_gru + beam_head`
   - radar+GPS fusion -> `modular_sequence + radar_cnn + gps_mlp + early_concat_gru + beam_head`
4. 添加 config-load 和 synthetic forward tests，确认 migrated configs 可构建、forward 输出 logits，并保留 task input contract。
5. 将旧注册名改为 removed guard，更新 default component import 和 allowlist。
6. 更新 `docs/model_architecture_inventory.md`，把退役名移到退役边界或 migration table，不再列为 current model/encoder/core/head。
7. 运行 focused validations。

Rollback:

- 若配置迁移失败，可先回滚对应 config 到旧整模型，并暂不移除该注册名。
- 若 removed guard 造成未预期依赖失败，可临时恢复该名称并在退役清单中标记 blocked，直到依赖迁移完成。
- 回滚不得恢复 KD/teacher/student 旧 alias；只能恢复本 change 新退役的 strong/lightweight 或 feature extractor 名称。

## Open Questions

- `fusion_strong` 当前无实体 config 依赖；实现时是否和 `fusion_lightweight` 一起退役，还是保留到后续 fusion cleanup？本设计倾向一起退役，但 tasks 会要求先用引用扫描确认。
- `point_cloud_mlp` 是否有近期点云实验计划？若没有，退役；若有，应改为 experimental hidden component，并补 config/test/spec 证明其 current 价值。
- `token_transformer_fusion` 是否应在下一轮迁移到 `modular_sequence + token_transformer`？本 change 先不处理。
