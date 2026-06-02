## Context

当前 MMW/HiST-Beam 快速验证已经存在 v8/v9 机制，但合法性汇总显示所有 run 都被排除，主要原因包括 target oracle 字段使用、target radio/path label supervision 和 split eligibility unknown。与此同时，实验信号被 GPS/LiDAR/无线派生字段、prototype 粒度、source prior collapse 和 target support outlier 混在一起，难以判断 image backbone 本身是否可迁移。

本设计把本轮工作收敛成一个 image-only 合法协议 probe：只使用 image 作为 sensing input，只允许 target support 的 beam label 做 few-shot supervision，并复用现有 A2、V8 A3 和 V9 sector proto 思路。所有项目相关 Python 命令、smoke test 和实验脚本必须通过 `conda run -n kd_mm_beam <command>` 执行。

## Goals / Non-Goals

**Goals:**

- 新增 `configs/hist_beam/image_only_legal_crossroad_probe.yaml`，声明 image-only、禁用敏感字段、label budget、split eligibility 和四个 probe mode。
- 保证 dataset、collate、forward、loss、target adaptation、evaluation 和 summary 全链路只消费 image 与合法 beam label。
- 支持 I0 image source-only、I1 frozen backbone target linear probe、I2 image V8 target prior head、I3 image V9 sector proto 四个对照。
- 新增 image backbone feature cache，提升 small head/adaptation 快速迭代效率，并通过 metadata 防止 checkpoint/cache 不匹配。
- 修复 eligibility checker，让合法 image-only run 不因原始数据中存在 path/radio/channel/beam_power 字段而被排除。
- 输出 prediction histogram、confusion-by-true-beam、combined summary 和 eligibility diagnostics，用于观察 source prior collapse 与适配趋势。

**Non-Goals:**

- 不删除现有多模态、v7/v8/v9 或历史实验代码路径。
- 不新增 v10、新复杂算法、beam-level prototype 主线、pseudo-label self-training 或 target unlabeled 默认适配。
- 不启用 GPS/LiDAR/radar/mmWave/CSI/channel/path/beam_power 作为 image-only probe 的输入、辅助监督或 target adaptation oracle。
- 不做 image backbone full fine-tuning；本轮 target adaptation 只训练 small head、adapter、prior/proto 相关轻量参数和必要 norm affine。

## Decisions

1. 使用配置驱动的 image-only profile，而不是复制一套独立训练入口。

   - 决策：新增 canonical probe config 和 shell runner，底层复用现有 dataset/model/adaptation/evaluation builder，通过 `modalities: ["image"]`、`protocol.image_only: true` 和 probe mode 分支控制行为。
   - 理由：减少重复训练代码，确保旧多模态路径保持兼容。
   - 替代方案：新增完全独立 image-only trainer。该方案短期更直接，但会绕开现有 artifact、summary 和 eligibility 体系，后续对比成本更高。

2. image-only batch 采用 allowlist 输出。

   - 决策：当 resolved modalities 等价于 `["image"]` 且启用 image-only protocol 时，batch preparation 只保留 `image`、`beam`/目标 beam、`scene`、`sample_id`、`split` 等合法字段；禁用字段可留在原始 manifest，但不得进入模型输入、loss 或 adaptation payload。
   - 理由：eligibility 应依据实际 consumed fields，而不是文件系统或 manifest 中可用字段。
   - 替代方案：dataset 彻底删除禁用字段读取。该方案更强，但风险是破坏既有多模态 manifest、diagnostics 和旧实验。

3. image-only fusion 默认使用 `identity`。

   - 决策：复用 image encoder 与 projection，默认 `hist_beam.image_only.fusion_mode: identity`，必要时保留 `single_token_transformer` 作为兼容选项。
   - 理由：本轮目标是降低复杂度，先判断 image feature + small head 是否稳定，而不是测试 fusion transformer。
   - 替代方案：强制走现有 fusion transformer 单 token 路径。该方案更接近旧模型，但引入不必要变量。

4. V8 target prior 只能由 target support labels 初始化。

   - 决策：I2 的 `target_prior_bias` 使用 target support beam labels 加 Gaussian smoothing 得到，final logits 为 `target_logits + beta * target_prior_bias`，默认不混入 source logits，并用 sigmoid cap 限制 learnable beta。
   - 理由：防止 target test label 泄漏，也避免 source 高频 beam 继续主导 final logits。
   - 替代方案：沿用旧 V8 中 source logits 和 target prior 混合。该方案可能保留历史提升，但无法隔离 prior collapse。

5. V9 只实现 sector prototype 主线。

   - 决策：I3 从 target support image features 按 `beam // sector_size` 建 sector prototype，映射回 beam logits 后参与 final logits；默认 `sector_size=2`，不启用 beam-level prototype。
   - 理由：beam-level prototype 在少量 support 和 outlier 下过敏，本轮先验证更稳的 coarse sector anchor。
   - 替代方案：同时保留 beam-level 与 sector-level proto。该方案会扩大变量，偏离降复杂度目标。

6. Feature cache 只缓存 image backbone feature 与最小评估 metadata。

   - 决策：缓存文件按 split 分开保存 `features`、`labels`、`scene`、`sample_id`、`split`，并以 `cache_meta.json` 记录 checkpoint、feature_dim、modalities、encoder、场景和 label_budget；保存 dtype 默认为 fp32 或在 meta 明确。
   - 理由：target head/proto 训练可快速复跑，同时避免把 GPS/LiDAR/path/radio 信息带入 cache。
   - 替代方案：缓存完整 batch。该方案容易把禁用字段带入 adaptation，增加 eligibility 风险。

7. Eligibility checker 记录实际 oracle usage。

   - 决策：训练、adaptation、eval 和 summary 记录 consumed fields/stages；eligibility checker 用 split metadata、protocol flags 和 consumed fields 判断 `target_oracle_fields_used`、radio/path supervision 和 unknown 状态。
   - 理由：解决“字段存在即被判 oracle”的误判，并让真正不合法 run 输出具体 config path 和使用阶段。
   - 替代方案：只通过配置声明判断合法性。该方案易被实现漂移绕过。

## Risks / Trade-offs

- [Risk] 现有 batch 或 evaluator 仍隐式期待 GPS/LiDAR/path/beam_power key → Mitigation：增加 image-only dataloader one batch、source forward、adaptation forward、loss backward 和 eval metrics smoke test。
- [Risk] target_test cache labels 被 adaptation 误读 → Mitigation：target support/test cache 分文件，adaptation API 只接受 support cache labels；test cache labels 只在 evaluation scope 解锁。
- [Risk] 旧 v8/v9 多模态路径被 image-only 分支污染 → Mitigation：所有行为由 `protocol.image_only`、resolved modalities 和 probe mode 显式控制，旧配置默认不进入新分支。
- [Risk] `eligible_run_count` 仍为 0 但原因不清楚 → Mitigation：eligibility 输出 machine-readable reasons、config path、consumed field、stage 和 split diagnostics path。
- [Risk] cache 与 checkpoint 不匹配导致错误结论 → Mitigation：cache meta 校验 checkpoint fingerprint、feature_dim、modalities、encoder、target scene、source scenes 和 label_budget，不匹配时拒绝或要求 overwrite。

## Migration Plan

1. 先实现配置解析与 image-only batch allowlist，保证原多模态配置默认行为不变。
2. 接入 image-only model forward 和输出 dict 兼容层，补 smoke test。
3. 实现 I0/I1/I2/I3 probe mode、feature cache、prediction diagnostics 和 summary writer。
4. 修复 eligibility checker 并让 image-only runner 在每个 run 结束写出 eligibility metadata。
5. 新增 `scripts/run_image_only_legal_crossroad_probe.sh`，脚本内所有 Python 命令使用 `conda run -n kd_mm_beam`。
6. 回滚策略：删除或停用新增 image-only config/runner，不影响旧多模态配置；若 eligibility 新逻辑异常，可通过旧 summary artifact 对比 consumed fields 诊断后修复。

## Open Questions

- `beam` 标签在现有 MMW batch 中的 canonical key 是否已统一，还是需要在 image-only adapter 中兼容 `target_beam`/`future_beam_label` 等别名。
- BPL dB 和 NRP 在完全禁用 `beam_power` 时是否应统一标记为 unavailable，还是允许 source/eval 阶段读取 beam_power 仅作离线指标。本变更默认不使用 `beam_power`，如无法合法计算则明确跳过。
