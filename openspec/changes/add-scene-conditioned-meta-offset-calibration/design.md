## Context

本仓库当前主线已经从旧 KD/HiST/Top8/residual 路线收敛到 `src/kd_sensing` 包内的配置驱动工作流：模型通过 registry 和 `modular_sequence` 组合，数据通过 descriptor/index/adapter/target provider 输出 flat sample，训练和评估通过 `prepare_task_inputs`、`forward_task_model`、`adapt_model_output` 和 objective metadata 消费统一 batch/output。现有 OpenSpec 还要求新增普通 baseline 优先走 component baseline，只有明确无法由 encoder/projector/core/head 表达时才允许 whole-model exception。

用户需求中的“场景条件化元学习与多偏移头校准”横跨模型结构、support/query episode、few-shot adaptation、synthetic dataset、配置矩阵、loss/metric 和防泄漏策略。它不能作为独立顶层 `scenario_hyperbeam/` 项目落地，否则会绕过当前包结构、CLI allowlist、runtime metadata 和本地产物边界；但它也不适合作为单个最后层 head，因为 image/fusion/align/radio/object 等偏移必须作用在不同层级。

本方案的默认 canonical visual/JEPA 基底 MUST 使用 `overlap_k16_s8_stage1`：overlap patch tokenizer（kernel 16、stride 8、max tokens 729）+ GPS-query pooling。它来自当前 visual architecture sweep 的本地最佳候选；本 change 将其作为方法基底使用，但不单独把它升级为仓库级 mainline claim，claim 升级仍需走主线文档和 strict comparability 账本。

## Goals / Non-Goals

**Goals:**

- 在现有 `kd_sensing` 包内新增一个可复现实验框架，支持 dummy/synthetic 数据先跑通 global、hyper_all_heads、MAML/FOMAML/ANIL 和 hypernetwork + MAML smoke。
- 默认以 `overlap_k16_s8_stage1` 作为 canonical predictor 的视觉/JEPA 基底；其它 patch16 mean、GPS-biased、ResNet+GPS、GPS-only 只作为显式 control、ablation 或 fallback。
- 将核心方法拆成 canonical predictor、scene/support encoder、hierarchical hypernetwork、offset/adapters、meta/adaptation runtime、loss/metric/reporting 六个可测试层。
- 保留 `canonical_logits`、每个 offset 的输出、scene embedding、gate/norm/diagnostic metadata，支持 canonical-only 和 offset 子集贡献评估。
- 复用现有 target-shot split、sensitive field guard、dataset runtime metadata、objective metadata、config recipe 和 runtime output layout。
- 用 base recipe + overrides 生成实验矩阵，避免把 80 个重复 YAML 实体化进源码表面积。

**Non-Goals:**

- 不创建新的顶层 Python package、旧式 `scripts/train.py` 主入口或兼容聚合层。
- 不恢复 KD、HiST/Hist、GPS residual、camera residual、geometry-residual label、Top8 standalone selector 或 Raymobtime/Multimodal-NF 退役路线。
- 不把真实 AoA/AoD、CSI/channel、path gain、beam power vector 或 target_test label 作为测试输入或 adaptation 选择依据。
- 不要求首个实现完成真实 DeepSense6G/MMW 全矩阵长训练；首个里程碑以 synthetic/smoke、schema、shape、loss 下降和防泄漏为 apply-ready 标准。

## Decisions

### Decision 1: 包内扩展，不新建 `scenario_hyperbeam/`

实现落在 `src/kd_sensing` 下的窄模块：`models/scene_conditioning.py`、`models/offset_heads.py`、`models/hypernetworks.py`、`models/meta_offset.py`、`data/synthetic_scenario_hyperbeam.py`、`data/episodic.py`、`engine/meta_adaptation.py`、`engine/offset_losses.py`、`config/scenario_meta_offset_recipes.py` 等。CLI 只使用包内 thin parser，例如 `kd_sensing.cli.scenario_meta_offset_sanity` 或复用 `kd-sensing-train`/`kd-sensing-evaluate`。

Rationale: 这样能继承已有 registry、batch runtime、objective metadata、runtime output layout、architecture boundary tests 和 conda 环境约束。备选方案是按用户文本创建独立 `scenario_hyperbeam/` 项目，但会产生第二套训练/评估/配置系统，和仓库治理契约冲突。

### Decision 2: 默认 canonical predictor 使用 `overlap_k16_s8_stage1`

scene meta-offset 的真实数据 recipe 和低内存正式 recipe 默认从 `overlap_k16_s8_stage1` 构建 canonical predictor：visual tokenizer 为 `overlap_patch`，`kernel_size=16`，`stride=8`，`max_tokens=729`，pooler 为 GPS-query attention，默认 `k_queries=2`，输出仍保持当前 downstream 可消费的 frame-level representation。Synthetic smoke 可以使用同结构的小张量或随机初始化权重，但 metadata MUST 记录 base variant、是否加载真实 Stage 1/checkpoint、以及 fallback 原因。

Rationale: 当前本地 visual architecture sweep 中该候选在 lowmem summary 的 `final_eval_dba` 与 `final_eval_top1` 排名最好，更适合作为多偏移校准的强基底。备选方案是继续沿用 patch16 mean/GPS-biased 基底，但那会把方案建立在较弱候选上；ResNet+GPS 和 GPS-only 更适合作为 control。

### Decision 3: 核心模型使用受控 whole-model exception，子能力仍尽量组件化

新增一个明确注册名，例如 `scene_conditioned_meta_offset`。它作为 whole-model exception 的理由是：forward 需要同时接收 query batch、可选 support batch、scene embedding、hypernetwork 小参数、多个层级 offset/adapters 和 canonical/offset 诊断输出，超出当前 `modular_sequence` 的单一路径 core/head 组合表达能力。与此同时，image adapter、fusion gate、offset heads、scene/support encoder 和 hypernetwork 子模块使用窄 class/factory，并能被单元测试直接构建。

Rationale: 直接扩展 `ModularSequenceModel` 会把 episode/support、hypernetwork 和 offset 诊断逻辑塞进通用模型，污染普通 baseline。备选方案是只新增最后层 head，但无法满足 image/fusion/align/radio/object 层级校准和贡献分析。

### Decision 4: 模态与非模态字段分层

canonical sensing modalities 仍只使用中心化契约中的 `image`、`radar`、`gps`、`lidar`、`mmwave`、`csi`。`target_state`、`object_tokens`、`scene_params`、support metadata、scene/town/weather/domain key 属于 conditioning/context/metadata 字段，不作为新模态注册，也不生成 `delayed_gps`、`image_hard`、`object` 等伪模态。

Rationale: 避免破坏现有 modality ordering、fusion slug、dataset flags 和 batch input mapping。备选方案是把 object/scene/support 注册成 modality，但这会让配置、dataset flag 和 missing-modality benchmark 语义混乱。

### Decision 5: Synthetic dataset 走 descriptor/runtime，真实数据逐步接入

新增 synthetic scenario-hyperbeam descriptor 和 sample index，生成可控 scene shift：global angular offset、weather image corruption、modality reliability shift、scene logit residual、object appearance shift、target-position beam shift。episode sampler 从 flat sample 构造 support/query，支持 K=0/1/5/10/20、labeled/unlabeled support、domain type 和 split seed。真实 DeepSense6G/MMW 后续通过 adapter/target provider 补齐 `target_state`、`object_tokens`、`scene_params`，不在首个实现里硬编码路径。

Rationale: 先用 synthetic 保证模型、loss、meta-loop 和 attribution 可测试，再接入真实数据能降低风险。备选方案是先绑死某个真实目录 schema，但会违反本地数据边界并让 CI/smoke 不可复现。

### Decision 6: Meta-training 是 engine extension，不复制训练循环

普通 supervised 路径继续走现有 train/evaluate；episodic meta-training 增加 `engine.meta_adaptation` helper，负责 sample episode、clone/adapt 参数子集、计算 support/query loss、记录 oracle usage 和返回标准 loss/metrics。MAML/FOMAML/ANIL 的 inner-loop 更新范围由 `meta.adapt_modules` 白名单解析，默认只更新 offset heads/adapters/beam head，backbone 冻结或仅显式 opt-in 更新。

Rationale: 这样能复用 optimizer、AMP、checkpoint、runtime metadata 和 objective loss。备选方案是新增完整 `meta_trainer.py` 主循环，但会复制训练状态、checkpoint 和 logging 逻辑。

### Decision 7: Config recipe 生成矩阵，实体 YAML 只保留样例和 smoke

新增 `configs/scenario_meta_offset/base.yaml`、`configs/scenario_meta_offset/smoke.yaml` 或等价少量样例；完整矩阵由 recipe/generator 输出到 ignored `outputs/analysis/scenario_meta_offset/config_matrix/` 或用户指定目录。可生成路径只覆盖本 change 声明的 current overlay，不接管任意缺失 YAML，也不恢复 retired 路径。

Rationale: 80 个重复实体 YAML 会扩大配置表面积并触发维护成本。备选方案是逐个手写模型/实验配置，短期直观但长期难以校验等价和防漂移。

### Decision 8: 辅助物理字段只作为 target/diagnostic

Angle、beam power、LOS/NLOS、path count、dominant path angle 可进入 loss 或 metrics，但必须由 target provider 标为 auxiliary target/diagnostic，并受 split/label budget/target subset policy 约束。模型 forward 输入不得接收真实 AoA/AoD、CSI/channel、path gain 或 target beam power vector。

Rationale: 保持论文级可比性和防 label leakage。备选方案是把这些字段作为输入增强，但那会把 radio oracle 泄漏到测试和 adaptation。

## Risks / Trade-offs

- [Risk] whole-model exception 容易扩大模型表面积 → Mitigation: 在 design/spec/tasks 中限定注册名、forward/output/metadata、focused tests 和架构边界测试，子模块保持窄文件和 registry/factory 可测。
- [Risk] `overlap_k16_s8_stage1` 是本地 sweep winner 但尚未升级为仓库级主线 claim → Mitigation: 本 change 只把它作为方案默认基底，metadata 与报告必须记录 base variant 和 claim caveat；主线 claim 升级另走 docs/claim 账本。
- [Risk] meta-learning 内循环内存和运行时间高 → Mitigation: 首个实现支持 synthetic small batch、first-order 模式、inner steps 0/1/3/5、adapt_modules 白名单和 AMP 兼容；长训练不作为 apply-ready 阻塞项。
- [Risk] support/query 防泄漏容易被 loss 或 evaluator 绕过 → Mitigation: 复用 sensitive field guard，metadata 记录 target oracle usage，label_budget=0/unlabeled support 访问敏感监督字段必须失败。
- [Risk] offset heads 互相替代导致贡献不可解释 → Mitigation: dummy generator 提供 head-specific shift modes，模型提供 `ablate_offsets` 或等价评估 helper，报告 offset norm/gate/贡献表。
- [Risk] config matrix 生成器可能重新实体化大量 YAML → Mitigation: 生成默认写 ignored outputs，源码只保留 base/smoke/example 和 recipe tests。
- [Risk] 真实数据字段不足以构造 target/object/scene context → Mitigation: 真实 adapter 首版允许字段 unavailable 并记录 reason；synthetic 和 smoke 不依赖真实数据；真实 claim 仅在 strict metadata 完整时升级。

## Migration Plan

1. 创建 specs 和 focused tests，先锁定 registry、synthetic sample、episode、model forward、ModelOutput adaptation、loss 和 leakage guard 契约。
2. 实现 synthetic dataset、episode sampler、collate 和 minimal config recipe，使 dummy sanity 不读取真实 `dataset/`。
3. 实现 scene/support encoder、offset heads、hierarchical hypernetwork 和 `scene_conditioned_meta_offset` 注册模型，默认 canonical predictor 使用 `overlap_k16_s8_stage1` 基底，先覆盖 supervised/global/hyper_all_heads。
4. 接入 meta-adaptation helper，跑通 MAML/FOMAML/ANIL/hyper_maml synthetic episode。
5. 增加 evaluation/reporting、matrix generator 和 docs quickstart，所有运行产物写入 ignored outputs。
6. 后续 change 再接入 DeepSense6G/MMW 真实 adapter 字段和完整矩阵长训练。

Rollback: 新功能默认 opt-in；移除或禁用新增 config recipe、注册名和 CLI 不影响现有 canonical configs。若某个阶段失败，保留已通过的 synthetic/component tests，撤回对应 recipe 或模型注册即可。

## Open Questions

- 首个真实数据 adapter 优先接入 DeepSense6G scene31-34 还是 MMW Town10 多场景？建议先 synthetic，再以 `overlap_k16_s8_stage1` 的 DeepSense6G S32-S34/S31-S34 口径作为首个真实适配验证。
- `object_tokens` 的 detector bbox schema 是否已有稳定本地来源？首版可只实现 GT/mock token 和 detector-loader 接口，不声明真实 detector 结果。
- meta-training 是否需要支持 full second-order MAML 默认开启？建议默认 FOMAML/ANIL，完整二阶仅在小 synthetic test 和显式配置下启用。
