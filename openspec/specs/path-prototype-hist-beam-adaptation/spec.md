# path-prototype-hist-beam-adaptation Specification

## Purpose
定义 P3/HiST-Beam path prototype adaptation 的数据、输入边界、source-only prototype 构建、target 防泄漏和 LOSO 评估契约，用于审计 path-level propagation auxiliary 信号如何参与跨场景 few-shot 适配而不成为 sensing input。
## Requirements
### Requirement: Path-level 数据巡检与字段映射
系统 MUST 提供 Multimodal-Wireless path 数据巡检能力，用于扫描数据根目录下的 CARLA sensor 文件、Sionna channel/path 文件和 metadata 文件，并输出 town、scenario、weather、模态可用性、beam label、beam_power、CSI/channel 和 path-level propagation parameter 的可用性。巡检 MUST 不写死 path 字段名，MUST 支持通过 `data.field_map` 将实际字段映射到内部 path key。

#### Scenario: 扫描 MMW root 并报告 path 可用性
- **WHEN** 用户运行 `scripts/inspect_dataset.py` 或等价 CLI 检查 Multimodal-Wireless root
- **THEN** 系统 MUST 打印或写出每个可识别 domain 的 town、scenario、weather 和 sample count
- **AND** 系统 MUST 报告 camera、radar、gps、lidar、imu、depth 的可用性
- **AND** 系统 MUST 报告 beam label、beam_power、CSI/channel 和 path-level parameters 是否存在

#### Scenario: 识别 Sionna path 等价字段
- **WHEN** Sionna path 文件包含 gain、delay、AoD/AoA、valid mask 或 Tx/Rx pose 的任意等价字段名
- **THEN** 巡检 MUST 通过自动候选匹配或 `data.field_map` 将其映射到 `a`、`tau`、`aod_azimuth`、`aod_zenith`、`aoa_azimuth`、`aoa_zenith`、`valid_mask`、`tx_pose`、`rx_pose` 等内部 key
- **AND** 巡检 MUST 在字段缺失或 shape 不可解析时记录 unavailable reason

### Requirement: Path-level dataset 输出与输入边界
MMW dataset MUST 在保留现有 flat sample 字段的基础上，按配置可选返回 `path_params`、`path_descriptor`、`path_semantic_label` 和 `path_valid`。CSI/channel/path_params/path_descriptor/path_semantic_label MUST NOT 被注册或消费为模型 sensing input modality。

#### Scenario: 返回 path auxiliary 字段
- **WHEN** dataset 样本存在可解析 path-level propagation parameters 且 `path_semantic.enabled: true`
- **THEN** `__getitem__` 结果 MUST 保留 camera/radar/gps/lidar/imu optional tensors、beam、beam_power、radio_semantic_label、scenario、town、weather、domain 和 sample_id
- **AND** `__getitem__` 结果 MUST 额外包含可选 `path_params`、`path_descriptor`、`path_semantic_label` 和 `path_valid`

#### Scenario: path 字段不作为输入模态
- **WHEN** 用户启用 HiST-Beam 或 P3-HiST-Beam 训练
- **THEN** 模型输入 MUST 只来自配置启用的 sensing modalities
- **AND** channel、CSI、path_params 和 path_descriptor MUST 只可用于 source label construction、few-shot labeled target supervision、offline evaluation 或 diagnostics

#### Scenario: 无标签 target dataloader 屏蔽训练访问
- **WHEN** target adaptation 的 `label_budget=0` 或 batch 属于 unlabeled target subset
- **THEN** training loss MUST NOT 读取 target beam、beam_power、CSI/channel、path_params、path_descriptor、path_semantic_label 或 radio_semantic_label
- **AND** 若 loss 尝试访问这些字段作为训练监督，系统 MUST raise error

### Requirement: PathFeatureBuilder 物理 descriptor
系统 MUST 提供 `PathFeatureBuilder`，从 path-level propagation parameters 构造固定维度 path descriptor。builder MUST 支持 complex gain shape 为 `[P]`、`[P, Rx, Tx]`、`[P, subcarrier, Rx, Tx]` 或等价 path-first/可配置 path-axis 张量，MUST 支持 delay、AoD/AoA azimuth/zenith 和 optional valid mask。

#### Scenario: 从 complex gain 计算 path power
- **WHEN** `build_descriptor(path_params)` 收到 complex gain `a` 和可选 valid mask
- **THEN** builder MUST 沿所有非 path 维计算 `power_p = sum(abs(a_p)^2)`
- **AND** builder MUST 计算 `q_p = power_p / sum(power_p)`，只统计 valid path
- **AND** 若 path_params 缺失或没有 valid path，builder MUST 返回 unavailable 而不是伪造 descriptor

#### Scenario: 构造基础 path descriptor
- **WHEN** valid path power、delay 和 azimuth 字段可用
- **THEN** descriptor MUST 至少包含 log_total_power、dominant_path_ratio、top3_path_mass、path_entropy、effective_num_paths、mean_excess_delay、rms_delay_spread、dominant_aod_azimuth_sin、dominant_aod_azimuth_cos、dominant_aoa_azimuth_sin、dominant_aoa_azimuth_cos、weighted_aod_angular_spread、weighted_aoa_angular_spread 和 los_like_score
- **AND** path_entropy MUST 使用 valid path 数归一化

#### Scenario: 使用 circular statistics 计算角度 spread
- **WHEN** AoD/AoA azimuth 或 zenith 接近 `-pi/pi` 跳变边界
- **THEN** builder MUST 使用 circular statistics 计算 dominant direction 和 angular spread
- **AND** descriptor MUST NOT 因普通线性差值在边界处产生虚假大 spread

#### Scenario: 可选 zenith 与归一化
- **WHEN** `use_zenith: true` 且 zenith 字段可用
- **THEN** descriptor MUST 包含 weighted aod/aoa zenith spread 或等价可诊断字段
- **AND** 若 `normalize_output: true`，builder MUST 使用配置或 artifact 中的统计量归一化输出

### Requirement: PathSemanticLabelBuilder
系统 MUST 提供 `PathSemanticLabelBuilder`，支持 `kmeans_path_descriptor`、`rule_path_pattern`、`radio_power` 和 `coarse` 模式。默认主方法 MUST 为 `kmeans_path_descriptor`，默认 `num_path_classes` MUST 为 24。

#### Scenario: source-only KMeans fit
- **WHEN** source train split 存在合法 path_descriptor 且 mode 为 `kmeans_path_descriptor`
- **THEN** builder MUST 只在 source train descriptor 上 fit StandardScaler 和 KMeans
- **AND** builder MUST 保存 scaler_mean、scaler_std、kmeans_centers、descriptor_dim、num_path_classes、seed、source domain 和 config 到 artifact

#### Scenario: 非 source train split 只 transform
- **WHEN** source val、labeled target、unlabeled target 或 target test 需要 path label
- **THEN** builder MUST 加载已保存 artifact 进行 transform 或 predict
- **AND** builder MUST NOT 在这些 split 上重新 fit scaler 或 KMeans

#### Scenario: rule path pattern fallback
- **WHEN** mode 为 `rule_path_pattern` 或 KMeans artifact 不可用且配置允许 fallback
- **THEN** builder MUST 基于 dominant_path_ratio、rms_delay_spread 和 angular spread 分箱
- **AND** builder MUST 将分箱组合映射到受控数量的 path classes

#### Scenario: radio_power 与 coarse baseline
- **WHEN** mode 为 `radio_power`
- **THEN** builder MUST 复用已有 beam_power peak/spread radio label 作为 V6 baseline
- **AND** 当 mode 为 `coarse` 时，builder MUST 复用 `beam // group_size` 作为 V5 baseline label

### Requirement: P3-HiST-Beam 模型 forward
P3-HiST-Beam MUST 在现有 shared/private/fusion 架构上新增 path head、可选 path attribute regression head 和 path embedding。BeamHead MUST 支持读取 `concat(c, s_star)` 或 `concat(c, s_star, e_path)`，且 prototype MUST NOT 直接输出 beam prediction。

#### Scenario: source 阶段 path-conditioned beam head
- **WHEN** source forward 启用 `use_path_head: true` 和 `use_path_condition_in_beam_head: true`
- **THEN** 模型 MUST 输出 `path_logits: [B, K_path]`、beam_logits、shared representation `c` 和 private representation `s`
- **AND** 模型 MUST 使用 `softmax(path_logits / tau)` 加权 path embedding 得到 `e_path`
- **AND** BeamHead MUST 从 `concat(c, s, e_path)` 产生 beam_logits

#### Scenario: target 阶段 prototype-conditioned beam head
- **WHEN** target adaptation 或 target inference 启用 path condition 且存在 `mu_path_c`
- **THEN** 模型 MUST 基于 `cosine(c, mu_path_c)` 和 `proto_tau` 计算 path assignment
- **AND** 模型 MUST 用 assignment 加权 path embedding 得到 `e_path`
- **AND** BeamHead MUST 从 `concat(c, adapter(s), e_path)` 产生 beam_logits

#### Scenario: 关闭 path condition 保持旧语义
- **WHEN** `use_path_condition_in_beam_head: false`
- **THEN** BeamHead MUST 只读取 `concat(c, s_star)`
- **AND** path head MAY 继续输出 path_logits 用于 auxiliary supervision 或 diagnostics

#### Scenario: prototype 不作为 beam hierarchy parent
- **WHEN** 系统启用 path prototype 或 path semantic label
- **THEN** 系统 MUST NOT 实现 `p(beam)=p(path_class)*p(offset|path_class)` 或等价直接 prototype-to-beam 预测
- **AND** path_semantic_label MUST NOT 被当作 beam hierarchy parent label

### Requirement: Source path loss
source training MUST 支持 beam CE、path semantic CE、可选 path descriptor regression、orthogonality、shared scene confusion 和 private scene preservation 的组合 loss。Path loss MUST 只在 batch 中存在合法 path target 时启用。

#### Scenario: 有 path_semantic_label 时计算 path CE
- **WHEN** source batch 包含合法 `path_semantic_label` 且 `lambda_path > 0`
- **THEN** training loop MUST 对 `path_logits` 计算 CE loss
- **AND** diagnostics MUST 记录 path loss 和有效样本 coverage

#### Scenario: 有 path_descriptor 时计算 regression
- **WHEN** source batch 包含合法 `path_descriptor`、模型输出 `path_attr_pred` 且 `lambda_path_reg > 0`
- **THEN** training loop MUST 对 `path_attr_pred` 和 `path_descriptor` 计算 SmoothL1 或配置指定 regression loss
- **AND** diagnostics MUST 记录 path descriptor regression MSE 或等价指标

#### Scenario: 保留旧 loss 选项
- **WHEN** 用户运行 V5 coarse、V6 radio-semantic 或 hierarchical beam baseline
- **THEN** 系统 MUST 保留旧 radio semantic loss 和旧 hierarchical beam loss 配置
- **AND** 系统 MUST NOT 强制启用 path loss

### Requirement: Source path prototype artifact
source pretraining 后，系统 MUST 能基于 source train split forward 生成 path prototype artifact。artifact MUST 至少包含 `mu_path_c`、`count_path` 和 config，MAY 包含仅用于 diagnostics 的 `mu_path_descriptor`。

#### Scenario: 保存 shared path prototypes
- **WHEN** source train forward 得到 shared representation `c` 且样本有合法 path_semantic_label
- **THEN** prototype generator MUST 按 path label 聚合 normalized 或配置指定形式的 shared representation
- **AND** artifact MUST 保存 `mu_path_c: [K_path, shared_dim]` 和 `count_path: [K_path]`
- **AND** artifact metadata MUST 记录 `prototype_space=shared_path_physical`

#### Scenario: 同时保留 coarse 与 radio prototypes
- **WHEN** 配置需要 V5、V6 和 V8 对比
- **THEN** prototype artifact 或 artifact registry MUST 能区分 `mu_coarse_c`、`mu_radio_c` 和 `mu_path_c`
- **AND** summary MUST 记录每个 variant 实际使用的 `proto_type`

#### Scenario: 默认不使用 source private prototype
- **WHEN** 旧代码或旧 artifact 包含 source private prototype 字段
- **THEN** target adaptation 默认 MUST NOT 使用 source private prototype 对齐 target private representation
- **AND** 只有显式 `use_source_private_proto=true` 时才可进入兼容路径

### Requirement: Path prototype target adaptation
target adaptation MUST 支持 `proto_type=none|coarse|radio_semantic|path`。当 `proto_type=path` 时，系统 MUST 使用 source shared path prototype 给 target 样本分配 path class，并维护 target-private prototype bank `nu_path_s`。

#### Scenario: 基于 source path prototype 分配 target 样本
- **WHEN** target adaptation batch forward 得到 shared representation `c` 且存在 `mu_path_c`
- **THEN** 系统 MUST 计算 `alpha_path = softmax(cosine(c, mu_path_c) / proto_tau)`
- **AND** 系统 MUST 记录 `k_hat=argmax(alpha_path)`、confidence、coverage 和 assignment histogram

#### Scenario: EMA 更新 target-private path bank
- **WHEN** target 样本 assignment confidence 高于 `confidence_threshold`
- **THEN** 系统 MUST 使用 `normalize(s_adapt)` 以 `target_proto_momentum` EMA 更新 `nu_path_s[k_hat]`
- **AND** 系统 MUST 维护 `nu_count` 或等价 initialized mask

#### Scenario: warmup 后计算 private clustering loss
- **WHEN** `proto_warmup_epochs` 已完成且 `nu_path_s[k_hat]` 已初始化
- **THEN** 系统 MUST 对高置信样本计算 `L_proto_private = ||normalize(s_adapt) - stopgrad(nu_path_s[k_hat])||_2^2`
- **AND** 该 loss MUST 表示 target 内部 private clustering，而不是 source private alignment

#### Scenario: label_budget 为 0 时禁止 target path 监督
- **WHEN** `label_budget=0`
- **THEN** target adaptation MUST NOT 使用 target beam、beam_power、CSI/channel、path_params、path_descriptor、path_semantic_label 或 radio_semantic_label 计算训练 loss
- **AND** adaptation 只能使用模型预测、source prototypes 和 target sensing inputs

#### Scenario: labeled target path supervision 显式开启
- **WHEN** `label_budget>0` 且 `allow_labeled_target_path_supervision=true`
- **THEN** labeled target subset MAY 使用 path_semantic_label 或 path_descriptor 计算监督 loss
- **AND** unlabeled target subset MUST 继续禁止这些 path labels 或 descriptors 作为训练监督

### Requirement: P3 inference 与诊断指标
P3-HiST-Beam inference MUST 使用 sensing inputs 产生 fusion feature、shared/private representation、可选 adapter private representation、path logits/path assignment 和 beam logits。evaluation MUST 保留既有 beam/power/adaptation 指标，并新增 path diagnostics。

#### Scenario: path-aware inference
- **WHEN** target-adapted inference 执行
- **THEN** 系统 MUST 计算 `h=fusion(encoders(x))`、`c=Ec(h)`、`s=Es(h)`、`s_star=adapter(s)` 或 `s`
- **AND** 若启用 path condition，系统 MUST 优先用 `mu_path_c` assignment 构造 `e_path`，否则用 `path_logits` softmax 构造 `e_path`
- **AND** 最终 `pred_beam` MUST 来自 `argmax(beam_logits)`

#### Scenario: 输出 path diagnostics
- **WHEN** target test 有 path labels 或 path descriptors 可用于离线评价
- **THEN** metrics MUST 包含 path semantic accuracy、path descriptor regression MSE、prototype assignment confidence、prototype coverage per class 和 source-target path class histogram
- **AND** evaluation MUST NOT 将 target test path labels 回传给 adaptation、threshold selection 或 prototype update

#### Scenario: 保留 power 与效率指标
- **WHEN** beam_power 存在
- **THEN** metrics MUST 保留 Top-1、Top-3、Top-5、normalized received power 和 beam power loss dB
- **AND** adaptation metrics MUST 保留 trainable parameter ratio 和 adaptation time

### Requirement: P3 配置、变体与 smoke tests
系统 MUST 提供 V8 path prototype 相关 YAML 配置和 smoke tests，覆盖 dataset inspection、descriptor、KMeans label、dataset output、model forward、source training、prototype saving、target adaptation 防泄漏和 evaluation diagnostics。

#### Scenario: V8 配置默认值
- **WHEN** 用户加载 V8 path prototype 配置
- **THEN** 配置 MUST 包含 `path_semantic.enabled=true`、`mode=kmeans_path_descriptor`、`num_path_classes=24`、`fit_on_source_only=true`、`fallback_if_missing=radio_power` 和 `use_path_regression=true`
- **AND** 配置 MUST 包含 `model.use_path_head=true`、`model.use_path_condition_in_beam_head=true` 和 `model.path_embed_dim=32`
- **AND** 配置 MUST 包含 `target_adapt.proto_type=path`、`proto_tau=0.1`、`confidence_threshold=0.75`、`proto_warmup_epochs=5`、`target_proto_momentum=0.9` 和 `allow_labeled_target_path_supervision=false`

#### Scenario: 第一阶段 smoke
- **WHEN** 完成 P0-P4 实现
- **THEN** smoke tests MUST 验证能扫描 MMW 数据结构、从一个 Sionna path sample 构造 path_descriptor、在 source train 上 fit KMeans path labels、dataset 返回 path_descriptor/path_semantic_label、model forward 输出 beam_logits/path_logits/c/s
- **AND** smoke tests MUST 验证 V0/V3/V5/V6 不受影响

#### Scenario: 第二阶段 smoke
- **WHEN** 完成 P5-P8 实现
- **THEN** smoke tests MUST 覆盖 source training one epoch、保存 `mu_path_c`、`label_budget=0` target adaptation、no target leakage assertion 和 Top-K/NRP/path diagnostics evaluation
- **AND** 所有项目相关 Python 命令 MUST 使用 `conda run -n kd_mm_beam`
