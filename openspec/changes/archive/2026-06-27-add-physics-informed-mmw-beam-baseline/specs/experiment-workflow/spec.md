## ADDED Requirements

### Requirement: Physics-informed MMW workflow entrypoints
系统 MUST 通过现有 `kd-sensing-train`、`kd-sensing-evaluate` 和包内 CLI 支持 physics-informed MMW baseline。项目 MUST 不新增仓库根训练/评估脚本或 `scripts/*.py` thin alias；dataset inspection MUST 作为包内 CLI、console script 或训练 debug shape summary 实现。

#### Scenario: 通过训练 CLI 启动 physics-informed debug 配置
- **WHEN** 用户运行 `conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/physics_informed_mmw_debug.yaml`
- **THEN** 系统 MUST 构建 `mmw` dataset、`pinn_multimodal_beam` primary model、physics-informed loss 和现有 optimizer/checkpoint/runtime
- **AND** 系统 MUST 不调用根目录 `train.py`

#### Scenario: inspection 不使用 scripts thin alias
- **WHEN** 用户需要检查 MMW physics 字段和 shape
- **THEN** 系统 MUST 提供包内 CLI 或 debug shape summary
- **AND** 文档 MUST 不推荐 `python scripts/inspect_dataset.py`

### Requirement: Physics-informed ablation configs
系统 MUST 提供可配置 ablation workflow，用于关闭 physics loss、CSI reconstruction、path loss、array consistency 或 physics head，并提供 `vision_only`、`partial_csi_multimodal`、`history_csi_multimodal` 和 `oracle_full_csi` 四类 CSI 使用设置。ablation 配置 MUST 复用同一 dataset split、beam label space、output boundary 和 training CLI。

#### Scenario: no physics ablation
- **WHEN** 用户加载 no-physics ablation 配置
- **THEN** final config MUST 将 CSI/path/array/alignment physics loss 权重置零
- **AND** model metadata MUST 记录 physics branch 或 physics head 被关闭

#### Scenario: modality ablation
- **WHEN** 用户加载 CSI-only、image-only、image+CSI 或 full multimodal 配置
- **THEN** dataset 和 model MUST 只要求配置声明的启用模态
- **AND** 未启用模态的缺失文件 MUST 不阻止实验启动

#### Scenario: leakage-safe CSI 实验配置
- **WHEN** 用户加载 `physics_informed_mmw_vision_only.yaml`
- **THEN** 配置 MUST 设置 `use_csi_input=false` 且模型不启用 CSI 输入
- **WHEN** 用户加载 `physics_informed_mmw_partial_csi_multimodal.yaml` 或 `physics_informed_mmw_history_csi_multimodal.yaml`
- **THEN** 配置 MUST 启用多模态 sensing 加受限 CSI 输入，并将当前完整 CSI 仅用于监督
- **WHEN** 用户加载 `physics_informed_mmw_oracle_full_csi.yaml`
- **THEN** 配置 MUST 设置 `csi_input_mode=oracle_full` 和 `allow_oracle_full_csi_input=true`

### Requirement: Physics workflow artifacts and documentation
系统 MUST 将 physics-informed run 的 final config、metrics、loss breakdown、shape summary、sensitive usage flags 和 claim status 写入现有运行产物或文档索引。README MUST 只提供简短入口和数据/产物边界；详细实验口径 MUST 进入现有 docs 主线实验文档。

#### Scenario: 运行产物记录物理字段
- **WHEN** physics-informed 训练或评估完成
- **THEN** final config 或 run metadata MUST 记录 enabled modalities、physics losses、array/codebook config、shape summary 和 main-conclusion eligibility
- **AND** metrics MUST 包含普通 beam 指标和可用的 physics metrics

#### Scenario: 文档不声明未验证 claim
- **WHEN** 文档新增 physics-informed MMW baseline 条目
- **THEN** result claims registry MUST 将真实性能 claim 标记为 pending 或 unverified，直到有可追溯运行产物
- **AND** 文档 MUST 不把 synthetic smoke 或 debug run 写成正式实验结果
