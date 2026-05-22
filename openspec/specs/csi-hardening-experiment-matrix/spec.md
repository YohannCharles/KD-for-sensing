# csi-hardening-experiment-matrix Specification

## Purpose
定义 CSI hardening 实验矩阵的控制变量、配置命名和解释边界。
## Requirements
### Requirement: CSI hardening 控制变量配置矩阵
系统 MUST 提供可加载的 CSI hard-to-learn 控制变量配置矩阵，用于比较 clean CSI、destructive degradation、information-preserving hardening、CSI encoder 架构消融以及 hardening+架构组合。配置 MUST 固定 beam prediction 任务、MMW Town10 skybridge 场景、100 epoch 默认训练长度和可覆盖的 seed。

#### Scenario: CSI-only 第一批配置可加载
- **WHEN** 用户加载 CSI-only hardening matrix 的第一批配置
- **THEN** 系统 MUST 至少提供 A0 clean full teacher、A1 mild pilot estimation、A2 destructive degradation negative control、B3 antenna calibration、B4 fixed antenna permutation、B5 mild hardening combo、B6 medium hardening combo、C1 view gate warmup 和 C2 no internal GRU 配置
- **AND** 每个配置 MUST 使用 `modular_sequence`、`pilot_dual_view_csi` 和 beam prediction 输出契约

#### Scenario: CSI-only 第二批组合配置可加载
- **WHEN** 用户加载 CSI-only hardening matrix 的组合配置
- **THEN** 系统 MUST 至少提供 D1 mild hardening + gate warmup、D2 mild hardening + no internal GRU、D3 mild hardening + gate warmup + no internal GRU 和 D4 medium hardening + gate warmup + no internal GRU 配置
- **AND** D 组配置 MUST 不启用 destructive `csi_degradation`

#### Scenario: destructive negative control 使用已有 degradation
- **WHEN** 用户加载 A2 destructive degradation negative control 配置
- **THEN** 配置 MUST 显式启用 `data.dataset.csi_degradation`
- **AND** 配置 MUST 不启用 `csi_hardening`

### Requirement: Easy modality + CSI 多模态验证配置
系统 MUST 提供 easy modality + CSI 验证配置，用于比较普通 joint training、CSI-prioritized warmup 和 G2D-style training 对 slow CSI 贡献恢复的影响。默认 easy modality MUST 为 GPS，配置 MUST 允许后续替换为其它 easy modality。

#### Scenario: GPS+CSI 验证配置可加载
- **WHEN** 用户加载多模态验证配置
- **THEN** 系统 MUST 至少提供 E0 GPS-only、E1 GPS+clean CSI joint、E2 GPS+slow CSI joint、E3 GPS+slow CSI + CSI-prioritized warmup 和 E4 GPS+slow CSI + G2D-style 配置
- **AND** E1 到 E4 MUST 使用 `modalities: [gps, csi]` 或等价归一化后的模态集合

#### Scenario: CSI-prioritized warmup 配置表达训练阶段
- **WHEN** 用户加载 E3 CSI-prioritized warmup 配置
- **THEN** 配置 MUST 表达前期只训练或优先训练 CSI encoder 与 fusion/head 的阶段
- **AND** 配置 MUST 表达 warmup 结束后 GPS 与 CSI 联合训练的阶段

#### Scenario: G2D-style 配置引用单模态 teacher
- **WHEN** 用户加载 E4 GPS+slow CSI + G2D-style 配置
- **THEN** 配置 MUST 为 `gps` 和 `csi` teacher 提供 checkpoint 或 checkpoint registry 解析入口
- **AND** 配置 MUST 启用 G2D feature KD、logit KD、teacher confidence 和 SMP 或等价的 G2D-style 模态优先训练

### Requirement: CSI hardening sweep 分析脚本
系统 MUST 提供 `scripts/analyze_csi_hardening_sweep.py`，读取多个训练 run 的日志并输出候选排序。分析脚本 MUST 计算 final last10、best、E50、E80、E90、ceiling gap、E90 ratio、destructive 判定和 slow-high-ceiling 判定。

#### Scenario: 生成 summary 与 ranked candidates
- **WHEN** 用户运行 `conda run -n kd_mm_beam python scripts/analyze_csi_hardening_sweep.py --runs_root <dir> --pattern "csi_*" --clean_teacher_run csi_A0_clean_full_teacher --out <out_dir>`
- **THEN** 脚本 MUST 在输出目录生成 `summary.csv`
- **AND** 脚本 MUST 生成按推荐分数排序的 `ranked_candidates.csv`

#### Scenario: 计算收敛 epoch
- **WHEN** 某个 run 存在逐 epoch `beam/accuracy_val` 或等价验证准确率序列
- **THEN** 脚本 MUST 将 `final_acc` 计算为最后 10 个可用 epoch 的平均值
- **AND** `E50`、`E80`、`E90` MUST 分别为第一次达到 `0.50 * final_acc`、`0.80 * final_acc`、`0.90 * final_acc` 的 epoch

#### Scenario: 标记 destructive 与 slow-high-ceiling
- **WHEN** clean teacher run 和 variant run 都有 final accuracy
- **THEN** 脚本 MUST 计算 `ceiling_gap_acc = final_acc_clean_teacher - final_acc_variant`
- **AND** 当 `ceiling_gap_acc > 0.05` 时 MUST 标记 `is_destructive`
- **AND** 当 `ceiling_gap_acc <= 0.03` 且 `E90_ratio >= 1.5` 时 MUST 标记 `is_slow_high_ceiling`

#### Scenario: 输出可视化图表
- **WHEN** 输入 run 至少包含两个有效验证曲线
- **THEN** 脚本 MUST 输出 `learning_curves.png`
- **AND** 脚本 MUST 输出 `ceiling_gap_vs_E90_ratio.png`
