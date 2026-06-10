# 研究笔记收束

本文件浓缩原根目录零散方案草稿中仍有价值的研究判断。实现契约仍以 `openspec/specs/` 为准；运行入口、实验矩阵和可视化说明分别看 `README.md`、`docs/experiment_matrix.md`、`docs/Raymobtime_s008_selection.md` 和 `tools/visualization/README.md`。

## 文档取舍

- 根目录只保留 `README.md` 和 `AGENTS.md`；历史方案草稿不再作为入口文档维护。
- 需求、架构边界和行为契约进入 `openspec/specs/`。
- 实验运行顺序、配置矩阵和常用命令进入 `docs/experiment_matrix.md`。
- Viewer、manifest 和性能排查进入 `tools/visualization/README.md`。
- 本文件只保留研究路线判断、负结果解释和跨文档的阅读导航。

## 已退役研究线

模态失衡、蒸馏、HiST-Beam/P3、GPS coarse anchor、Top8 selector、GPS residual 和 camera residual 研究线已经退役。本仓库不再维护专用审计、互补 case mining、阶段性效用验证、Raymobtime 失衡诊断、G2D 失衡结果汇总、HiST LOSO、history-anchor Hist、P3/V7/V8/V9 probe 或 GPS residual/camera residual 入口；旧模态子集/扰动独立脚本也不再作为长期入口。当前主线回到普通训练、统一评估、配置化通用模态子集调试、DeepSense6G/MMW BGAM、MMW GPS v2、Raymobtime s008 current snapshot workflow、CSI hardening 和少样本跨场景 adaptation。

历史输出如已存在于本地 `outputs/`，可作为静态资料保留，但 README、OpenSpec 和工具文档不再把这些研究流程列为当前可运行入口。

## CRAF、MARF 和 G2D 历史结论

CRAF 早期实验显示：warmup 太短、基于训练 loss 的反事实 target 噪声大、弱模态 auxiliary loss 持续干扰，容易得到“训练集更强、验证集更差”的 all-modal 过拟合。teacher-prior CRAF 曾尝试先用单模态 teacher 的验证表现建立 prior，再学 residual gate，并冻结或选择性微调部分 encoder；这些入口现在已退役。

MARF 曾定位为不写死强弱模态，而是用 teacher prior、horizon-wise router、anchor fusion、residual adapter 和 subset-aware training 支持不同模态组合、缺失和扰动。相关模型、训练 helper、配置和测试已经从当前支持面删除。

G2D 曾作为通用多模态蒸馏方法，采用 future-only 标签约定：

```text
labels: [B, 3] = [t+1, t+2, t+3]
logits: [B, 3, 64]
```

不要再把历史最后一帧 beam 当作训练 label，也不要输出旧的 current/h0 指标。G2D-lite、G2D-global 和 G2D-horizon 的配置、蒸馏运行时、SMP 和诊断入口已经退役；当前主线入口是 supervised/adaptation，旧 `logits_kd` 和 `rkd` 路径会被 migration guard 拒绝，fusion KD virtual alias 不再生成。

## CSI 和 MMW 研究判断

CSI encoder 的主线是 pilot-based dual-view，而不是直接对输入加普通 AWGN：

```text
clean CSI
-> train-set global RMS normalization
-> pilot estimation h_hat = h + e
-> frequency view + delay view
-> CNN tokenizer
-> symmetric gated view fusion
-> 1-layer GRU
-> [B, T, 64]
```

标签仍由 clean channel/beam power 生成，不要用 noisy CSI 重新生成标签。SNR、pilot length、pilot power、delay taps、view fusion 和 hardening 控制见 `openspec/specs/csi-modality-model/spec.md`。

MMW path-level CSI 退化用于把 ray-tracing perfect channel 变成更接近真实 estimated CSI 的输入。优先 profile：

- `medium`: gain AWGN 10 dB、path dropout 20%、AoA/AoD noise 3 deg、delay noise 0.5 ns、antenna phase error 10 deg、temporal shift `[-1,0,1]`。
- `hard`: gain AWGN 5 dB、path dropout 30%、dominant path attenuation 0.5、AoA/AoD noise 5 deg、delay noise 1 ns、antenna phase error 20 deg、temporal shift `[-2,-1,0,1,2]`。

CSI hardening 的目标不是降上限，而是制造 high-ceiling but slow-to-learn CSI。候选设置应满足 `ceiling_gap <= 0.02~0.03` 且 `E90_ratio >= 1.5`；下降超过 `0.05` 的配置只作为 destructive negative control。配置矩阵与分析脚本契约见 `openspec/specs/csi-channel-degradation/spec.md` 和 `openspec/specs/csi-hardening-experiment-matrix/spec.md`。

## 模态预处理判断

- GPS：优先使用 UTM/XY 米制坐标、UE-BS 相对位置、训练集 scaler，并加入 `dx/dy/speed/d/sin(theta)/cos(theta)` 等运动和几何特征。不要用全数据 fit scaler。
- LiDAR：多模态 DeepSense 场景优先 BEV height/intensity/density、ROI/FoV crop、可选背景过滤和安全增强；LiDAR-only 轻量 future beam baseline 可保留 angle-bin/range vector。不要在 Dataset 初始化阶段全量读取 LiDAR 再 `cat` 算 mean/std；如需全局统计，使用 streaming stats artifact。
- mmWave：DeepSense 64 维 power vector 应走 dB 压缩、finite/NaN 清洗和 train-split per-beam z-score，保留绝对功率/SNR/path-loss 信息；不要把每条样本 softmax 成纯相对分布。

相关契约见：

- `openspec/specs/gps-preprocessing/spec.md`
- `openspec/specs/lidar-preprocessing/spec.md`
- `openspec/specs/mmwave-preprocessing/spec.md`
- `docs/extension_guide.md`

## 数据集与 Viewer

Raymobtime s008 在本仓库中定义为 current snapshot beam selection：当前 `coord/image/lidar/ray` 预测当前 beam class，同时预测 LOS/NLOS 与 link quality。运行说明见 `docs/Raymobtime_s008_selection.md`。

DeepVerse DT31 的价值在于把监督来源和模型输入解耦：用 ray-tracing channel 生成 beam label，用 LoS/status 生成 blockage label，用 mobility ground truth 生成 trajectory label，但模型默认只输入 camera、LiDAR、radar、weak wireless history 和 noisy position。先完成 cache/manifest/label，再做训练和 task-aware gate 实验。契约见 `openspec/specs/deepverse-dt31-data-generation/spec.md`。

Viewer 已从静态 PNG 方向收束为 Gradio manifest workflow。重点能力包括 raw/processed modalities、prediction/quality/gate 合并和 future beam distribution。入口和字段约定见 `tools/visualization/README.md`。

## 原论文复现边界

如果目标是复现原始上游代码或 `All_models/` 历史权重，需要特别注意：

- 原始角色化 GRU 层数和当前 canonical 配置可能不同；加载旧权重时应显式报告 missing/unexpected keys。
- `strict=False` 会掩盖结构漂移，做复现时应优先使用严格加载或清晰的兼容报告。
- batch size、seed、learning rate 等当前配置不等价于原始实验参数。
- 当前包结构和 CLI 已经重构；旧顶层脚本入口不再作为维护入口。

## 已收束的历史草稿

以下根目录草稿已合并到本文、`docs/` 或 OpenSpec，不再单独保留：

- 已退役的模态失衡研究草稿和 `G2D对比实验.md`
- `CSI编码器.md`、`CSI模态加噪方案.md`、`CSI模态处理对比实验.md`
- `gps模态处理方案.md`、`LiDAR模态处理方案.md`、`LiDAR模态读取方案.md`、`mmWave模态处理方案.md`
- `Raymobtime数据集.md`、`deepverse6g.md`
- `可视化程序方案.md`、`可视化程序修改建议.md`、`可视化程序置信度部分.md`
- `与原论文差异.md`
