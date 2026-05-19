## Context

项目当前的训练与预处理主线围绕 DeepSense6G：CSV 中保存模态相对路径，dataset 按启用模态懒加载 image、radar、GPS、LiDAR、mmWave power vector，并从 beam power 文件 `argmax` 得到 beam 标签。MMW 数据的官方 V2I 结构不同：传感器数据按 `Town/scenario/agent/frame` 组织，CAV 提供四路 RGB、LiDAR、yaml 和 `_paths` 信道文件，RSU 提供 LiDAR、相机、yaml 和 radar JSON；信道数据包含多径增益、时延、AoD/AoA 等字段。现有 layout 已为 `dataset/MMW/<condition>/Sensor_Data` 和 `dataset/MMW/<condition>/Channel_Data` 预留目录，但还没有可运行的数据准备流程。

本变更面向用户已在本地持有的 `Town10_skybridge_seed24.zip` 与信道 `Town10.zip`。当前仓库内未包含这两个 zip，因此实现必须通过配置或 CLI 参数接收输入路径，并避免把大数据文件纳入源码管理。

## Goals / Non-Goals

**Goals:**
- 从本地传感器 zip 和信道 zip 生成项目可消费的 MMW Town10 skybridge 数据目录、manifest/CSV、beam power vector、beam 标签、split metadata 和 sanity report。
- 保留 MMW 原始文件层级，另行生成轻量派生产物，避免复制大图像、点云或原始信道文件。
- 默认输出与现有 beam 任务兼容的 `beam*`、`future_beam*` 和 `mmwave*` 路径列，beam label 继续由 64 维 power vector 的 `argmax` 派生。
- 支持按启用模态懒加载 MMW 数据，未启用模态的缺失文件不阻塞无关任务。
- 记录足够 metadata，使信道到 beam 的 codebook、天线数量、频率、zip 摘要、跳过原因和 split 可复现。

**Non-Goals:**
- 不下载、移动或提交真实 MMW zip。
- 不复现 CARLA、Blender 或 Sionna 数据生成流程，只消费已有 zip。
- 不引入 TensorFlow/Sionna 作为数据准备运行时依赖；只用 NumPy 从保存的 channel/path 数组派生标签。
- 不在本变更中训练模型或调优 MMW 模型结构。
- 不改变 DeepSense6G 的默认目录、CSV 协议或标签语义。

## Decisions

1. 使用独立 MMW 准备模块与脚本，而不是扩展 DeepSense6G 预处理脚本。

   新增 `src/kd_sensing/data/mmw/` 保存 MMW zip 索引、metadata 解析、channel label 派生、manifest/split/sanity 写出；新增 `scripts/mmw/prepare_town10_skybridge.py` 或等价 CLI。MMW 和 DeepSense6G 的原始数据结构差异较大，独立模块能减少 DeepSense6G 场景逻辑的条件分支。共用部分只复用通用 layout helper、codebook helper、split metadata 和 mmWave feature 读取逻辑。

   备选方案是把 MMW 作为 DeepSense6G 新 scene 接入。该方案短期能复用 dataset，但会把 MMW 的 agent/frame/channel 规则塞进 DeepSense6G 命名空间，后续维护成本高。

2. 原始文件保留官方层级，派生产物写到 `Prepared/`。

   解包后保留 `Sensor_Data/Town10/Town10_skybridge_seed24/...` 和 `Channel_Data/Town10/...`。脚本在同一 condition 下写 `Prepared/Town10_skybridge_seed24/`，包含 `beam_power/`、`manifests/`、`splits/`、`metadata.json` 和 `sanity_report.json`。CSV 中路径统一相对 MMW condition root，便于 dataset 与诊断工具按同一根目录解析。

   备选方案是把派生 power vector 写回 CAV agent 目录。该方案路径更短，但会混淆原始数据和再生成数据，也不利于清理或对比不同 codebook 配置。

3. beam 标签由保存的 channel/path 数组离线派生，默认复用项目现有 64-beam DFT codebook。

   MMW channel 文件保存多径复增益 `a` 以及本地/全局 AoD/AoA。实现应优先从 channel 文件中的复信道系数构造发射端等效 channel，并用 `kd_sensing.data.deepverse.codebook` 的 DFT codebook 计算 64 维 beam gain；如果文件只提供多径角度和增益，则使用配置化 steering vector 从 AoD 与 path gain 合成 beam power。每个输出 power vector 保存为现有 loader 可读的 64 个浮点值，top-1 label 为 `argmax` 的 0-based beam index。metadata 必须记录算法、`num_beams`、推断或配置的天线数量、使用的 channel 字段和失败原因。

   备选方案是直接用最强路径 AoD 量化成 beam label。该方案实现简单，但不能提供现有 mmWave 输入需要的 64 维 power vector，也会丢失多径叠加信息。

4. manifest 使用现有序列列协议，并保留 MMW 扩展字段。

   兼容列包括 `camera1..cameraN`、`lidar1..lidarN`、`gps1..gpsN`、`mmwave1..mmwaveN`、`beam1..beamN`、`future_beam1..future_beamH` 和 `seq_index`。默认 `camera*` 指向 CAV `camera0` 前向 RGB；四路相机的原始路径另写扩展列或 JSON metadata，避免当前 image encoder 一次性被迫支持多视角输入。RSU radar JSON、RSU camera/depth 和 agent metadata 也写入 manifest metadata 或可选列，供后续诊断和模型扩展使用。

   备选方案是立即把四路 CAV 相机建模成新 image 张量契约。该方案超出本次“生成各模态数据和 beam 标签”的核心目标，会牵动模型、transform 和配置矩阵。

5. split 以 agent/连续片段为组，默认不跨 agent 和不跨不连续帧生成窗口。

   每个 CAV agent 独立形成时间序列。窗口必须使用同一 agent 内连续帧，历史长度和预测长度由配置控制。默认 split 以 `seq_index` 或连续片段为最小单位，避免同一连续滑窗同时出现在 train/test 中。split metadata 记录 seed、比例、agent、seq 分配和 beam 分布。

## Risks / Trade-offs

- [Risk] 官方 zip 内信道扩展名可能是 `_paths.npy` 或 `_paths.npz`，且字段 shape 可能随生成脚本变化。→ Mitigation：loader 同时支持 `.npy`、`.npz` 和 dict-like payload，严格校验必需字段与 shape，并在 sanity report 中记录实际字段。
- [Risk] Town10 信道 zip 的目录层级可能与传感器 zip 不完全一致。→ Mitigation：先建立 frame/agent 索引，再用可配置 pattern 匹配 channel 文件；匹配失败按 frame 和 agent 记录跳过原因。
- [Risk] 从多径数据派生 beam power 的公式会影响标签分布。→ Mitigation：默认复用项目已有 DFT codebook helper，metadata 固化算法版本，并提供小夹具测试保证 deterministic；后续如果采用论文指定 codebook，可通过配置版本化输出目录。
- [Risk] MMW 多视角相机和 RSU 模态超出现有 dataset 输入契约。→ Mitigation：首版使用前向 CAV RGB 作为兼容 image 输入，同时把其余视角和 RSU 模态保留在 manifest 扩展字段中。
- [Risk] 大 zip 解包和扫描耗时较长。→ Mitigation：支持 manifest cache、输入 zip 摘要、`--force` 重建和 dry-run/sanity-only 模式。

## Migration Plan

1. 新增 MMW layout helper 和准备配置，不改变 DeepSense6G 默认行为。
2. 实现 zip 索引、解包校验和 frame/agent 对齐，先产出 dry-run sanity report。
3. 实现 channel-to-beam 派生与 power vector 写出，并生成 frame-level manifest。
4. 实现序列 CSV、split metadata 和 MMW dataset/manifest 读取接入。
5. 补充测试和 README 命令，所有项目相关命令使用 `conda run -n kd_mm_beam ...`。

Rollback 策略：删除或忽略 `dataset/MMW/<condition>/Prepared/Town10_skybridge_seed24/` 派生产物，并移除 MMW 配置入口即可；DeepSense6G 训练路径不受影响。

## Open Questions

- `Town10_skybridge_seed24.zip` 对应的 condition 是 `sunny` 还是需要由用户显式传入？本提案默认通过配置传入，默认值为 `sunny`。
- 信道 zip 中是否包含所有 CAV 的每帧 channel 文件，且文件名是否保持官方 `_paths` 约定？实现阶段需要用真实 zip 做 smoke 验证。
- 是否需要首版训练直接使用 RSU radar JSON？本提案先生成和索引 RSU radar 路径，模型输入接入可以按现有 radar 特征契约另行扩展。
