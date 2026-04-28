## Context

项目目前通过注册表和 YAML 配置支持 image、radar、GPS 及可选 fusion。数据集 `Scenario9Dataset` 统一返回 dict batch，训练流程根据 `experiment.task` 和 fusion `modalities` 准备输入；模型侧每个模态遵循 `FeatureExtractor`、teacher `ModalityNet`、student `StudentModalityNet` 的风格，并返回 `(pred, features, output_features)`。

LiDAR 方案建议优先采用 ROI/FoV 裁剪、BEV 伪图像化和轻量增强。结合当前项目的 image/radar CNN+GRU 形态，第一版应将 LiDAR 点云转成 `[T, C, H, W]` 的 BEV 张量，再复用 CNN frame encoder + GRU temporal modeling + classifier 的接口，而不是直接引入 PointNet、SparseConv 或复杂 Transformer。

## Goals / Non-Goals

**Goals:**
- 支持从序列 CSV 读取 LiDAR 路径，并在 dataset 中返回 `lidar` batch 字段。
- 支持 LiDAR BEV 预处理：无效点过滤、ROI/FoV 裁剪、可选地面/背景过滤、height/intensity/density 三通道伪图像化、训练集归一化。
- 新增 `LidarFeatureExtractor`、`LidarModalityNet`、`LidarStudentModalityNet`，注册名分别与现有风格一致。
- 支持 `experiment.task: lidar` 的训练、验证、评估和 no-KD/logits-KD/RKD 配置。
- 支持 fusion `modalities` 包含 `lidar`，并保持旧 image/radar/GPS 配置兼容。

**Non-Goals:**
- 不在第一版实现 PointNet、PointPillars、SparseConv 或 raw point cloud 模型。
- 不把 angle-bin/range vector 作为默认 LiDAR 输入；该方向仅作为后续消融预留。
- 不强制引入 Open3D 等大型依赖；第一版优先支持 DeepSense 常见 ASCII PCD/文本点云和 `.npy` 点云数组。
- 不改变现有 image、radar、GPS 模态的默认训练语义。

## Decisions

1. **默认 LiDAR 表征使用 BEV 伪图像**

   LiDAR 点云先映射到固定大小 BEV 网格，默认通道为 max height、max/mean intensity、log-normalized density。这样可直接适配现有 CNN 特征提取和 fusion 分支，输入形状也与 image/radar 统一。相比 raw point cloud 模型，BEV 对当前样本规模和训练框架更稳；相比 angle-bin，BEV 保留更多空间遮挡和车辆几何信息。

2. **预处理函数放入数据/预处理层，模型只消费 BEV 张量**

   `src/kd_sensing/data/transforms.py` 负责样本级读取和 BEV 构造，`src/kd_sensing/preprocessing/` 负责可离线运行的 CSV/BEV 生成器。dataset 可按配置选择在线读取点云生成 BEV，或读取预先缓存的 BEV `.npy`。模型不解析点云文件，避免模型层绑定数据格式。

3. **LiDAR 模型结构与现有模态保持一致**

   `LidarFeatureExtractor` 使用卷积、注意力或 adaptive pooling 输出每帧 `feature_size`；`LidarModalityNet` 使用 feature extractor、LayerNorm、GRU、可选 MHA/temporal attention 和 classifier；`LidarStudentModalityNet` 使用 depthwise separable convolution + adaptive pooling 的轻量路径。三者命名和返回值与 image/radar/GPS 对齐，便于 KD 与 fusion 复用。

4. **Fusion 扩展采用统一分支注册，不复制训练循环**

   `VALID_FUSION_MODALITIES` 扩展为 `("image", "radar", "gps", "lidar")`。fusion teacher 使用 `LidarFeatureExtractor` 作为 LiDAR 分支；fusion student 使用轻量 LiDAR CNN 分支并将 pooled embedding 与其它模态拼接。`prepare_fusion_inputs` 根据配置准备 LiDAR 输入，不启用 LiDAR 时不要求 batch 存在该字段。

5. **背景过滤作为可选配置，不作为默认硬要求**

   LiDAR 方案指出背景过滤有价值，但最优配置未必启用。第一版默认启用 ROI/FoV 裁剪和 BEV，背景过滤通过配置项打开，可使用 occupancy frequency 或场景级 moving average 生成静态背景 mask。这样能先跑通主路径，再做 `BEV`、`BEV+FoV`、`BEV+FoV+background` 消融。

6. **训练归一化遵循 GPS scaler 的训练集原则**

   LiDAR BEV 归一化统计只从训练集估计，测试集复用训练统计。dataset 构建流程需要像 GPS scaler 一样把 train-fitted LiDAR normalizer 传给 test split，避免验证/测试信息泄露。

## Risks / Trade-offs

- **点云文件格式不统一** → 先实现 ASCII PCD、文本/CSV 点云和 `.npy` 的清晰错误处理；二进制 PCD 通过错误信息提示需要离线转换或后续可选依赖。
- **在线 BEV 构造拖慢训练** → 提供缓存目录和离线 BEV 预处理器；配置可选择读取缓存 `.npy`。
- **ROI/FoV 参数错误导致 LiDAR 变成噪声模态** → 配置显式暴露 ROI/FoV 范围，并增加单元测试验证裁剪和输出 shape；默认参数保守，允许按场景覆盖。
- **水平翻转会破坏 beam label** → 第一版只实现点 dropout 和 jitter 这类不改 label 的增强；多模态 flip 必须在所有模态和 label 同步变换后再启用。
- **新增模态影响旧配置** → 旧配置默认不启用 LiDAR；dataset 仅在 `use_lidar` 或 task/modalities 需要 LiDAR 时校验 LiDAR 列。

## Migration Plan

1. 新增 LiDAR 预处理、dataset 字段、模型和 fusion 分支，并补齐测试。
2. 新增 `configs/preprocess/*lidar*.yaml`、`configs/lidar/*.yaml` 和含 LiDAR 的 fusion 示例配置。
3. 使用 synthetic/fixture 点云运行 smoke tests，再使用小比例真实数据验证 `conda run -n kd_mm_beam python scripts/train.py --config ... -o training.epochs=1`。
4. 旧 image/radar/GPS 配置无需迁移；若用户要启用 LiDAR，需要重新生成带 `lidar1..lidarN` 的序列 CSV 或提供 BEV 缓存路径列。

## Open Questions

- DeepSense 场景 33 的 LiDAR 原始列名是否固定为 `unit1_lidar`，还是不同数据包存在其它列名。
- 当前本地数据集是否包含 ASCII PCD、二进制 PCD、`.bin` 还是已转换 `.npy` 点云；实现时需要根据实际样例确认 reader 覆盖范围。
- 默认 ROI/FoV 数值应以场景 33 还是更通用的场景 31-34 为基准。
