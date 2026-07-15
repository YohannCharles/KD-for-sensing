## Context

MMW Town03 五场景共享 64-beam ULA/DFT label，但 RSU yaw 分别约为 -135、-90、45、-30、135 度。当前 YAML GPS loader 直接对 UE GPS 与 BS position 的世界坐标差计算 `atan2`，且主实验配置使用 `relative_polar`，因此五场景 pooled 模型需要同时学习五套旋转后的 GPS 到 beam 映射。只读反事实表明，使用真实最后历史帧 GPS 时，按 RSU yaw 转为局部角可将 pooled ±3 beam 的 1-NN 准确率从 56.90% 提高到 86.97%，但该结果仍需 matched T2 训练确认。

现有 DeepSense 文本 GPS 没有 RSU pose yaw，既有 `relative_polar` 又是已发布输入契约，因此本变更必须 opt-in、MMW YAML 专用并保持旧数值完全不变。

## Goals / Non-Goals

**Goals:**

- 提供可配置、可测试的 MMW RSU 局部相对极坐标输入，保持 `[T, 3]` shape 和 GPS encoder 不变。
- 从每帧权威 BS YAML 读取 yaw，避免依赖场景名或手写常量。
- 让 mode、yaw source、scaler 和实验配置可审计。
- 用相同代码版本重跑 world/local 两个 T2 seed1，并复用同一 missing mask cache 做配对比较。

**Non-Goals:**

- 不用 GPS 几何直接生成 beam label，不修改 64-beam circular topology 或 prototype loss。
- 不为缺少 RSU yaw 的 DeepSense 文本 GPS 猜测朝向。
- 不改变 seq_len、GPS 维度、网络结构、split、天气标签或缺失采样协议。
- 不将 seed1 local validation 升级为正式多 seed claim。

## Decisions

### 1. 新模式 opt-in，旧模式逐值兼容

新增 `rsu_local_relative_polar`，默认仍为 `relative_polar`。新模式只改变角度参考系，距离和输出顺序保持不变。这样旧 config/checkpoint 无需迁移，matched 对照也能在同一代码版本中运行。

替代方案是直接修正 `relative_polar`，但这会静默改变所有历史配置和 scaler 语义，无法公平复现旧结果，因此拒绝。

### 2. 每帧读取 `sensors.rsu_pose.rotation.yaw`

局部角定义为 `wrap(atan2(ue_y-bs_y) - yaw_rsu)`，最终仍编码为 `sin/cos`。yaw 与 BS XY 来自同一个 `bs_gpsN` YAML；每帧读取并校验同一历史窗口内 yaw 在数值容差内一致。Town03 契约是静态 RSU，缺少、非有限或窗口内不一致的 yaw 均 fail closed。

不使用 `lidar_pose`、camera yaw 或场景常量，因为 Town03 中这些安装朝向并不等同于阵列/RSU pose 契约。也不硬编码额外 beam offset；局部角到 DFT beam 的非线性和多径关系继续由模型学习。

### 3. 复用现有相对极坐标与训练集 scaler

实现只扩展 GPS YAML pose 读取和 mode dispatch，复用现有 `_relative_polar_features`。新模式仍输出三维 float32，现有 GPS encoder、batch 和训练集 scaler 逻辑不变；不同 mode 必须各自 fit 并保存自己的 train-only scaler。

帧级 cache 同时缓存 RSU XY 和 yaw，避免五帧重叠窗口重复解析 YAML。持久化 sample cache 必须由 resolved config/mode 区分，禁止把 world mode cache 用于 local mode。yaw 只作为输入 provenance，不写入 scaler 数值工件。

### 4. 两层配对验证后才判断收益

第一层使用 frozen H5/P1 train/test 和实际最后历史帧 GPS，比较 world/local 的非参数 circular beam 误差，验证坐标变换。第二层在相同 15-domain inventory、seed1、40 epoch、optimizer、domain-balanced sampler、missing augmentation 和 `last.pth` policy 下并行训练 world/local T2，随后使用同一 v2 mask cache评估 clean、whole-modality 与 temporal missing。

历史 world checkpoint只作参考；最终主要 delta 使用本变更后重跑的 world checkpoint，减少代码和运行环境漂移。若资源不足，允许先完成特征诊断和双分支 smoke，但不得据此宣称 T2 已改善。

## Risks / Trade-offs

- [GPS 测量噪声使 Exact Top1 提升有限] → 同时报告 ±1、±3、平均 circular error、clean Top1 与 missing robustness，不只看 exact 1-NN。
- [RSU pose yaw 与物理阵列仍可能有固定安装偏置] → 本变更只使用数据集权威 pose；不引入由 test label 拟合的 offset。若后续需要 offset，必须仅用 train split 校准并另提 change。
- [重跑 40 epoch 耗时] → world/local 两进程并行使用空闲 GPU，先做 loader、one-batch 和短 smoke，失败时不启动正式训练。
- [旧 sample cache 混用] → 对照配置关闭旧 sample cache或使用 mode 隔离的新 cache key，并在运行 provenance 中记录 mode。
- [GPS 修复不能消除 router pattern bias] → GPS 坐标修复和 router 反事实分别报告，不把所有负增益都归因于坐标系。

## Migration Plan

1. 增加 mode/yaw parser 和 focused tests，确认旧 `relative_polar` 数值不变。
2. 更新 config/runtime contract 与 metadata，运行 OpenSpec、GPS dataset、MMW 和 config focused checks。
3. 生成 world/local 两份 ignored 配置，执行真实单样本、one-batch 和短 smoke。
4. 在空闲 GPU 上并行运行 matched T2 seed1，冻结各自 `last.pth` 后使用共享评估协议比较。
5. 保留旧模式为默认；回滚只需将实验配置切回 `relative_polar`，无需转换 checkpoint 或数据。

## Open Questions

无。若 paired T2 结果显示 local GPS 仍为负贡献，再单独检查 router calibration；本变更不预先扩张到模型结构。
