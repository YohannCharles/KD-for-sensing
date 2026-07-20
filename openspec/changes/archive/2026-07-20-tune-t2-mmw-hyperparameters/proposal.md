## Why

当前 MMW T2 的结构性消融已支持 prototype head、BPA 与 circular DFT 码本先验，但训练仍使用固定学习率和未校准的辅助目标/缺失采样比例。需要在不改变 T2 架构、数据划分或正式评估协议的前提下，进行可审计的开发筛选，选择一个冻结配置进入后续多 seed 正式复验。

## What Changes

- 新增 MMW T2 超参数开发筛选矩阵：固定 15 个场景天气域、40 epoch、epoch-40 `last.pth` 和现有 T2 结构，仅改变预注册的 BPA、temporal mask、优化日程或 superset KL。
- 新增最小 launcher：生成带 provenance 的本地 YAML、按 GPU 分配任务、支持统一 batch-size 探测，并记录每个任务的显存探测和训练状态。
- 固定筛选选择规则：同时衡量 Clean、Drop1/2/3、temporal AUC 和 temporal Drop80；筛选产物标记为 development evidence，不能直接升级为论文主结论。
- 筛选期间允许每 5 epoch 观察 validation，但比较仍固定使用 epoch-40 `last.pth`；不启用 early stopping，不以最佳 checkpoint 替代固定预算。

## Capabilities

### New Capabilities

- `mmw-t2-hyperparameter-screening`: 定义冻结 T2 架构下的可复现 MMW 配置筛选、显存探测、GPU 并行、选择指标和证据边界。

### Modified Capabilities

无。

## Impact

- 新增一个本地 MMW 调参 launcher 与相应测试，复用现有 `kd-sensing-train`、MMW all-weather evaluator 和 generated-config 模式。
- 读取 tracked `configs/mmw/t2.yaml` 及 shared base 作为基线配置来源；新 YAML、日志、checkpoint、评估结果均写入 ignored `outputs/`。
- 不新增依赖、不修改默认 CLI、不改变已完成 T2、S1、AMBER-Full 或 RMBP-MM 产物。
