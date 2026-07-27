## Why

当前 MMW 正式研发协议只支持按域内时间前后段划分，不能衡量对完整未见轨迹的泛化，并可能在按 CAV 拆分时遗漏共享 RSU Radar、BS-GPS 等资源耦合。需要新增一个固定、可审计、test 默认封存的轨迹互斥协议，使普通模型比较建立在同一组完整 trajectory groups 上。

## What Changes

- **BREAKING**：MMW 普通模型研发主协议改为 `mmw_trajectory_disjoint_v1`，不再把 chronological tail split 作为唯一正式研发协议。
- 从 15 个 Town3 domain 的 46,860 个候选窗口生成来源清单；显式 run 元数据优先，否则基于场景执行与共享资源图重建 trajectory groups。
- 在完整 group 层面以 seed 2026 确定性生成 80/10/10；恰好 50 组时严格为 40/5/5，其他数量保证 validation/test 至少各一组。
- 对 train/validation/test 两两执行 sample、target、CSV row、依赖帧、四模态资源、channel 审计身份、trajectory 和场景执行的零交集审计；失败时禁止训练。
- 生成可复核 manifest、hash、分布统计、异常记录、历史暴露与 claim eligibility；所有内容写入本地 `outputs/`，不进入源码。
- 默认训练只暴露 train/validation；test 评测必须显式授权，本 change 不执行 test 推理。
- 复用 Candidate12 的 encoder/fusion，提供 M0 线性、M1 普通 prototype、M2 topology prototype、M3 topology prototype + random-balanced 四个固定基线及 GPU 0--3 启动、监控和 validation 汇总。
- 保持现有 5 帧历史、1 帧预测、64 beam、四模态输入、预处理、优化目标以外的模型和数据定义不变；channel/path/beam power、历史 beam index、未来 GPS 均不作为模型输入。

## Capabilities

### New Capabilities

- `mmw-trajectory-disjoint-protocol`: 定义 trajectory unit 重建、group-level split、资源泄漏审计、历史暴露、test 封存、协议产物与公平基线要求。

### Modified Capabilities

- `clean-data-integrity`: 将 MMW 正式研发数据域从仅允许 clean-inner 扩展为可精确绑定且通过审计的 trajectory-disjoint train/validation，并继续禁止 validation/test 参与拟合状态。
- `repo-boundaries`: 允许一个项目内研究工具和脚本实现新协议，同时继续禁止新增 public CLI、提交本地数据与运行产物。
- `u0-mainline`: 允许限定的 M0--M3 轨迹协议基线复用当前 Candidate12 公共架构，但不扩大 canonical recipe 或恢复退役模型路线。

## Impact

- 数据协议 owner：`src/kd_sensing/data/mmw/` 与 `src/kd_sensing/engine/`。
- 本地研究执行面：`tools/`、`scripts/` 和 `tests/test_mmw_trajectory_split.py`。
- OpenSpec：新增协议 capability，并更新三个 current capability 的 delta requirements。
- 不新增依赖，不修改 public train/evaluate/preprocess CLI，不跟踪 `dataset/`、`outputs/`、日志或 checkpoint。
