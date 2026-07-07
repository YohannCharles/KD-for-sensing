## Why

`overnight_branch_router_v2`、PCPG 和 BPRR 已经把最终候选收敛到 `c2 = supervised router + soft hard-subset weighting + beam prototype alignment`。现在需要一次性冻结最终主方法消融矩阵，补齐显式 ablation 开关、统一 launcher/summary 和 focused tests，用同一输出根目录回答论文主问题。

## What Changes

- 新增 `final_c2_ablation_v1` 本地实验矩阵，覆盖主方法、多 seed 最强对照、router/prototype/fusion/pattern-weighting/negative trade-off 消融。
- 补齐显式 opt-in 的 router feature、prototype/head、fusion baseline 和 hard subset weighting 配置开关；默认训练行为保持不变。
- 新增 `scripts/launch_final_c2_ablation_v1.py`，生成 67 个 job，支持 GPU0-7、每卡 1 进程、dry-run、skip/force、实验/seed/max_epochs 过滤、manifest 和 failed_jobs。
- 新增 `scripts/summarize_final_c2_ablation_v1.py`，聚合新 root 与历史 baseline roots，写出主结果、router/prototype/fusion/pattern/negative trade-off 表和自动结论。
- 新增 focused tests 覆盖开关、fusion mask、soft_static、launcher dry-run 和 summary parser。

## Capabilities

### New Capabilities
- `final-c2-ablation-v1`: 定义最终 c2 消融实验矩阵、显式 opt-in ablation flag、统一 launcher/summary artifact 和 focused test 契约。

### Modified Capabilities
- `pcpg-radar-balance-robustness`: 扩展现有 PCPG/router/BPRR 本地实验能力，补充 average fusion、router feature ablation、prototype/head ablation 与 final summary 字段要求。

## Impact

- 影响 `src/kd_sensing/models/u_mask_beam_jepa.py`、`src/kd_sensing/engine/pcpg_radar_balance.py`、`src/kd_sensing/losses/u_mask_beam_jepa_config.py`、`src/kd_sensing/cli/train.py`、本地 `scripts/` launcher/summary 和 focused tests。
- 不新增 package console script，不复制训练框架，不修改旧 outputs，不把训练日志、checkpoint 或 outputs 产物纳入源码变更。
- 所有训练/测试命令继续使用 `conda run -n kd_mm_beam ...`。
