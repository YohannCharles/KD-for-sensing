## 1. Baseline 与契约刻画

- [x] 1.1 用历史源码、冻结 config 和本地公平评估产物记录 S1 exact semantics、Drop0-80 seed1 指标、参数/显存/时长口径，且不修改历史 outputs。
- [x] 1.2 新增 focused characterization，证明 current disabled U-Mask 行为不变、旧 S1-S4 `temporal_router_type` 继续拒绝、新 S1 masked mean 按 `[B,T,M]` mask 聚合。

## 2. 轻量模型实现

- [x] 2.1 实现 coverage、last-age、longest-gap、trailing-gap 和 missing-block count 的统一 mask statistics helper 与边界测试。
- [x] 2.2 在 U-Mask 中接入显式 opt-in `temporal_pooling`，实现 masked mean S1 baseline 与 shared runtime/ModelOutput/metadata 透传。
- [x] 2.3 实现 fixed-recency pooling 和 gap-aware residual pooling，覆盖缺失 hard mask、eta=0 exact fallback、单 cell、全空拒绝、有限 backward 和小于 0.03M 参数测试。
- [x] 2.4 将 mask statistics 作为独立 opt-in router features 接入现有 supervised router，并保持 disabled router state dict shape。
- [x] 2.5 实现 coverage-aware uniform shrinkage，覆盖 full rho=0、不可用权重0、权重和1、单模态恒等、rho上限与 diagnostics。

## 3. Superset 训练策略

- [x] 3.1 扩展 temporal operator：只在 opt-in 时保存 zero-fill 前 tensor 引用和 base mask，验证不 clone、`M- subseteq M+`、student 零填充和 disabled 零 payload。
- [x] 3.2 扩展 U-Mask training extension：复用同一 primary model 执行 no-grad/eval-mode superset forward，恢复模型状态并共享 teacher output，且不修改 trainer 主循环。
- [x] 3.3 实现 confidence-gated temperature KL，覆盖 teacher correctness/entropy gate、温度平方、加权归一、feature L2=0 和 diagnostics。
- [x] 3.4 实现 circular beam-risk monotonic ranking，覆盖 wraparound、risk tolerance、正确 student 梯度方向、violation rate、disabled behavior 和与 KD 共享 teacher forward。
- [x] 3.5 更新 loss/config metadata 与 distillation-free guard，确保不出现外部/checkpoint teacher、distiller registry 或 legacy `distillation.*`。

## 4. H5/P1 工作流

- [x] 4.1 参数化扩展现有 H5/P1 launcher，保留默认五方法并增加 S1 lightweight 八任务 profile；不新增 S1-S4 wrapper。
- [x] 4.2 扩展 eval/summary 解析实际可用的 Top1、Top3、Within@3、ADBA、MAE、pooling、teacher/ranking 和 router diagnostics，并输出 Drop0 guardrail 状态。
- [x] 4.3 用独立临时 root dry-run，确认 seed1 八任务分别映射 GPU0-7、`max_jobs=8`、`per_gpu=1`、相同 split/epoch/optimizer/sampler 和独立 output/log。
- [x] 4.4 在 GPU0-7 对 S1/T2 运行 batch/thread 吞吐基准，选择 batch64、intra-op 12、inter-op 1 和 persistent workers，并保持默认 profile 资源语义不变。

## 5. 验证与 Smoke

- [x] 5.1 运行 `openspec validate improve-s1-lightweight-temporal-robustness --strict` 和 `openspec validate --all --strict`。
- [x] 5.2 运行 `conda run -n kd_mm_beam pytest` 的 U-Mask、temporal missing、H5/P1、prediction objective、evaluation 和 config focused tests。
- [x] 5.3 运行 `make verify-quick`，并确认未改写用户已有 MMW/H5P1 工作树修改或纳入 dataset/outputs/logs/checkpoint。
- [x] 5.4 在 GPU0-7 上以每卡一任务运行八方法 1-epoch smoke，确认 forward/backward、显存、checkpoint 和 manifest 完成状态。

## 6. Seed1 八卡筛选

- [x] 6.1 使用独立 ignored root，在 GPU0-7 每卡一进程并行运行 S1、T1、T2、A1、A2、A3、T1+T2 和 J1 seed1 正式训练。
- [x] 6.2 使用相同 Scene31-34 validation split 与固定 temporal mask cache 并行评估八个 checkpoint，不用 test split 调参。
- [x] 6.3 汇总五档 mean Top1、Drop0-60 mean、Drop80、Top3/ADBA 与 diagnostics，按 Drop0 降幅不超过 0.005 判定晋级候选。

## 7. 条件式多 Seed 与 J2

- [x] 7.1 完成 seed1 门禁决策：J1 主指标与 Drop80 为负，J2 记为 ineligible；用户指定 T2 为主线，T1 保留严重缺帧/整模态缺失支线，T1+T2 不继续。本阶段只生成 T2/S1 seeds2/3，并由后继 change `validate-t2-beam-geometry-and-head` 接管。
- [x] 7.2 后继 change 已完成 LG/CLS 四候选 seed1 训练；current S1/T2 seeds2/3 在后续 MMW 资源切换决策中于 epoch 5/6 停止，未完成部分明确记为 cancelled，不作为 checkpoint 或 claim 证据。
- [x] 7.3 后继 change 已将三 seed mean/std、配对差值和主线判定记为 `unavailable/cancelled`；结果保持 local experimental，不更新正式 claim。

## 收口结果

- 本 change 的源码、focused tests、seed1 八方法筛选和 T2 主线选择已经完成。
- 条件式多 seed 阶段由后继 change 接管后，被 `run-mmw-all-weather-missing-modality-matrix` 记录的用户资源切换决策停止；该取消不回写为实验完成，也不支持三 seed claim。
