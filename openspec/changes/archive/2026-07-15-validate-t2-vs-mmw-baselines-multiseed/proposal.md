## Why

MMW seed1 已显示 T2 在 clean、整模态缺失和时间缺失上显著优于 AMBER-Full 与 RMBP-MM 本地公平适配，但现有证据仍是单随机种子 local validation，且 PCA/Isomap 特征图没有直接对应最终 beam 决策。需要补齐三方法 seeds1-3，并用相同样本、相同 mask 的任务输出诊断验证优势是否稳定，避免用不对齐的特征几何替代性能证据。

## What Changes

- 将 MMW 15-domain launcher 扩展为显式 `methods × seeds` 作业矩阵，补跑 T2、AMBER-Full、RMBP-MM 的 seed2/3；保持 seed1 的数据、四传感器输入、domain-balanced sampler、40 epoch、固定 `last.pth` 和缺失增强协议不变。
- 让 MMW evaluator 可显式消费 seed，并为 clean 与固定 temporal mask 保存逐样本 target/logits/prediction；所有方法和 seed 通过 domain、CSV checksum、样本位置和 mask digest 严格配对。
- 新增 local/manual 任务输出鲁棒性汇总，报告绝对 Top1、相对 clean 保持率、三方法共同 clean-correct 样本保持率、正确类 logit margin、clean/missing JS 距离、圆周 beam 误差和 15-domain T2-minus-baseline 热力图。
- 输出逐 seed 与三 seed mean/std、domain-macro、最差域和分组 bootstrap 区间；极端 85/90/95% modality-frame 结果继续与 0-80% 三 mask-type 主曲线分开。
- 保留 AMBER-Full 与 RMBP-MM 的 local-adaptation / out-of-paper-scope 声明，不将结果描述成原论文等价复现；不增加传感器噪声、模型模块、训练 loss 或 BPA-off 消融。

## Capabilities

### New Capabilities

- `mmw-baseline-multiseed-robustness-evidence`: 定义 T2 与两个本地适配 baseline 的三随机种子公平训练、逐样本配对任务输出诊断、统计汇总和论文 claim 边界。

### Modified Capabilities

无。

## Impact

- 本地实验编排与诊断：`scripts/launch_mmw_all_weather_matrix.py`、`scripts/eval_mmw_all_weather_matrix.py`、一个最小任务输出汇总/绘图脚本及 focused tests。
- 训练、checkpoint、逐样本 logits、CSV、图片和 Markdown 全部写入 ignored `outputs/`；不新增公共 CLI、依赖或实体 canonical YAML。
- GPU0-5 并行运行六个 seed2/3 作业，GPU6-7 保留空闲或用于后续只读评估分片。
