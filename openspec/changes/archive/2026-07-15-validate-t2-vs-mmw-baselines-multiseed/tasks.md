## 1. 多seed训练编排

- [x] 1.1 扩展MMW launcher的`build_config`与CLI，支持显式methods、seeds、GPUs、独立config/log/manifest和冲突fail-closed，同时保持seed1默认行为。
- [x] 1.2 增加focused tests，验证训练行为seed同步、split/inventory不变、六作业GPU映射、重复目标拒绝和dry-run不启动训练；使用`conda run -n kd_mm_beam pytest ...`。
- [x] 1.3 dry-run并在GPU0-5并行启动T2、AMBER-Full、RMBP-MM seed2/3，确认六个进程、显存、日志与run directory正确。

## 2. 多seed固定mask评估

- [x] 2.1 扩展MMW evaluator按seed发现config/checkpoint并写入seed分层目录，保持共享v2 mask cache、15-domain和baseline scope provenance。
- [x] 2.2 扩展或新增最小summary，输出逐seed、三seedmean/std、0-80 type-equal AUC、Drop80、极端稀疏曲线、T2-minus-baseline和门禁状态。
- [x] 2.3 增加focused tests并用`conda run -n kd_mm_beam pytest ...`验证跨seed mask identity、缺checkpoint unavailable和分层聚合。

## 3. 逐样本任务输出诊断

- [x] 3.1 扩展现有融合特征extractor支持seed，并保存稳定sample id与float32 logits；跨mask label/sample顺序不一致时fail closed。
- [x] 3.2 新增最小任务输出summary/绘图，生成全样本绝对性能、relative-clean、pairwise/three-way common-clean、margin delta、归一化JS、圆周误差和15-domain差值热力图。
- [x] 3.3 增加focused tests并用`conda run -n kd_mm_beam pytest ...`验证严格样本配对、共同集合冻结、domain/mask等权、圆周距离、margin和JS。

## 4. 实验执行与分析

- [x] 4.1 等待并审计六个40-epoch训练完成；任一失败保留manifest/log并按契约处理，不选择best checkpoint替代last。
- [x] 4.2 对三方法seeds1-3运行0-80固定mask聚合评估及85/90/95极端稀疏评估，确认15-domain、mask和checkpoint provenance完整。
- [x] 4.3 为seeds1-3提取逐样本任务输出，生成CSV、PNG/PDF和中文Markdown说明，明确local-adaptation范围与不一致指标。
- [x] 4.4 分析三seed结果并给出T2相对AMBER/RMBP的supported/partial/unsupported结论，不修改模型或数据制造差距。

## 5. 验证收口

- [x] 5.1 运行`openspec validate validate-t2-vs-mmw-baselines-multiseed --strict`和`openspec validate --all --strict`。
- [x] 5.2 运行目标focused tests、`conda run -n kd_mm_beam python scripts/verify_compile.py`和必要的quick验证。
- [x] 5.3 复核`git status --short`，确认dataset、outputs、logs、cache和checkpoint均未纳入源码变更，并记录最终产物路径。
