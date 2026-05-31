## Context

项目已有 `project-architecture` 对模块边界、轻量导入、入口生命周期和热点拆分的约束，但当前源码表面积仍有几个明显热点：

- `src/kd_sensing/engine/hist_beam_loso_execution.py` 约 2500 行，虽然已经存在 `hist_beam_loso_preflight.py`、`hist_beam_loso_stages.py`、`hist_beam_loso_summary.py` 和 `hist_beam_loso_matrix.py`，但 executor facade 仍承载大量 run record、stage metadata、summary/conclusion、CSV/JSON 写出和配置派生细节。
- `src/kd_sensing/data/mmw/preparation.py` 约 2168 行，混合了配置 schema、zip audit、sensor/channel indexing、sequence split、beam power 派生、manifest 写出、report 和 proxy geometry。
- 第二梯队热点包括 `models/fusion/hist_beam.py`、`diagnostics/run_index.py`、`tools/visualization/gradio_multimodal_viewer.py`、`data/transform_ops/csi.py`、`engine/batch.py` 和 `engine/evaluation_pass.py`，它们后续改动频繁，适合纳入 inventory 和架构测试防回流。

本变更是架构优化方案，不改变训练、评估、预处理和诊断的用户可见语义。所有 Python 验证命令必须使用 `conda run -n kd_mm_beam ...`，并且不得处理 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 等本地产物。

## Goals / Non-Goals

**Goals:**

- 将 `hist_beam_loso_execution.py` 的主实现拆到已有或新增窄模块，使该文件只保留公开入口、常量、少量兼容导出和顶层编排。
- 将 `data/mmw/preparation.py` 拆成 configuration、audit、indexing、splits、beam power、writers、reports、geometry/proxy features 等职责模块，保持现有公开函数 import 兼容。
- 更新 `docs/project_surface_inventory.md` 和 `tests/test_architecture_boundaries.py`，记录热点 facade、推荐窄模块和禁止内部回流路径。
- 为第二梯队热点建立分阶段拆分清单和轻量防护，避免继续扩张成新的 2000 行聚合文件。
- 保持公开 CLI、console script、manifest schema、run metadata、summary CSV/JSON、数据样本契约和默认路径策略兼容。

**Non-Goals:**

- 不重写 HiST-Beam、MMW 数据准备或 CSI 算法，不改变模型数值、loss、metric、split 或 label 语义。
- 不引入新的外部依赖，不新增旧入口、兼容聚合层或根目录运行方式。
- 不清理、移动、压缩或重写本地数据、输出、日志、缓存或 checkpoint。
- 不要求一次性拆完所有 800 行以上文件；第二梯队热点只在本 change 中建立 inventory、边界和必要的轻量抽取任务。

## Decisions

1. **按公开 facade + 窄模块实现拆分，而不是直接改名删除大文件。**

   `hist_beam_loso_execution.py` 和 `data/mmw/preparation.py` 已被测试、脚本和用户 import 依赖。首轮应保留这些公开路径，迁移主要实现到同包窄模块，并让 facade 只做参数编排和 re-export。这样可以控制破坏面，同时满足“新内部代码优先依赖窄模块”的架构要求。

   备选方案是直接删除或重命名大文件，但会造成不必要的 breaking change，也会增加用户配置、脚本和测试的迁移成本。

2. **优先拆最高风险的两个 2000 行级源码文件。**

   第一批必须处理 `hist_beam_loso_execution.py` 和 `data/mmw/preparation.py`。它们行数最高、职责最多，并且已经影响跨场景训练和 MMW 数据准备维护效率。完成后，`hist_beam_loso_execution.py` 目标收敛到薄 facade 级别；`preparation.py` 可保留公开入口但不再包含大段 private helper 实现。

   备选方案是按目录平均拆分多个文件，但容易造成大量低收益改动，且无法优先解决 2500 行热点。

3. **将拆分边界映射到自然职责，而不是按行数机械切割。**

   HiST-Beam executor 拆分建议：

   - `hist_beam_loso_preflight.py`: 数据可用性、prepared artifact、CSV/radar map preflight。
   - `hist_beam_loso_stages.py`: stage run loop、stage callbacks、DataLoader/stage 生命周期。
   - `hist_beam_loso_records.py`: run id、run dir、base/missing run record、stage record、progress event。
   - `hist_beam_loso_artifacts.py`: run metadata、summary CSV/JSON、JSONL progress 写出。
   - `hist_beam_loso_summary.py`: quick validation conclusion、eligibility、comparison helper。
   - `hist_beam_loso_matrix.py`: matrix profile、matrix summary、claim scope 元数据。
   - `hist_beam_loso_config.py`: scene cfg、stage cfg、enabled modalities、prototype/reuse/source key 决策。

   MMW preparation 拆分建议：

   - `preparation_config.py`: `MMWPreparationConfig`、配置加载、override normalization。
   - `preparation_audit.py`: zip/input audit、hash、extract marker、availability。
   - `preparation_index.py`: sensor frame、channel file indexing、scenario root/path parsing。
   - `preparation_splits.py`: sequence row 构造、group-safe split、leakage diagnostics。
   - `preparation_beam_power.py`: channel payload 读取、DFT/codebook beam power 派生、power validation。
   - `preparation_writers.py`: manifest CSV、rows、metadata、report 和 artifact path 写出。
   - `preparation_geometry.py`: relative geometry、proxy features、pose/path helper。
   - `preparation.py`: 保留 `prepare_town10_skybridge` 等公开 orchestration。

   备选方案是按“上半段/下半段”拆文件，但会保留跨职责耦合，后续仍难以定位改动。

4. **架构测试使用 inventory + facade 行数/禁止片段双重防护。**

   现有 `test_hotspot_facades_delegate_to_narrow_responsibility_modules` 已覆盖部分 facade，但 `hist_beam_loso_execution.py` 的 `max_lines` 仍允许 2500 行。实现时应降低第一批 facade 的阈值，并补充 `data/mmw/preparation.py` 的 helper 断言。对于第二梯队热点，先记录 inventory 和禁止新增内部 facade 依赖，避免在未拆完前引入脆弱阈值。

   备选方案是只依赖人工 review，但热点文件回流通常是渐进发生，测试更适合防止回归。

5. **迁移顺序以 characterization tests 锁定行为。**

   每个热点拆分前先确认现有 focused tests 覆盖关键公开字段；不足时先补小型 characterization tests。迁移后再运行对应 focused tests、架构边界测试、CLI smoke 和 OpenSpec strict validation。对数值语义敏感的逻辑只做搬迁，不做同时重写。

   备选方案是边拆边重构算法，但会把结构风险和业务风险叠加，定位回归更困难。

## Risks / Trade-offs

- [Risk] 机械搬迁 private helper 可能引入循环 import。→ Mitigation：优先把 dataclass/constants/schema 放入低层模块，重依赖逻辑保持函数内导入或由 orchestration 注入。
- [Risk] facade re-export 增加短期重复 import 面。→ Mitigation：inventory 标记哪些 facade 只服务公开兼容，内部代码和测试优先依赖窄模块。
- [Risk] 行数阈值过激导致一次实现范围失控。→ Mitigation：第一批只对两个 2000 行级文件设置硬性收敛目标，第二梯队先做 inventory 和局部抽取。
- [Risk] 拆分过程中改变 summary、manifest 或 run metadata 字段顺序/默认值。→ Mitigation：用 focused characterization tests 比较关键字段和路径，不以重排输出作为目标。
- [Risk] 当前工作区已有未提交实现变更。→ Mitigation：实施时先读取相关文件最新状态，只做增量迁移，不回退用户或其他变更。

## Migration Plan

1. 记录当前热点行数、公开入口和 helper 分布，更新 OpenSpec 与 inventory。
2. 为 `hist_beam_loso_execution.py` 补齐 characterization tests，锁定 run metadata、summary JSON/CSV、progress event、preflight error 和 quick validation conclusion 关键字段。
3. 将 executor 的 record/artifact/config helper 迁出，降低 facade 行数，并把内部调用改为窄模块。
4. 为 `data/mmw/preparation.py` 补齐 MMW preparation focused tests，锁定 manifest、split metadata、beam power、availability/report 关键字段。
5. 将 MMW preparation 的 config/audit/index/split/beam/writer/geometry helper 迁出，保留公开 orchestration 和 import 兼容。
6. 更新架构边界测试与 inventory，加入禁止回流路径和推荐窄模块。
7. 运行 OpenSpec 校验、focused tests、架构边界测试、CLI smoke；最终视改动范围运行全量 `conda run -n kd_mm_beam pytest -q`。

## Open Questions

- `hist_beam_loso_execution.py` 的最终 facade 阈值建议定为 700-900 行；实施时可根据公开常量和 re-export 数量确定精确值。
- `data/mmw/preparation.py` 是否需要长期保留所有现有公开 helper re-export，需要在实施时根据测试和外部引用扫描确认。
- 第二梯队热点是否在同一 implementation pass 做轻量抽取，还是只建立防护后另开 change，取决于第一批拆分后的测试成本和工作区已有变更状态。
