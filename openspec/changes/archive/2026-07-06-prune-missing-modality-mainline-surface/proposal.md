## Why

项目主线已经转向多模态缺失模态鲁棒性波束预测，当前核心证据集中在 U-MaskBeamJEPA、Scene31-34 local/manual 复现实验、AMBER/RMBP-MM/TII-VLRG 风格对照与相关缺失模态评估上。与此同时，仓库仍保留大量历史 one-shot 脚本、重复 encoder ablation 生成器、局部 polish/presentation 帮手、Scene31 旧 overlay 配置、RBMA/KD/BTAPA 等分支配置以及未登记的大文件热点。

这些表面不一定都是废代码，但已经造成三个直接问题：

- 新协作者很难判断哪些入口是当前主线，哪些只是历史证据或本地手工产物。
- `project_surface_doctor` 已能发现未分类脚本、未分类配置和未登记大 owner；继续堆叠会让治理表失去信号。
- PatchViT/TinyViT 等局部 ablation 出现重复生成器趋势，如果继续加 runner，会扩大维护面，而不是提高主线能力。

本 change 的目标是先把“能删、能并、必须保留”的边界定清楚，再按最小波次清理代码表面：删除或合并确认非主线的入口，保留并标注仍支撑论文证据链的入口，避免为历史兼容新增 wrapper。

## What Changes

- 建立“缺失模态主线代码表面清理”波次：
  - Wave 0：冻结并复核候选清单，包括 `project_surface_doctor` 的未分类 scripts/configs/hotspots、Scene31/Scene31-34 本地脚本、historical table/presentation/polish helper、RBMA/KD/BTAPA 配置族。
  - Wave 1：合并 TinyViT/PatchViT Scene31-34 encoder ablation 生成器，禁止新增单独 PatchViT runner；runner 应按 encoder family 或 manifest 驱动。
  - Wave 2：处理 `export_scene31_34_presentation_artifacts.py`、`run_final_scene31_34_polish.sh`、historical table/conclusion helper：若仍支撑当前论文交付，则登记为 local/manual analysis helper；否则删除对应脚本、测试和文档引用。
  - Wave 3：缩小 Scene31 与 RBMA/KD/BTAPA 配置表面，只保留当前主线、复现实验或明确 claim 证据需要的 YAML；可由生成器/manifest 覆盖的重复 YAML 应移除或改为生成产物。
  - Wave 4：登记或拆分大 owner 热点，尤其是 `gps_query_evidence.py`、`run_metadata.py`、`u_mask_beam_jepa.py`；核心模型与诊断能力默认保留，不用“删文件”代替边界治理。
- 明确保留边界：
  - 保留 U-MaskBeamJEPA 模型、loss、run metadata、migration guards、retired route 测试和当前 Scene31-34 主线 runner/summary/export 路径。
  - 保留 Image+GPS JEPA、MMW/CSI/physics-informed 路线作为已登记 secondary/supporting surface，除非另开 change 明确 de-scope。
  - 保留用户输入、输出落盘、数据 split、label-space 与 retired-route migration guard；只删除重复、历史、未引用或可再生成的入口。
- 文档和治理同步：
  - 更新 `docs/project_surface_inventory.md`、`docs/mainline_model_catalog.md`、相关 scoped context 与 OpenSpec spec delta，确保“主线/secondary/local/manual/retired”一致。
  - 更新或新增最小测试，使 doctor 能防止重复 generator/runner 回流，并要求未分类脚本、配置和 hotspot 有明确处置。

## Capabilities

### New Capabilities

- 无。本 change 是收缩和治理，不引入新的训练能力、模型能力或外部依赖。

### Modified Capabilities

- `project-surface-cleanup`：增加缺失模态主线清理波次、删除/合并证据要求、配置族缩小规则，以及“不要删除核心证据链”的保护边界。
- `project-entrypoint-lifecycle`：要求 Scene31/Scene31-34 local/manual 入口可收敛；encoder ablation 不得按 encoder 复制脚本；final polish、presentation 和 historical table helper 必须删除或显式登记。
- `project-hotspot-governance`：要求未登记大 owner 在清理波次中被登记、拆分或接受；禁止用新增重复脚本扩大热点。

## Impact

- 影响范围：`scripts/` 下 Scene31/Scene31-34 本地脚本、`configs/scene31/` 与 `configs/fusion/experiments/` 的部分实验 YAML、项目表面治理文档、OpenSpec specs、doctor/architecture/ablation 相关测试。
- 不影响范围：`dataset/`、`outputs/`、checkpoint、训练产物、容器或系统启动配置；本 change 不迁移本地实验结果。
- 兼容性：会删除或停止推荐部分 local/manual 历史入口；这是有意的破坏性收缩，不提供旧入口 wrapper。仍需可通过当前主线 CLI、配置加载和保留的 Scene31-34 工作流完成复现实验。
- 验证：至少运行对应 OpenSpec strict 校验、`project_surface_doctor`、架构边界测试、CLI/config 快速测试、encoder ablation 相关测试和 compile 验证。
