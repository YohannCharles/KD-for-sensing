## Context

本 change 来自一次 repo-wide ponytail 审计。审计结果显示，当前仓库的主要成本不在训练主路径，而在剩余维护表面：退役整模型类仍可直接导入，旧 alias 和专用 guard 继续保活，`pyyaml` 之外还维护手写 YAML 子集解析器，若干 dataclass/recipe 文件只包装小常量表，dataset descriptor 与 `modalities.py` 重复表达 profile 信息，objective metadata 被拆成多份私有常量表，run index/cleanup 继续扩张历史输出考古分支。

这些改动横跨模型、registry、config、dataset runtime、objective metadata、diagnostics、CLI 和测试，所以需要先写清楚边界。目标是删除或合并低价值维护面，不改变当前训练、评估、预处理、诊断和本地产物保护契约。

当前约束：

- 所有项目相关 Python 验证使用 `conda run -n kd_mm_beam ...`。
- 不删除、不移动、不修改 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、`All_models/` 或历史权重。
- 不恢复旧 KD、HiST/Hist、Top8、residual、BGAM、viewer manifest、Raymobtime、CRAF/MARF/G2D、Multimodal-NF 入口。
- 当前 `kd-sensing-*` console scripts、canonical config、registry build、`modular_sequence`、BeamBench 专用 CLI、JEPA diagnostics 继续可用。

## Goals / Non-Goals

**Goals:**

- 为 ponytail 审计候选建立删除、合并、保留和后续处理边界。
- 删除退役整模型类和旧 alias 的直接导入/forward 保活路径，同时保留当前特征提取器。
- 将配置加载收敛到 `pyyaml`，删除手写 YAML fallback。
- 折叠只包装常量表的小 dataclass/recipe/objective/dataset descriptor 层。
- 收缩 legacy-only guard、run-index 和 cleanup 分支，保留会防止静默误跑的安全边界。
- 把测试改成验证当前行为和 owner module，而不是继续保活旧 import path。
- 用 focused tests 覆盖每个 wave，必要时再跑全量回归。

**Non-Goals:**

- 不改变模型数学结构、loss 公式、batch contract、dataset split、beam label、metric 口径、checkpoint schema 或默认输出分区。
- 不删除当前推荐 CLI、当前 canonical configs、当前 diagnostics workflow 或当前 BeamBench package scripts。
- 不新增 registry、配置数据库、治理 YAML、兼容 wrapper 或通用抽象来替代被删除的小层。
- 不把本 change 扩展成新实验、新论文结果 claim 或本地产物清理。

## Decisions

1. **按可回滚 wave 实施，而不是一次性大删。**  
   顺序为：基线和候选证据、退役模型类、config/parser、recipe/objective/dataset 小层、run-index/cleanup legacy 分支、facade/`__all__`/CLI wrapper、文档和验证。每个 wave 都有最小 focused tests。替代方案是按审计清单一次性删除；定位失败太慢，不采用。

2. **保留当前特征提取器，删除退役整模型类。**  
   `GpsFeatureExtractor`、`ImageFeatureExtractor`、`RadarFeatureExtractor`、`LidarFeatureExtractor`、`MmWaveFeatureExtractor` 仍被 `modular.py` 和 fusion owner 使用，不能删。`GpsModalityNet`、`ImageModalityNet`、`RadarModalityNet`、`LidarModalityNet`、`MmWaveModalityNet`、`FusionTeacherModalityNet`、`FusionStudentModalityNet` 及旧 alias 已退出 registry，只由测试保活，应删除。替代方案是保留直接实例化测试；它会继续把退役类当 current API，不采用。

3. **unknown-name 错误足够时，不维护专用 tombstone guard。**  
   只保留会防止静默误跑或高频迁移的 guard，例如 KD token、image profile/encoder、场景 dataset alias 和明确仍被 current config load 触发的拒绝项。完全退役路线交给 registry/config unknown-name、OpenSpec tombstone 和 inventory。替代方案是所有旧路线都专用错误；维护成本线性增长，不采用。

4. **`pyyaml` 是唯一 YAML parser。**  
   `pyyaml` 已是 runtime dependency，手写 `parse_simple_yaml` 既不完整，也制造第二套语义。删除 optional-yaml 分支后，`safe_load_yaml` 只是 `yaml.safe_load` 的薄包装或直接内联。替代方案是保留 fallback 给极简环境；项目环境契约已经要求依赖安装，不采用。

5. **常量表优先普通字典，dataclass 只在有行为时保留。**  
   `canonical_recipes`、objective metadata/history、dataset descriptor 中只包装静态表的 dataclass 可以内联或合并到 owner。需要校验、metadata 输出或错误信息的 helper 保留为函数。替代方案是继续把每张表拆成独立 capability-like 文件；没有收益，不采用。

6. **run-index 和 cleanup 保留安全，删除历史考古扩张。**  
   保留只读索引、状态分类、dry-run manifest、保护边界、显式确认和 legacy archive 基础分类。删除只服务特定退役研究线的细粒度输出规则，除非它们仍是 manifest 安全删除的必要条件。替代方案是把每个历史实验目录都写成 runtime rule；不是当前产品面，不采用。

7. **内部导入用 owner module，公开入口只留 CLI 和必要 package API。**  
   删除 3 行 CLI wrapper 时，`pyproject.toml` 直接指向真实 `main`。删除 `__all__` 时，不影响显式 import。若某 `__init__.py` 是公共 package API 且不引入重依赖，可保留最小入口；不能让它承载旧 alias 或大转发表。替代方案是保留所有 re-export 以免外部脚本改 import；这与收缩目标冲突，不采用。

8. **TinyViT 注册只去重，不改变行为。**  
   四个注册名保持不变，构建、metadata、权重加载和错误行为保持不变；仅把复制粘贴注册改成 preset 表循环。替代方案是新增 TinyViT registry/factory 抽象；四个 preset 不值得。

## Risks / Trade-offs

- **[Risk] 外部脚本直接导入退役整模型类或旧 alias。** → **Mitigation:** 标记 breaking change；文档指向 `modular_sequence`、当前 encoder 和具体 owner module；当前 CLI 不删除。
- **[Risk] 删除 guard 后错误信息不如旧专用文案。** → **Mitigation:** 保留高风险 guard；registry/config unknown-name 必须列可用名称；retired tombstone 和 inventory 保留历史说明。
- **[Risk] YAML parser 收缩暴露之前 fallback 容忍的非法 YAML。** → **Mitigation:** 以 `pyyaml` 为项目契约；运行 config load characterization 和涉及 YAML 的 focused tests。
- **[Risk] 合并常量表时漏掉 objective metric alias 或 tensorboard scalar。** → **Mitigation:** 保留 objective metadata focused tests，覆盖默认 metric、alias、history fields、available metrics 和 TensorBoard scalar。
- **[Risk] cleanup/run-index 分支收缩误删安全保护。** → **Mitigation:** 删除的是历史识别规则，不删除保护边界；runtime cleanup tests 必须覆盖 tracked file、dataset、cache、active run 和 confirm 参数。
- **[Risk] 大量测试更新掩盖行为回归。** → **Mitigation:** 测试删除只限退役类保活；当前 model/config/CLI/registry tests 必须继续覆盖 public behavior。

## Migration Plan

1. **Wave 0：基线和证据。**  
   记录 `git status --short`；用 CodeGraph/`rg` 确认候选调用方；运行 `openspec validate prune-remaining-overengineered-surface --strict`。

2. **Wave 1：退役整模型类收缩。**  
   删除退役整模型类、旧 alias 和只服务它们的 direct-forward tests；保留 feature extractor。更新 registry/component tests，确认 `modular_sequence` 和当前 configs 仍通过。

3. **Wave 2：配置 parser 和 guard 收缩。**  
   删除 `parse_simple_yaml`、optional-yaml dump/load 分支和低价值 retired-route guard。运行 config load characterization、component registry 和 architecture boundary tests。

4. **Wave 3：小层合并。**  
   合并 canonical recipe dataclass 层、objective metadata/history 常量拆分、dataset descriptor/profile 重复层。保持函数名或迁移调用方，避免新增兼容 wrapper。运行 objective metadata、config load、dataset profile focused tests。

5. **Wave 4：run-index/cleanup legacy 分支。**  
   收缩历史输出规则，保留 dry-run、安全删除、保护路径和 canonical layout。运行 run index 和 runtime artifact cleanup tests。

6. **Wave 5：facade、`__all__` 和 CLI wrapper。**  
   删除无价值 re-export、内部 `__all__` 镜像和 3 行 CLI wrapper；`pyproject.toml` 指向真实 owner。运行 CLI help、architecture boundary 和 import smoke。

7. **Wave 6：TinyViT 注册去重。**  
   用 preset 表循环注册四个 TinyViT encoder，保持 registry names 和 metadata。运行 TinyViT focused tests。

8. **Wave 7：文档和 inventory。**  
   更新 inventory、agent navigation、maintainer context index 和必要 README/docs，说明 breaking import changes、保留理由和后续候选。

9. **最终验证。**  
   至少运行：
   - `openspec validate prune-remaining-overengineered-surface --strict`
   - `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`
   - `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py tests/test_component_registry.py -q`
   - `conda run -n kd_mm_beam pytest tests/test_runtime_artifact_cleanup.py tests/test_run_index.py -q`
   - `conda run -n kd_mm_beam pytest tests/test_tinyvit_image_encoder.py tests/test_prediction_objectives.py -q`

回滚策略：每个 wave 独立提交或保持独立 diff。若某 wave 失败，只恢复该 wave 文件，不恢复已经验证通过的前序瘦身。

## Implementation Resolutions

- `dataset_descriptors.py` 保留轻量 descriptor 查询和错误信息；profile 名称、sample key、fusion input key、shape/metadata 改为复用 `modalities.py`，不再维护第二套 profile 镜像。
- `runtime_artifact_cleanup.py` 删除只服务 HiST/P3/V8/V9 旧目录名的细粒度规则；保留 dry-run manifest、保护边界、显式确认、路径重验证、checkpoint retention 和 organize 的 legacy root/numeric/eval/registry 基础分类。
- `models/fusion/__init__.py` 保留最小 current fusion owner imports，用于默认组件注册；内部源码和测试不再从 package facade 或已删 `fusion.networks` 导入实现符号。
