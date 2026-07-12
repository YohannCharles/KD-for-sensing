## Context

`kd_sensing.config.normalization` 只需要时序缺失模式和聚合方式的字符串规范化，却从 `kd_sensing.data.temporal_missing` 导入这两个 helper。后者在模块顶层导入 `torch` 并同时承载 tensor 聚合、mask 采样、batch 变换和固定评估 mask cache，导致冷进程导入 `kd_sensing.config` 也加载完整 PyTorch runtime。当前实测约为 1.55 秒和 616-620 MiB 峰值内存，且 `torch` 明确出现在 `sys.modules`；这与现行 `project-architecture` 的配置轻量导入契约冲突。

实施探针还发现 configuration validation 导入 `difficulty.schema` 时，会经顶层 preset import 加载依赖 torch 的 missing-pattern runtime。该链不参与普通 schema 解析；preset 只在规范化显式 `missing_modality_stress` profile 时需要。

本 change 必须保持现有时序缺失配置、运行函数、训练行为和 artifact schema 不变，并避开当前在途的 MMW/H5P1/LMDB 实现文件。项目静态护栏还禁止在 runtime source 中新增 `from __future__ import annotations`，因此不能依赖 postponed annotations 简化 torch 的延迟导入。

## Goals / Non-Goals

**Goals:**

- 让 `import kd_sensing.config`、配置 normalization 和普通 configuration validation 保持纯配置依赖，不导入 `torch` 或训练 runtime。
- 为时序缺失模式与聚合方式保留单一规范化来源，避免 config 与 runtime 维护两套允许值和错误语义。
- 保持 `kd_sensing.data.temporal_missing` 现有运行函数、常量可见性和 tensor 行为兼容。
- 用 fresh-process 回归检查长期保护导入边界，同时避免易受硬件和环境噪声影响的性能阈值。

**Non-Goals:**

- 不改变时序 mask 采样、聚合数学语义、batch 字段、固定 mask cache 或 H5/P1 workflow。
- 不开展全包 lazy import 重构，也不拆分其它 temporal runtime helper。
- 不处理 CLI 未知参数静默丢弃、专用 CLI flag 后置校验或 runtime metadata 重复采样；这些是独立后续候选。
- 不新增配置字段、console script、第三方依赖、兼容 facade 或本地产物迁移。

## Decisions

### Decision 1: 提取最小纯时序配置契约

新增 `kd_sensing.data.temporal_missing_contract` 或等价的窄纯标准库 owner，只承载 `TEMPORAL_MISSING_MODES`、`TEMPORAL_AGGREGATION_MODES`、`normalize_temporal_missing_mode()` 和 `normalize_temporal_aggregation()`。`config.normalization` 直接依赖该 contract；`data.temporal_missing` 也从同一 contract 导入并继续在现有模块路径暴露这些名称，因此 tensor runtime 依赖纯契约，而纯契约不反向依赖 runtime。

该边界只提取 config 与 runtime 确实共同消费的四个符号。`TemporalMissingConfig`、默认模态、分层缺失类型、tensor 聚合、mask 采样和 cache IO 仍留在 `data.temporal_missing`，避免把一个根因扩展成整模块拆分。

备选方案是在 `data.temporal_missing` 中删除顶层 torch import，并在每个 tensor 函数内延迟导入。该方案需要改写多处运行时类型标注或引入项目静态护栏禁止的 postponed annotations，修改面更大。另一备选是让 config 复制两个 normalizer；它会产生两套允许值与错误信息，拒绝采用。

### Decision 2: 保持现有 runtime import 与行为兼容

`data.temporal_missing` 继续使用并暴露原有规范化函数和 mode 常量，现有 difficulty operator 与其它 runtime caller 不需要迁移到新的 contract 路径。新增 contract 是实际依赖方向的 owner，不提供旧路径转发层、registry 或包级 barrel export。

测试复用现有 temporal/config characterization 覆盖有效值、非法值和 tensor 行为，不引入新的 benchmark framework。

### Decision 3: 用 fresh-process 模块探针保护边界

在架构边界测试中启动独立 Python 子进程，导入 `kd_sensing.config` 后检查 `sys.modules`。探针至少拒绝 `torch`、模型实现、dataset runtime、诊断渲染和训练主循环，并验证 config 导入成功。独立进程可避免 pytest 自身已导入 torch 造成误判。

自动化测试不固定耗时或 RSS 上限，因为这些数值受 PyTorch 版本、动态链接器、机器负载和平台影响。实施后可用同一冷进程命令记录前后观测值，但规范验收以依赖边界为准。

### Decision 4: validation schema 按需加载 preset owner

`difficulty.schema` 仅在规范化单个 profile 时导入 missing-modality preset helper，不在模块导入阶段加载 preset 或 missing-pattern tensor runtime。difficulty operator 的 runtime 注册继续由 pipeline 在执行 batch 变换前完成；不复制 operator allowlist，不改变未知 operator 的现有拒绝语义。

## Risks / Trade-offs

- [Risk] 移动规范化符号可能破坏从 `data.temporal_missing` 导入的现有 caller。 -> 在原 runtime 模块直接导入并继续暴露这些符号，运行现有 temporal 与 difficulty focused tests。
- [Risk] 新 contract 反向导入 config/runtime 会形成新的循环或重新引入重依赖。 -> contract 只允许标准库和纯值逻辑，fresh-process 探针直接验证最终依赖图。
- [Risk] 架构测试若断言所有 `kd_sensing.data`/`engine` 包均未出现，会误伤合法的轻量 package marker 与 objective metadata。 -> 只拒绝 tensor/dataset 实现、模型、训练和渲染 owner，不拒绝轻量 marker 或纯 metadata 模块。
- [Risk] difficulty preset 延迟导入可能改变 profile 展开。 -> 只移动 import 时机，并运行现有 modality difficulty focused tests。
- [Trade-off] 为四个共享符号增加一个小文件。 -> 该文件承载真实的 config/runtime 依赖方向，换取单一来源和可测试边界；不继续扩展为通用 schema 框架。

## Migration Plan

1. 添加纯时序配置 contract，并把两个 mode 常量和两个 normalizer 迁入该 owner。
2. 更新 config 与 temporal runtime import，保持原 runtime 模块的符号和 `__all__` 行为。
3. 将 configuration validation 的 difficulty preset owner 改为按需导入，保持 profile 与 operator 语义。
4. 增加 fresh-process 架构探针，并运行 config/temporal focused tests 确认语义兼容。
5. 运行 change strict validation、相关快速验证和全量 OpenSpec strict validation。

该变更不涉及数据或 artifact migration。若出现回归，可回退 contract/import 调整和对应探针；配置文件、checkpoint 与本地产物无需处理。

## Open Questions

- 无。
