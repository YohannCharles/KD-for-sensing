## Context

本仓库当前已经有 DeepSense6G 场景选择、Vision-Position baseline suite、BeamBench/Arnold22 Image AE + GPS Direct、本地结果账本、Scenario D image observability benchmark 和 GPS+LiDAR BGAM workflow。用户本次要求复现 IEEE Xplore article `11282996` 对应的 AMR-Net-gps-image，但只使用 GPS 和 image 模态，不使用 LiDAR。

当前网络环境无法直接读取 IEEE Xplore 页面全文，但公开 metadata 足以暴露一个关键冲突：Crossref/IEEE metadata 将 document `11282996` 指向 DOI `10.1109/JIOT.2025.3641184`、IEEE Internet of Things Journal 2026 的 AMR-Net 多模态 beam prediction 论文；公开作者页面和代码包可核对到的 DeepSense6G drone beam prediction 路线则是 GLOBECOM 2022 `Towards Real-World 6G Drone Communication: Position and Camera Aided Beam Prediction`，DOI `10.1109/GLOBECOM48099.2022.10000718`，IEEE document `10000718`，Scenario 23、Top-1/Top-3/Top-5 beam accuracy、作者代码分为 `image_beam` 和 `pos_beam`。因此设计必须把 source audit 作为前置步骤：实现期先记录 IEEE title/DOI/PDF、官方或作者代码 commit、数据场景、split、模型、指标和 target 表格；metadata conflict 或缺少官方协议时只能跑 mock/local substitute，不得声明 official reproduction。

本设计遵守项目边界：源码落在 `src/kd_sensing` 包内；所有项目 Python 命令用 `conda run -n kd_mm_beam ...`；真实数据、checkpoint、cache、predictions、plots 和 reports 写入 ignored 的 `outputs/` 或 `logs/`；不恢复旧 KD/HiST/residual/Top8，不新增绕过包结构的 root-level 长期入口。

## Goals / Non-Goals

**Goals:**

- 为 AMR-Net-gps-image 建立可审计的 GPS+Image-only 复现 workflow。
- 固化 source audit schema，记录 IEEE/作者来源、论文协议、官方代码、数据场景、split、模型组、metric profile 和 claim status。
- 支持论文所需 DeepSense6G 场景描述符，预计为 Scenario 23；实现必须由 source audit 确认后启用。
- 提供 paper-specific config/manifest，严格启用 `image` 与 `gps`，并拒绝 LiDAR、radar、mmWave、CSI 或 all-modalities fallback。
- 复用 Vision-Position baseline suite 和共享 batch/runtime；模型行优先使用 modular components 或 paper workflow helper，不复制通用训练循环。
- 输出 Top-k beam accuracy、可用 DBA/beam-distance、overhead reduction 或 paper 等价指标，并维护 local/official claim caveat。

**Non-Goals:**

- 不在本 change 中运行真实长训练或提交真实 benchmark 结果。
- 不使用 LiDAR，不读取 LiDAR BEV/cache，不复用 GPS+LiDAR BGAM 或 all-modalities config 作为替代。
- 不把 local substitute 数值写成官方 IEEE 复现结果。
- 不改写真实 `dataset/`、官方代码、split CSV、beam labels 或已有 checkpoint。
- 不新增旧式 root-level train script；如需要命令入口，使用包内 CLI 或当前 allowlist 的薄 alias。

## Decisions

1. **Source audit 是 workflow 的第一步。**

   新增 `source_audit` 结构，记录 `ieee_article_number=11282996`、IEEE URL、title、DOI、publication venue/year、PDF/source availability、作者代码 URL/commit、dataset/scenario、modalities、target label、split、metrics 和阻塞项。IEEE 页面不可访问、PDF 未提供，或 article metadata 与 Scenario 23 作者包不一致时，runner 仍可生成 blocked/local report，但不能把结果标为 official。备选方案是直接按公开作者页面实现，但这会把未核对的 IEEE article 细节写成事实。

2. **DeepSense6G Scenario 23 通过场景描述符接入。**

   公开作者页面和代码包指向 Scenario 23；实现期新增 scene descriptor，包含 `scene_id: 23`、`scene_slug: scene23`、默认路径 `dataset/DeepSense6G/scenario23`、legacy path、train/test CSV 名和输出分区。若 source audit 证明 article `11282996` 不是 Scenario 23，必须先更新 OpenSpec artifact 再改代码。备选方案是在 runner 中硬编码 data_root，但这会绕过现有 scene-selection 和 runtime metadata。

3. **Paper-specific workflow 复用 Vision-Position baseline suite。**

   普通 Image/GPS 模型组通过 `modular_sequence` 配置表达：image encoder 可选 CNN/ResNet/Camera AE，GPS encoder 使用 paper-aligned direct/MLP features，fusion 使用 late concat 或 paper-audited FCN。若作者代码需要特殊两阶段训练或报告产物，则 helper 位于 `src/kd_sensing/baselines/ieee11282996/`，但训练、评估、metrics 和 output layout 仍复用共享 runtime。备选方案是 vendoring 作者代码为独立脚本，这不符合当前包结构和扩展契约。

4. **LiDAR 禁止由配置和 runner 双重校验。**

   Paper preset 必须显式声明 `modalities: [image, gps]`，dataset flag 只能启用 `use_gps` 和 image 输入。配置加载、runner audit 和 tests 都要拒绝 `lidar`、`use_lidar: true`、LiDAR CSV/cache 输入或 GPS+LiDAR BGAM checkpoint。这样不会因为现有 CSV 名包含 `LIDAR` 或仓库已有 BGAM 能力而误读 LiDAR。备选方案是只在 README 中提醒不用 LiDAR，但不可测试。

5. **指标分为 paper-aligned 和 local diagnostics 两层。**

   Paper-aligned 层至少包含 Top-1、Top-3、Top-5 beam accuracy，并在 source audit 记录 paper 使用的 beam count、index base、target source 和 beam-training overhead 口径。Local diagnostics 可额外输出 DBA、linear/circular beam distance、confusion matrix 或 Scenario D robustness sweep，但必须标记为 local control，不得混入 paper target table。备选方案是统一复用 BeamBench DBA 作为主指标，但 IEEE drone paper公开摘要强调 Top-k accuracy，DBA 可能不是原文主口径。

6. **Claim status 写入报告和结果账本。**

   结果状态至少包括 `blocked_official`、`paper_protocol_audited`、`local_substitute`、`local_control`、`mock_smoke` 和 `official_reproduction`。只有 source audit 完整、数据/split/权重/官方训练或评估协议满足时才允许 `official_reproduction`。输出 report 必须记录命令、git status 摘要、manifest digest、scenario、enabled modalities、checkpoint provenance 和 warnings。

## Risks / Trade-offs

- IEEE 页面当前不可访问且公开 metadata 指向非 Scenario 23 作者包 → source audit 支持 metadata conflict / blocked 状态，要求用户提供 PDF、BibTeX、官方代码或可访问 metadata 后再宣称 official。
- Scenario 23 数据可能不在本地 → 提供 dataset check 和 mock smoke，真实训练 blocked 时报告原因，不生成伪指标。
- 作者代码的模型细节可能与本仓库组件不完全一致 → 先实现 paper-equivalent config；若需要特殊流程，放入 paper workflow helper 并记录差异。
- CSV 名可能包含 `LIDAR` 但复现不能使用 LiDAR → 用启用模态和实际 batch keys 判断输入，不以 CSV 文件名推断 LiDAR 使用。
- Local substitute 与 official result 容易混淆 → 强制 claim status、metric profile 和 source audit digest 进入 report/metadata。

## Migration Plan

1. 实现 source audit schema、Scenario 23 descriptor 和 dataset check/mock smoke，不运行真实训练。
2. 增加 paper-specific Image/GPS-only config/manifest 与 LiDAR 禁用校验。
3. 接入 Vision-Position model groups、metrics aggregation 和 report writer。
4. 增加包内 CLI 或薄 alias，并同步 README、inventory、result claims 和架构边界测试。
5. 运行 `openspec validate reproduce-ieee-11282996-gps-image --strict`、相关 focused pytest 和 CLI help smoke。

Rollback 策略：禁用或删除 paper-specific config/CLI 即可回退；新增 scene descriptor 不改变默认 DeepSense6G scene31 行为。若 source audit 证明 article `11282996` 与 Scenario 23 不一致，应先更新本 change 的 proposal/design/spec/tasks，再继续实现。

## Open Questions

- IEEE `11282996` 的准确 title、DOI、venue/year 和 PDF 是否可由用户提供或由机构网络访问？
- 论文是否只报告 image-only 与 GPS-only，还是包含 Image+GPS fusion 行？
- 原文 GPS 特征是经纬度 direct、distance/angle、relative-polar，还是作者代码中的其它转换？
- 论文使用的 beam 数、beam index base、split 名和 train/test ratio 是否与公开作者代码完全一致？
