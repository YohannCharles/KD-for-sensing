## Context

MMW Town10 prepared pipeline 当前从 channel `_paths` 派生 64 维 beam power vector，并用 `argmax` 作为 0-based raw beam label。dataset 再把 `beam*` 和 `future_beam*` 路径解析为 `input_beam` 与 `target_beam`，soft target、beamspace physical label、HiST-Beam coarse/fine label、DBA 和分布诊断都默认这些 class ID 已经具有一致的环形拓扑。

GPS-angle 诊断显示 raw beam label 与几何角度之间存在 scene 级方向和 offset 差异，0/63 边界在部分 town 上会把不应相邻的区域错误视作相邻。这个问题不是某个模态输入缺陷，而是 MMW raw label space 与训练 label space 没有显式校准。

## Goals / Non-Goals

**Goals:**

- 为 MMW dataset 和诊断流程提供可配置 beam label calibration，默认关闭。
- 让 `input_beam`、`target_beam`、soft beam distribution、beamspace physical label、prediction artifact 和分布诊断共享同一 calibrated class order。
- 在 metadata 中同时保留 raw label、calibrated label、mapping 参数和 inverse mapping 信息，便于旧结果审计。
- 保持 GPS、image、LiDAR、radar、CSI 和 `mmwave` sensing feature 的读取与张量 shape 不变。

**Non-Goals:**

- 不重新定义 beam power vector 的物理计算方式，不改 codebook 或 channel projection 算法。
- 不自动用 target_test label 拟合 calibration 参数。
- 不要求迁移或重写已有 raw-label checkpoint；启用 calibration 的新 run 与旧 run 通过 metadata 区分。
- 不引入外部依赖或全局改变 DeepSense6G/Raymobtime label 语义。

## Decisions

1. **新增中心化 label mapping helper**

   使用小型 helper 表达 raw→calibrated 与 calibrated→raw 的可逆映射，第一版支持 affine circular mapping：

   ```text
   calibrated = (direction * raw + offset) mod num_classes
   direction in {1, -1}
   ```

   同时支持显式 permutation list 作为未来扩展，但默认推荐 affine。这样可以覆盖当前 GPS-angle 诊断观察到的 scene 级 offset 和方向问题，并保持 class 数不变。

   备选方案是直接重写 split CSV 中的 `future_beam_label*`。该方式会丢失 raw label provenance，也容易让旧 artifact 和新训练混淆，因此不作为主路径。

2. **dataset 返回 calibrated hard label，metadata 保留 raw label**

   MMW dataset 在启用 calibration 时对 `input_beam` 和 `target_beam` 做映射；缓存 key 仍以 beam path 为主，但缓存值需要记录 raw 与 calibrated 两种 label 或至少记录当前 mapping fingerprint。样本 metadata 记录 `beam_label_space`、`beam_label_mapping`、`raw_input_beam`、`raw_target_beam` 或等价字段。

   这样训练循环仍消费现有 `target_beam` 字段，不需要修改所有 loss 和模型接口。

3. **class-indexed distribution 按同一映射重排**

   对 `target_beam_distribution`、`beamspace_power_label`、prediction probability 和 histogram，使用 `dist_calibrated[mapping(raw)] = dist_raw[raw]`。作为 sensing input 的 `mmwave` 历史 power vector保持 raw 顺序，因为它表达原始接收功率特征；只有当该 64 维向量被声明为 label distribution、physical label 或 metric reference 时才重排。

   备选方案是把 `mmwave` 输入也重排。该做法会改变模型输入分布，破坏已训练 mmWave encoder 的物理特征语义，且不是解决 label topology 的必要条件，因此排除。

4. **配置和 metadata 分层**

   配置入口放在 `data.dataset.beam_label_calibration`，仅 `data.dataset.type: mmw` 时生效。支持全局默认和 scene overrides：

   ```yaml
   data:
     dataset:
       beam_label_calibration:
         enabled: true
         label_space: calibrated_gps_angle
         num_classes: 64
         direction: 1
         offset: 32
         scene_overrides:
           Town10_curvyroad_seed42:
             direction: -1
             offset: 52
   ```

   run metadata、prediction CSV/JSON 和分布诊断输出 MUST 写入 mapping fingerprint，避免 raw-label 与 calibrated-label 结果被误合并。

5. **calibration 参数来源必须可审计**

   calibration 参数可以来自手工配置、source/support split 拟合或离线诊断 artifact，但 metadata 必须记录来源和拟合 split。target_test 不得用于训练期 calibration 拟合；如果离线诊断为了审计读了 target_test label，输出必须标记为 diagnostics-only。

## Risks / Trade-offs

- **旧 checkpoint 指标不可直接比较** → run metadata 明确记录 `beam_label_space`，分析脚本按 label space 分组；旧结果视为 raw label baseline。
- **soft label 与 hard label class order 不一致** → soft target 生成和读取分布时统一经过 mapping helper，并添加测试覆盖 0/63 边界。
- **beamspace physical label cache 复用错误** → cache metadata 纳入 mapping fingerprint；mapping 不匹配时重建或拒绝复用。
- **scene-specific calibration 误用 target_test** → calibration fitting API 和 metadata 记录 split source，训练/适应路径禁止使用 target_test 拟合结果作为监督校准。
- **coarse/fine group 语义改变** → 这是 calibration 的预期效果；HiST-Beam metadata 记录 calibrated grouping，并在测试中验证 `label // group_size` 基于 calibrated label。

## Migration Plan

1. 默认关闭 calibration，保证现有配置、测试和旧 artifact 继续使用 raw label space。
2. 新增 mapping helper 与单元测试，覆盖 affine、inverse、distribution reorder、fingerprint。
3. 在 MMW dataset 接入 hard label 映射和 metadata，不改变非 MMW dataset。
4. 接入 soft target、beamspace physical label cache、prediction export 和分布诊断。
5. 增加一个显式 MMW calibrated 配置或 CLI override，用于重跑 GPS/GPS+mmWave 和 HiST-Beam 对比。
6. 若需要回滚，关闭 `beam_label_calibration.enabled` 即恢复 raw label space；已生成 calibrated run 通过 metadata 与 raw run 区分。

## Open Questions

- 第一版是否只使用手工/诊断产物给出的 scene mapping，还是同时提供从 source/support GPS-angle 自动拟合 mapping 的 CLI？
- calibrated label space 的默认名称是否固定为 `calibrated_gps_angle_v1`，还是允许用户在配置中命名以表达不同校准版本？
- 对纯 mmWave 模型，是否需要提供可选实验路径把 `mmwave` 输入也按 mapping 重排作为 ablation，而不是默认行为？
