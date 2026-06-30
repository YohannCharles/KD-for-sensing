# U-MaskBeamJEPA Current Scope

当前实现覆盖 U-MaskBeamJEPA 的核心训练机制：full-modal teacher、missing-mask student、global JEPA NLL、modality NLL 和 uncertainty-aware fusion。正式 Scenario 32 配置在 `configs/fusion/u_mask_beam_jepa_s32.yaml`，快速 smoke 配置仍保留 Scenario 31。

当前预处理不是严格 JEPA-MSAC 复现口径：

- image: RGB/ImageNet resize + normalize。
- radar: RA/DA map。
- LiDAR: BEV representation，不是 depth projection。
- GPS: relative polar representation，不是 raw local XY。

当前单模态 encoder 也是 simplified encoder：对输入做 mean pooling，再 Linear projection 到 `d_model`。它不是最终论文版 JEPA-MSAC modality backbone。

后续正式论文版 TODO：

- replace simplified encoders with stronger modality-specific encoders。
- add missing-pattern evaluation matrix。
- add corrupted-modality protocol。
- add uncertainty calibration metrics。
