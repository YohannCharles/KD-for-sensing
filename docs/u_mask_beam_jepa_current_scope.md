# U-MaskBeamJEPA Current Scope

U-MaskBeamJEPA 服务 `configs/mmw/t2.yaml`、`s1.yaml` 与 `configs/deepsense6g/t2.yaml`。保留 supervised router、prototype/BPA、embedded teacher CE、same-model temporal superset consistency 和 active CMA ablation；S1 仅关闭 superset consistency，且保持 MMW 对照角色。

其余 branch 已退役，不提供兼容入口。四模态输入、训练和评估边界以 `openspec/specs/u-mask-beam-jepa/spec.md` 为准。
