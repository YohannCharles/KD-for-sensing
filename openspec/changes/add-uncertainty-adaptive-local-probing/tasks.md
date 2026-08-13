## 1. Contract reset

- [x] 1.1 将 active/current specs 改为原生四模态、单阶段、无参数 posterior fusion 与原生 15-mask evidence。
- [x] 1.2 从 README、agent navigation、clean-data/MMW/repo boundary 中删除 Stage2/3、risk/fusion 与 sparse-CSI active claims。

## 2. Model and training cleanup

- [x] 2.1 用 `four_modal_topology_predictor` 替换旧 model owner，保留 encoders、shared temporal transformer、prototype bank 和 posterior statistics。
- [x] 2.2 用单一 topology-prototype loss/config 替换 stage-aware risk loss，并注册 topology on/off matched configs。
- [x] 2.3 删除 sparse-CSI sidecar/encoder/simulator、stage preparation/gate/initialization/continuation 接线与旧 registry/config fields。

## 3. Evaluation and tools

- [x] 3.1 将 prediction evidence 改为原生四模态 15-mask，删除 risk/fusion replacement 与五模态过滤逻辑。
- [x] 3.2 精简本地 runner/evaluator，只保留单阶段 resolve/preflight/train、matrix evidence、likelihood 与 probing actions。
- [x] 3.3 保持 TBCP requested-only、train-only likelihood 与 protocol/topology provenance 测试通过。

## 4. Tests and verification

- [x] 4.1 删除旧 risk/fusion/stage/sparse-CSI tests，新增四模态 state dict、masked mean、topology on/off、15-mask 与旧字段拒绝测试。
- [x] 4.2 运行 focused tests、OpenSpec strict、CLI/config、compile 和 full pytest；不访问 test、不启动训练。

## 5. Ignored artifact cleanup

- [x] 5.1 删除确认清单中的旧 PCPF、五模态 TBCP replay 与 sparse-CSI cache，保留 split、topology audit 和 `tbcp7_probe_calibration`。
- [x] 5.2 核验删除路径消失、保留路径存在并报告实际回收空间。

## 6. K=3 主线与联动诊断修订

- [x] 6.1 将主 probing 契约改为 TBCP-3/Batch-TBCP-2+1，K=5/7/9 保留为固定预算 sensitivity，并同步报告、反馈开销和方法命名。
- [x] 6.2 更新 probing focused tests，覆盖 TBCP-3 默认预算、2+1 batch、主线配置和 K=3/5/7/9 sensitivity 不漂移。
- [x] 6.3 读取 topology-on/off 三 seed 15-mask 证据，完成 posterior 熵/置信度与 modality-pattern 尤其 Radar-only 的联动诊断；不访问 test、不修改 checkpoint。
- [x] 6.4 第一轮诊断结论为拓扑辅助项过重且纯单模态暴露不足；先完成 radar-robust matched 重训。后续 Full-posterior 诊断将剩余问题定位为弱 Radar 的固定等权稀释，修复范围在第 10 节重新预注册，且不使用 validation 调权重。
- [x] 6.5 运行 focused tests、OpenSpec strict 与 compile/config checks，并记录 TBCP-3 主线的 validation-only claim 边界。

## 7. Radar robustness evidence consolidation

- [x] 7.1 核对 radar-robust topology-on/off seed 1/2/3 的 validation-best sidecar、train-only normalization、whole-modality schedule 与 test seal。
- [x] 7.2 核对六个 checkpoint 的完整 15-mask validation matrix、TBCP-3/Batch-TBCP-2+1 replay、K=5/7/9 sensitivity 与三 seed summary。
- [x] 7.3 记录 topology-on 的总体收益与 Radar-only 负交互边界，停止继续 validation-driven Radar 调参。

## 8. 公平 ablation 与 baseline 实现

- [x] 8.1 为 topology loss 增加预注册 `hard_label_smoothing`，实现 uniform label smoothing control，并补 config/loss focused tests。
- [x] 8.2 新增 Soft-only、Prototype-only、Uniform-LS 三组 local templates，固定 whole-modality schedule、40 epochs、seed 1/2/3 和独立 output root。
- [x] 8.3 为 AMBER-Full-local/RMBP-MM-local 固定 sensing-only whole-modality 训练编排、validation-best checkpoint selection、cache/protocol provenance 和独立 output root。
- [x] 8.4 新增 generic baseline posterior evidence adapter；复用统一 15-mask、train-only likelihood、Direct/Posterior/OpenLoop/TBCP-3 evaluator，不放宽 native topology loader。
- [x] 8.5 补齐 ablation/baseline provenance、mask identity、checkpoint role、test seal 和 CLI smoke tests。

## 9. 运行与汇总

- [x] 9.1 启动 Soft-only、Prototype-only、Uniform-LS × seed 1/2/3，共9条 fresh validation-only训练。
- [x] 9.2 启动 AMBER-Full-local、RMBP-MM-local × seed 1/2/3，共6条 fresh validation-only训练。
- [x] 9.3 对15条 checkpoint 生成15-mask evidence和统一 Direct/Posterior/OpenLoop/TBCP-3 replay，汇总All-15、Full/drop-1/drop-2/Single Macro/Worst及三seed mean/std。
- [ ] 9.4 生成 paired delta、参数量/训练预算/3-slot probing开销表；不访问test，不把privileged history reference混入主排名。

## 10. Static reliability fusion repair

- [x] 10.1 用冻结 validation evidence完成 Full posterior 诊断，记录等权 Radar 稀释、no-Radar 反事实与 topology/TBCP 非主因；不访问 test、不调权重。
- [x] 10.2 为 predictor 增加可选四标量 `trainable_static_reliability` fusion；mean control 保持无参数，补参数、Single-mask、mask renormalization 与 gradient tests。
- [x] 10.3 新增 static-reliability topology off/on tracked templates，固定 whole-modality schedule、40 epochs、seed 1/2/3、fresh-start 和独立 output root。
- [x] 10.4 运行 focused model/config tests、OpenSpec strict、compile/config checks。
- [x] 10.5 启动 static-reliability topology-on seed1 fresh validation-only pilot；停止其余预启动任务并保留中断记录，不进行多 seed 资源投入。
- [x] 10.6 对 seed1 pilot 生成15-mask evidence/TBCP-3 replay，与既有 mean topology-on seed1 比较 learned weights、Full/missing-radar/All-15/Drop/Single/Radar-only；记录总体改善与GPS-only/collapse边界。
- [x] 10.7 无界 static-reliability seed1 虽改善 Full/All-15，但出现 LiDAR 97% 塌缩和 GPS-only 退化；停止该分支的完整多 seed panel并保留诊断证据。
- [x] 10.8 增加独立 `bounded_static_reliability` 模式：复用四个全局 logits，在 availability-masked softmax 前固定 `tanh`，保持 mean/旧无界模式语义不变，并补上限、初始化、Single-mask、梯度和 checkpoint metadata tests。
- [x] 10.9 新增 bounded-static-reliability topology off/on tracked templates，固定 whole-modality schedule、40 epochs、fresh-start 和独立 output root；运行 focused/OpenSpec/config/compile 验证。
- [x] 10.10 完成 bounded topology-on seed1 fresh validation-only pilot、15-mask与TBCP-3；权重不再单模态独占，但进入两组边界饱和。
- [x] 10.11 bounded seed1 的All-15 TBCP-3相对mean为 `-0.01 pp`，停止 bounded off/on多seedpanel并保留诊断证据。

## 11. Standard masked feature fusion

- [x] 11.1 预注册标准 `masked_feature_mlp` 与单模态/融合特征共享唯一Prototype Bank契约；明确该backbone不作为创新点，且不读取额外信息。
- [x] 11.2 在 predictor 内增加固定两层mask-aware feature MLP；保持mean/static诊断语义不变，并补mask归零、无显式gate、唯一Bank、shape和gradient tests。
- [x] 11.3 新增 feature-fusion topology off/on及两条组件诊断tracked templates，固定whole-modality schedule、现有radar-robust topology loss、40 epochs、fresh-start和独立output root；focused/OpenSpec/config/compile验证通过。
- [x] 11.4 按用户授权启动 feature-fusion topology on/off × seed 1/2/3 六条matched fresh validation-only训练，并启动soft-only/prototype-only seed1两条组件诊断，共八条独立GPU任务；启动后确认八卡均进入训练、test封存且无OOM。
- [x] 11.5 完成八条任务的15-mask/TBCP-3；结果显示feature MLP修复Full候选覆盖，但soft+prototype组合低于两个单项，触发第12节的统一loss修复。

## 12. Joint topology prototype loss

- [x] 12.1 预注册唯一 `joint_topology_weight=0.1`：对fused与availability-normalized unimodal环形soft CE等权平均一次；Hard control为0，旧三项在两模板中均置零。
- [x] 12.2 在现有four-modal topology loss中实现joint项与严格config诊断，复用唯一Bank/soft target，不新增模型模块；补公式、mask归一化、旧分项不重复和gradient tests。
- [x] 12.3 新增masked-feature Joint/Hard seed1 matched templates，运行focused/OpenSpec/config/compile验证。
- [ ] 12.4 在两张空闲GPU启动两条fresh validation-only训练；完成后运行15-mask/TBCP-3并执行go/no-go，不预启动多seed。

## 13. Prototype-only stability replication

- [x] 13.1 由用户授权在GPU0/1并行启动 masked-feature Prototype-only seed2/3；严格复用seed1 template、whole-modality schedule、40 epoch、validation-best、正式protocol/cache与test seal，每条使用独立resolved config、run目录和日志。
- [x] 13.2 两条训练完成后生成15-mask/TBCP-3 evidence，与seed1汇总三seed均值/标准差及对Hard control的paired delta；不访问outer test、不根据结果调参。

## 14. DeepSense6G secondary transfer

- [x] 14.1 固化filtered Scene31–34 protocol、linear-index Prototype-only、40 epoch/last checkpoint与one-shot test契约；明确不迁移MMW TBCP。
- [x] 14.2 为topology predictor增加linear topology支持和DeepSense6G窄配置校验；MMW ULA/test-seal契约保持不变并补focused tests。
- [x] 14.3 增加三方法×三seed的DeepSense6G配置生成/八卡队列，运行首batch/model/loss smoke并检查独立输出与日志。
- [ ] 14.4 完成9条fresh训练与15-mask Direct/Top-3/Top-5/DBA one-shot test，按scene与缺失模态数汇总；不报告DeepSense TBCP。

## 15. Frozen-mainline defense and final MMW test

- [x] 15.1 对冻结 Prototype-only 三seed运行 `{0,3,6} dB` matched measurement-error replay；3/6 dB各三个noise replicas，0 dB复用确定性锚点，并生成三seed汇总。
- [x] 15.2 实现并运行 Prototype-only vs Hard/RMBP 的trajectory与domain cluster paired bootstrap，固定10000次、seed 20260813，输出Direct/Posterior Top-3/TBCP-3的95% CI。
- [x] 15.3 对四个冻结方法测量参数量、profiler-covered FLOPs、batch1 sensing forward median/p95，并生成TBCP-3与Full-64的measurement/round/update开销表。
- [ ] 15.4 实现MMW final-test seal/preflight；在不构建test loader的条件下核验四方法×三seed的checkpoint/config/protocol/topology/normalization/likelihood/15-mask完整性。
- [ ] 15.5 仅在15.1–15.4完成且冻结manifest通过后统一解封一次test，生成12个run的Direct/Posterior Top-3/TBCP-3及缺失0/1/2/3模态汇总；之后不再调参。
