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
