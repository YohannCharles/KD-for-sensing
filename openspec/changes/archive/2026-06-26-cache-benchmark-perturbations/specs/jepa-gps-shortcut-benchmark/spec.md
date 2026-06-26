## ADDED Requirements

### Requirement: Benchmark perturbation cache
JEPA GPS shortcut benchmark MUST support an opt-in perturbation cache for deterministic difficulty suites. The cache MUST be keyed by suite id/type, condition, severity, split, seed, sample ids and difficulty digest, and MUST store perturbed input batches without modifying source dataset files, checkpoints or labels.

#### Scenario: 写出扰动 batch cache
- **WHEN** benchmark manifest enables perturbation cache mode `write` or `read_write`
- **THEN** runner MUST apply the shared difficulty pipeline once for each evaluated batch/condition
- **AND** runner MUST write the perturbed batch, warnings and replay metadata to an ignored local cache directory
- **AND** target labels、beam power、sample id and split metadata MUST remain unchanged

#### Scenario: 从缓存读取扰动 batch
- **WHEN** benchmark manifest enables perturbation cache mode `read`
- **THEN** runner MUST load the matching perturbed batch cache instead of reapplying difficulty operators
- **AND** missing or mismatched cache entries MUST fail with a clear cache key/path error
- **AND** metric rows MUST still record the original difficulty provenance and sample count

#### Scenario: 默认不改变现有评估
- **WHEN** benchmark manifest omits perturbation cache settings
- **THEN** runner MUST preserve existing online perturbation behavior
- **AND** no cache directory MUST be required

### Requirement: Cached benchmark comparability
Cached perturbation reuse MUST NOT weaken benchmark comparability checks. Rows produced from cached inputs MUST remain comparable only when split, label space, sample ids, metric profile, difficulty digest and seed match the requested suite.

#### Scenario: cache provenance 进入输出 manifest
- **WHEN** benchmark uses perturbation cache
- **THEN** benchmark manifest or equivalent output MUST record cache mode, cache directory, cache schema version, cache hits, cache misses and cache writes
- **AND** model config/checkpoint provenance MUST remain separate from cache provenance
