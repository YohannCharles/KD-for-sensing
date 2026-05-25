## 1. Cache validation and sidecar metadata

- [x] 1.1 Extend Multimodal-NF derived-cache sidecar metadata with lightweight source identity fields, cache schema version, source key, sample count, bytes, shape, dtype, and access layout, while preserving backward compatibility for existing sidecars.
- [x] 1.2 Change runtime cache resolution so `read_only` training uses lightweight validation by default and does not recompute full source SHA fingerprints during dataset construction.
- [x] 1.3 Add an explicit strong-validation path for users who need full source fingerprint verification, and record validation mode, duration, and result in cache metadata or run metadata.
- [x] 1.4 Add tests covering `read_only` cache initialization without source fingerprint calls, strong-validation mismatch detection, and old sidecar fallback behavior.

## 2. Dataset cache IO boundaries and instrumentation

- [x] 2.1 Expose runtime metadata for cache storage kind, active modalities, cache files opened per worker, cache layout, and expected read pattern.
- [x] 2.2 Instrument Multimodal-NF dataset/cache adapter code with low-overhead counters or timings for cache open and batch read phases without changing returned samples.
- [x] 2.3 Ensure each worker lazily opens only the cache file or shard required by its active modality path, including image-only, lidar-only, fusion, and GPS/beam runs.
- [x] 2.4 Add tests for sample equivalence between cached and uncached reads, runtime metadata fields, and no access to inactive modality cache files.

## 3. Locality-aware sampling and run metadata

- [x] 3.1 Add or formalize a locality-aware train subsampling mode that selects the same logical epoch subset as random subsampling but orders selected indices by source/cache locality when enabled.
- [x] 3.2 Preserve compatibility for existing `training.epoch_subsampling.shuffle` behavior and document the exact mapping from old random order to the new locality-aware option.
- [x] 3.3 Record subsampling shuffle, order/locality strategy, seed, subset size, and any block parameters in resolved config or run metadata.
- [x] 3.4 Add deterministic tests proving locality ordering does not change the selected sample set and remains reproducible across runs.

## 4. Profiling and parallel-training recommendations

- [x] 4.1 Extend `scripts/profile_training_io.py` to report dataset init time, cache validation time, cache open time, cache read time or proxy counters, DataLoader wait, and IO-risk classification for Multimodal-NF.
- [x] 4.2 Update the parallel-training recommendation workflow to emit Multimodal-NF-specific advice for lightweight cache validation, `training.epoch_subsampling.shuffle=false` or locality mode, `output.progress.enabled=false`, AMP, worker/prefetch settings, and GPU assignment of image/lidar/fusion jobs.
- [x] 4.3 Add or update Multimodal-NF experiment config examples so optimized settings are discoverable without changing existing config compatibility.
- [x] 4.4 Add tests covering profile JSON fields and recommendation output for representative image, lidar, fusion, GPS/beam, and LOS runs.

## 5. Validation

- [x] 5.1 Run `conda run -n kd_mm_beam pytest tests/test_multimodal_nf_dataset.py -q`.
- [x] 5.2 Run `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py -q`.
- [x] 5.3 Run `conda run -n kd_mm_beam pytest tests/test_epoch_subsampling.py tests/test_parallel_training_recommendations.py -q`, adding or adjusting these tests if the implementation introduces new test modules.
- [x] 5.4 Run `openspec validate reduce-multimodal-nf-cache-io --strict`.
- [x] 5.5 Run `openspec status --change reduce-multimodal-nf-cache-io`.
