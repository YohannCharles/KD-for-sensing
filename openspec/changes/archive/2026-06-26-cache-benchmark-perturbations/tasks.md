## 1. OpenSpec contract

- [x] 1.1 Add benchmark perturbation cache requirements and scenarios.
- [x] 1.2 Add difficulty pipeline cache provenance requirements.

## 2. Implementation

- [x] 2.1 Add a small perturbation cache helper that can write/read perturbed batch payloads by deterministic cache key.
- [x] 2.2 Wire benchmark perturbation generation to the cache helper with `off/write/read/read_write` modes.
- [x] 2.3 Expose manifest/config fields for cache mode and cache directory without changing default online behavior.

## 3. Tests and validation

- [x] 3.1 Add synthetic cache round-trip coverage for deterministic batch perturbations.
- [x] 3.2 Run focused JEPA benchmark and modality difficulty tests.
- [x] 3.3 Run `openspec validate cache-benchmark-perturbations --strict`.
