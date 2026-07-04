from collections import Counter, defaultdict
from typing import Any

from kd_sensing.diagnostics.research_claim_harvester_base import (
    CONSISTENCY_FIELDS,
    STRICT_FIELDS,
    ComparabilityWarning,
    _canonical_value,
    _missing,
)


def apply_strict_comparability_gate(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    output = [dict(candidate, warnings=list(candidate.get("warnings", []))) for candidate in candidates]
    for candidate in output:
        grouped[(candidate.get("method"), candidate.get("pattern"), candidate.get("run_family"))].append(candidate)

    for group in grouped.values():
        for field_name in STRICT_FIELDS:
            for candidate in group:
                if _missing(candidate.get(field_name)):
                    _append_warning(
                        candidate,
                        ComparabilityWarning(
                            field=field_name,
                            kind="required_missing",
                            severity="needs_review",
                            message=f"Strict comparability requires {field_name}.",
                        ),
                    )
        for field_name in CONSISTENCY_FIELDS:
            observed = [_canonical_value(candidate.get(field_name)) for candidate in group if not _missing(candidate.get(field_name))]
            if len(set(observed)) <= 1:
                continue
            expected = observed[0]
            for candidate in group:
                actual = _canonical_value(candidate.get(field_name))
                _append_warning(
                    candidate,
                    ComparabilityWarning(
                        field=field_name,
                        kind="field_conflict",
                        expected=expected,
                        actual=actual,
                        severity="not_comparable",
                        message=f"{field_name} differs within the candidate group.",
                    ),
                )
        seeds = [str(candidate.get("seed")) for candidate in group if not _missing(candidate.get("seed"))]
        duplicate_seeds = {seed for seed, count in Counter(seeds).items() if count > 1}
        if duplicate_seeds and len(group) > 1:
            for candidate in group:
                if str(candidate.get("seed")) in duplicate_seeds:
                    _append_warning(
                        candidate,
                        ComparabilityWarning(
                            field="seed",
                            kind="duplicate_seed",
                            actual=candidate.get("seed"),
                            severity="needs_review",
                            message="Multiple candidates in this group share the same seed.",
                        ),
                    )
        for candidate in group:
            severities = {warning.get("severity") for warning in candidate.get("warnings", [])}
            if "not_comparable" in severities:
                candidate["comparability_status"] = "not_comparable"
            elif "needs_review" in severities:
                candidate["comparability_status"] = "needs_review"
            else:
                candidate["comparability_status"] = "strict"
            candidate["next_action_hints"] = _candidate_next_actions(candidate.get("warnings", []))
    return output

def _required_warnings(data: dict[str, Any]) -> list[dict[str, Any]]:
    warnings = []
    for field_name in STRICT_FIELDS:
        if _missing(data.get(field_name)):
            warnings.append(
                ComparabilityWarning(
                    field=field_name,
                    kind="required_missing",
                    severity="needs_review",
                    message=f"Strict comparability requires {field_name}.",
                ).to_dict()
            )
    return warnings

def _append_warning(candidate: dict[str, Any], warning: ComparabilityWarning) -> None:
    record = warning.to_dict()
    duplicate = any(
        existing.get("field") == record.get("field")
        and existing.get("kind") == record.get("kind")
        and existing.get("expected") == record.get("expected")
        and existing.get("actual") == record.get("actual")
        for existing in candidate.get("warnings", [])
    )
    if not duplicate:
        candidate.setdefault("warnings", []).append(record)

def _candidate_next_actions(warnings: list[dict[str, Any]]) -> list[str]:
    fields = {warning.get("field") for warning in warnings}
    actions = []
    if "checkpoint_provenance" in fields:
        actions.append("add checkpoint sidecar or selected checkpoint provenance")
    missing = sorted(str(field) for field in fields if field and field != "checkpoint_provenance")
    if missing:
        actions.append("fill strict comparability fields: " + ", ".join(missing))
    if any(warning.get("severity") == "not_comparable" for warning in warnings):
        actions.append("rerun or separate non-comparable candidates before claim review")
    return actions
