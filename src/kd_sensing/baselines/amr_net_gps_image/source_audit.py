from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping


IEEE11282996_ARTICLE_NUMBER = "11282996"
IEEE11282996_URL = "https://ieeexplore.ieee.org/document/11282996/"
SCENARIO23_AUTHOR_CODE_URL = "https://github.com/gourangc/Sensing-Aided-Drone-Beam-Prediction"
SCENARIO23_AUTHOR_CODE_COMMIT = "4b80592ed3517726f3fc5af441db25acd1811d3e"

CLAIM_STATUSES = {
    "blocked_official",
    "paper_protocol_audited",
    "local_substitute",
    "local_control",
    "mock_smoke",
    "official_reproduction",
}
OFFICIAL_READY_STATUSES = {"available", "audited", "provided", "not_required"}


@dataclass(frozen=True)
class SourceAudit:
    article_number: str
    ieee_url: str
    title: str
    doi: str
    venue: str
    year: int | None
    pdf_available: bool
    pdf_source: str
    source_availability: str
    official_code_url: str | None
    official_code_commit: str | None
    author_code_url: str | None
    author_code_commit: str | None
    dataset_scene: int | None
    dataset_scene_slug: str | None
    split: dict[str, Any]
    modalities: tuple[str, ...]
    target_label: str
    metric_profile: str
    official_weights_status: str
    official_split_status: str
    official_training_protocol_status: str
    status: str
    claim_status: str
    blocked_reasons: tuple[str, ...] = ()
    article_metadata_conflict: bool = False
    local_substitute: dict[str, Any] = field(default_factory=dict)
    evidence_urls: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "article_number": self.article_number,
            "ieee_url": self.ieee_url,
            "title": self.title,
            "doi": self.doi,
            "venue": self.venue,
            "year": self.year,
            "pdf_available": self.pdf_available,
            "pdf_source": self.pdf_source,
            "source_availability": self.source_availability,
            "official_code_url": self.official_code_url,
            "official_code_commit": self.official_code_commit,
            "author_code_url": self.author_code_url,
            "author_code_commit": self.author_code_commit,
            "dataset_scene": self.dataset_scene,
            "dataset_scene_slug": self.dataset_scene_slug,
            "split": dict(self.split),
            "modalities": list(self.modalities),
            "target_label": self.target_label,
            "metric_profile": self.metric_profile,
            "official_weights_status": self.official_weights_status,
            "official_split_status": self.official_split_status,
            "official_training_protocol_status": self.official_training_protocol_status,
            "status": self.status,
            "claim_status": self.claim_status,
            "blocked_reasons": list(self.blocked_reasons),
            "article_metadata_conflict": self.article_metadata_conflict,
            "local_substitute": dict(self.local_substitute),
            "evidence_urls": list(self.evidence_urls),
        }
        payload["digest"] = source_audit_digest(payload)
        return payload


def build_default_source_audit(*, claim_status: str | None = None) -> SourceAudit:
    """Return the audited default state for the current change.

    IEEE document 11282996 resolves to a different JIOT article than the
    Scenario 23 GPS+Image author package.  The helper therefore blocks official
    claims while preserving a local substitute workflow for Scenario 23.
    """

    blocked = (
        "ieee_document_11282996_metadata_conflict_with_scenario23_author_package",
        "ieee_pdf_not_provided_locally",
        "official_11282996_code_not_audited",
        "official_11282996_split_not_audited",
        "official_11282996_weights_missing",
        "scenario23_author_package_is_ieee_document_10000718_not_11282996",
    )
    local = {
        "article_number": "10000718",
        "ieee_url": "https://ieeexplore.ieee.org/document/10000718/",
        "title": "Towards Real-World 6G Drone Communication: Position and Camera Aided Beam Prediction",
        "doi": "10.1109/GLOBECOM48099.2022.10000718",
        "venue": "GLOBECOM 2022 - 2022 IEEE Global Communications Conference",
        "year": 2022,
        "dataset_scene": 23,
        "dataset_scene_slug": "scene23",
        "author_code_url": SCENARIO23_AUTHOR_CODE_URL,
        "author_code_commit": SCENARIO23_AUTHOR_CODE_COMMIT,
        "author_code_folders": ["image_beam", "pos_beam"],
        "author_generated_csv": {
            "raw": "scenario23_dev/scenario23.csv",
            "image_train": "image_beam/scenario23_img_beam_train.csv",
            "image_val": "image_beam/scenario23_img_beam_val.csv",
            "image_test": "image_beam/scenario23_img_beam_test.csv",
            "position_train": "pos_beam/scenario23_pos_beam_train.csv",
            "position_val": "pos_beam/scenario23_pos_beam_val.csv",
            "position_test": "pos_beam/scenario23_pos_beam_test.csv",
        },
        "modalities": ["image", "gps"],
        "gps_feature_mode": "author_minmax_lat_lon_2d",
        "target_label": "unit1_beam",
        "metric_profile": "top1_top3_top5_accuracy",
        "claim_boundary": "local_substitute_only_for_amr_net_gps_image",
    }
    requested = claim_status or "blocked_official"
    ensure_claim_status_allowed(
        requested,
        {
            "article_metadata_conflict": True,
            "pdf_available": False,
            "official_code_status": "missing",
            "official_split_status": "missing",
            "official_weights_status": "missing",
            "official_training_protocol_status": "missing",
        },
    )
    return SourceAudit(
        article_number=IEEE11282996_ARTICLE_NUMBER,
        ieee_url=IEEE11282996_URL,
        title="Toward Reliable Multimodal Beam Prediction in mmWave Communications via Probabilistic Embedding and Uncertainty-Aware",
        doi="10.1109/JIOT.2025.3641184",
        venue="IEEE Internet of Things Journal",
        year=2026,
        pdf_available=False,
        pdf_source="not_provided; Crossref metadata exposes IEEE PDF URL but local official PDF was not audited",
        source_availability="metadata_conflict_blocked_official",
        official_code_url=None,
        official_code_commit=None,
        author_code_url=SCENARIO23_AUTHOR_CODE_URL,
        author_code_commit=SCENARIO23_AUTHOR_CODE_COMMIT,
        dataset_scene=None,
        dataset_scene_slug=None,
        split={"status": "official_11282996_split_not_audited", "local_substitute_scene23": local["author_generated_csv"]},
        modalities=("image", "gps"),
        target_label="beam_index_unverified_for_11282996",
        metric_profile="amr_net_gps_image_top1_top3_top5",
        official_weights_status="missing",
        official_split_status="missing",
        official_training_protocol_status="missing",
        status="blocked_official",
        claim_status=requested,
        blocked_reasons=blocked,
        article_metadata_conflict=True,
        local_substitute=local,
        evidence_urls=(
            "https://api.crossref.org/works/10.1109%2FJIOT.2025.3641184",
            "https://api.crossref.org/works/10.1109%2FGLOBECOM48099.2022.10000718",
            "https://raw.githubusercontent.com/gourangc/Sensing-Aided-Drone-Beam-Prediction/main/README.md",
        ),
    )


def source_audit_digest(audit: Mapping[str, Any] | SourceAudit) -> str:
    payload = audit.to_dict() if isinstance(audit, SourceAudit) else dict(audit)
    payload.pop("digest", None)
    canonical = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def missing_official_requirements(audit: Mapping[str, Any] | SourceAudit) -> list[str]:
    payload = audit.to_dict() if isinstance(audit, SourceAudit) else dict(audit)
    missing: list[str] = []
    if payload.get("article_metadata_conflict"):
        missing.append("article_metadata_conflict")
    if not bool(payload.get("pdf_available", False)):
        missing.append("official_pdf")
    if str(payload.get("official_code_status") or payload.get("source_availability") or "").lower() not in OFFICIAL_READY_STATUSES:
        missing.append("official_code")
    if str(payload.get("official_split_status") or "").lower() not in OFFICIAL_READY_STATUSES:
        missing.append("official_split")
    if str(payload.get("official_weights_status") or "").lower() not in OFFICIAL_READY_STATUSES:
        missing.append("official_weights")
    if str(payload.get("official_training_protocol_status") or "").lower() not in OFFICIAL_READY_STATUSES:
        missing.append("official_training_or_evaluation_protocol")
    if not payload.get("metric_profile"):
        missing.append("metric_profile")
    return missing


def ensure_claim_status_allowed(claim_status: str, audit: Mapping[str, Any] | SourceAudit) -> str:
    status = str(claim_status)
    if status not in CLAIM_STATUSES:
        raise ValueError(f"Unknown IEEE 11282996 claim status '{claim_status}'. Available: {sorted(CLAIM_STATUSES)}.")
    missing = missing_official_requirements(audit)
    if status == "official_reproduction" and missing:
        raise ValueError(
            "IEEE 11282996 official_reproduction is blocked until official evidence is complete; "
            f"missing: {missing}."
        )
    return status


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value
