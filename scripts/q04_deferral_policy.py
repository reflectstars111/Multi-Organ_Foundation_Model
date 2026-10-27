"""Auditable user-requested Q04 queue deferral; contains no training changes."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path

ACTIVE_METHODS = ("lanzhou", "aimagelab", "smart", "mofo")
DEFERRED_METHODS = ("universeg",)
SEEDS = (42, 43, 44)
SCHEMA = "q04_user_deferral_v1"
RESUME_POLICY = "manual_review_after_remaining_queue"


def deferral_digest(manifest):
    unsigned = {key: value for key, value in manifest.items() if key != "deferral_sha256"}
    return hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def load_deferral(path, q04_root=None, q02_root=None):
    """Validate the explicit exception and bind it to both original protocols."""
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError("Unknown Q04 deferral schema")
    if data.get("deferral_sha256") != deferral_digest(data):
        raise ValueError("Deferral hash mismatch")
    for field, expected in (("active_methods", list(ACTIVE_METHODS)),
                            ("deferred_methods", list(DEFERRED_METHODS)),
                            ("seeds", list(SEEDS)), ("resume_policy", RESUME_POLICY)):
        if data.get(field) != expected:
            raise ValueError(f"Unsupported deferral {field}")
    if not isinstance(data.get("reason"), str) or not data["reason"].strip():
        raise ValueError("Deferral requires the user's reason")
    try:
        created = datetime.fromisoformat(data["created_at"])
        if created.utcoffset() is None:
            raise ValueError("timezone required")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Deferral requires an aware created_at timestamp") from error
    roots = {}
    for name, supplied in (("q04", q04_root), ("q02", q02_root)):
        raw = data.get(f"{name}_root")
        if not isinstance(raw, str) or not Path(raw).is_absolute():
            raise ValueError(f"Absolute {name} root is required")
        root = Path(raw).resolve()
        if supplied is not None and root != Path(supplied).resolve():
            raise ValueError(f"Deferral {name} root mismatch")
        roots[name] = root
        expected = data.get(f"{name}_protocol_sha256")
        if (not isinstance(expected, str) or len(expected) != 64
                or any(c not in "0123456789abcdef" for c in expected)):
            raise ValueError(f"Invalid {name} protocol fingerprint")
        protocol = json.loads((root / "protocol.json").read_text())
        if protocol.get("protocol_sha256") != expected:
            raise ValueError(f"Deferral {name} protocol mismatch")
    if roots["q04"] == roots["q02"] or path != roots["q04"] / "queue_deferral.json":
        raise ValueError("Deferral must belong to a distinct Q04 root")
    return data


def active_matrix(manifest):
    return {(method, seed) for method in manifest["active_methods"] for seed in manifest["seeds"]}


def validate_remaining_summary(summary, manifest):
    """Check the exact 12-run / 24-checkpoint scope without pretending all Q04 finished."""
    if (summary.get("complete") is not False or summary.get("active_scope_complete") is not True
            or summary.get("protocol_sha256") != manifest["q04_protocol_sha256"]
            or summary.get("deferral_sha256") != manifest["deferral_sha256"]
            or any(summary.get(key) != manifest[key]
                   for key in ("active_methods", "deferred_methods", "seeds"))):
        raise ValueError("Remaining summary does not match the authorized scope")
    expected = {(method, seed, selection) for method, seed in active_matrix(manifest)
                for selection in ("best", "last")}
    results = summary.get("results", [])
    actual = [(r.get("method"), r.get("seed"), r.get("selection")) for r in results]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Remaining summary requires exactly 24 active test checkpoints")
    if any(r.get("complete") is not True or r.get("split") != "test"
           or r.get("protocol_sha256") != manifest["q04_protocol_sha256"]
           or r.get("deferral_sha256") != manifest["deferral_sha256"] for r in results):
        raise ValueError("Remaining summary contains unbound/incomplete test results")
    return summary
