"""Read-only inspection must not discard an experiment's accumulated summary."""
from types import SimpleNamespace

from unet_moe import fair_suite


def test_resume_dry_run_preserves_existing_summary(tmp_path, monkeypatch):
    summary = tmp_path / 'summary.json'
    original = '{"runs": [{"kind": "region", "epochs_completed": 12}]}\n'
    summary.write_text(original)
    protocol = dict(config_sha256='config', record_sha256='records', budget={})
    monkeypatch.setattr(fair_suite, '_prepare_protocol', lambda args: (protocol, [], {}))
    args = SimpleNamespace(output=tmp_path, resume=True, dry_run=True, evaluate_test=False)
    fair_suite.run_suite(args)
    assert summary.read_text() == original
