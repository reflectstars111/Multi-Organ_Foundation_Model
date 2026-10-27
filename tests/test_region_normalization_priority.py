"""The priority adapter changes the dependency gate, never frozen experiment work."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


ARMS = ['shared_gn', 'domain_gn', 'shared_bn', 'domain_bn']
ORDER = ['finish_active_q01', 'q03', 'remaining_q01']


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_signed(path, value, field):
    value = {key: item for key, item in value.items() if key != field}
    value[field] = digest(value)
    path.write_text(json.dumps(value))
    return value


@pytest.fixture
def adapter():
    return importlib.import_module('scripts.run_region_normalization_priority')


@pytest.fixture
def prepared(tmp_path, adapter):
    q01, q03, scheduler = (tmp_path / name for name in ('q01', 'q03', 'scheduler'))
    for root in (q01, q03, scheduler):
        root.mkdir()
    snapshot = q03 / 'source_snapshot'
    for package in ('scripts', 'unet_moe'):
        (snapshot / package).mkdir(parents=True)
        (snapshot / package / '__init__.py').write_text('')
    # Import a second script to catch polluted parent packages and sys.modules.
    (snapshot / 'scripts/helper.py').write_text('ORIGIN = "snapshot"\n')
    (snapshot / 'scripts/run_region_normalization.py').write_text('''
import json
from pathlib import Path
from scripts.helper import ORIGIN
class Controller:
    def __init__(self, output, after_root, **kwargs):
        self.root, self.reference_root = Path(output), Path(after_root)
        self.protocol = json.loads((self.root / 'protocol.json').read_text())
        self.protocol_sha256 = self.protocol['protocol_sha256']
        self.reference_sha256 = self.protocol['reference_protocol_sha256']
        self.status = {'origin': ORIGIN, 'protocol_sha256': self.protocol_sha256}
    def update(self, state=None, **extra):
        self.status.update(extra)
        if state is not None: self.status['state'] = state
        (self.root / 'controller_status.json').write_text(json.dumps(self.status))
    def wait_for_reference(self):
        raise AssertionError('original full Q01 wait must be replaced')
    def run(self):
        assert set(type(self).__dict__) - {'__module__', '__doc__'} == {'wait_for_reference'}
        self.wait_for_reference()
        self.update('complete', flow=['recover', 'four_preflights', 'train_12',
                                      'training_barrier', 'test_12', 'summary', 'gpu_locks'])
''')
    (snapshot / 'unet_moe/region_normalization_suite.py').write_text('''
import json
from pathlib import Path
def load_protocol(root, check_content=True):
    assert check_content is True
    root = Path(root)
    (root / 'content_checked').touch()
    return json.loads((root / 'protocol.json').read_text()), [], {}, {}
''')
    artifacts = {'records.json': ('record_sha256', [{'id': 1}]),
                 'sampling_plan.json': ('draws_sha256', {'draws': [1]}),
                 'data_content.json': ('content_sha256', {}),
                 'train_statistics.json': ('statistics_sha256', {'n': 1})}
    hashes = {}
    for name, (key, content) in artifacts.items():
        hashes[key] = digest(content)
        for root in (q01, q03):
            (root / name).write_text(json.dumps(content))
    reference = write_signed(q01 / 'protocol.json', {
        'protocol_version': 'q01_organ_weighted_v1', **hashes,
        'initialization_sha256': {'region': {str(seed): 'a' * 64 for seed in (42, 43, 44)}}},
        'protocol_sha256')
    sources = {str(path.relative_to(snapshot)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in snapshot.rglob('*.py')}
    protocol = write_signed(q03 / 'protocol.json', {
        'protocol_version': 'region_domain_normalization_v1',
        'config': {'epochs': 80, 'models': ['region'], 'seeds': [42, 43, 44], 'arms': ARMS},
        'reference_root': str(q01), 'reference_protocol_sha256': reference['protocol_sha256'],
        'source_files': sources, 'source_sha256': digest(sources), **hashes,
        'initialization_sha256': {arm: {str(seed): 'a' * 64 for seed in (42, 43, 44)} for arm in ARMS},
        'non_norm_initialization_sha256': {str(seed): 'b' * 64 for seed in (42, 43, 44)}},
        'protocol_sha256')
    plan = write_signed(scheduler / 'plan.json', {
        'schema': 'medical_queue_priority_v1', 'order': ORDER,
        'q01_root': str(q01), 'q03_root': str(q03),
        'q01_protocol_sha256': reference['protocol_sha256'],
        'q03_protocol_sha256': protocol['protocol_sha256'],
        'source_files': {'scripts/run_region_normalization_priority.py':
                         hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest()}}, 'plan_sha256')
    write_signed(scheduler / 'handoff.json', {
        'plan_sha256': plan['plan_sha256'], 'q01_root': str(q01), 'q03_root': str(q03),
        'q01_protocol_sha256': reference['protocol_sha256'],
        'q03_protocol_sha256': protocol['protocol_sha256'],
        'active_q01_verified': True, 'q01_controller_released': True}, 'handoff_sha256')
    return q01, q03, scheduler


def run(adapter, prepared):
    q01, q03, scheduler = prepared
    adapter.main(['--output', str(q03), '--after-root', str(q01),
                  '--priority-manifest', str(scheduler / 'plan.json'), '--gpus', '0,1'])


def test_preserves_frozen_run_and_overrides_only_wait(adapter, prepared):
    run(adapter, prepared)
    q01, q03, _ = prepared
    status = json.loads((q03 / 'controller_status.json').read_text())
    assert status['state'] == 'complete'
    assert status['flow'] == ['recover', 'four_preflights', 'train_12', 'training_barrier',
                              'test_12', 'summary', 'gpu_locks']
    assert status['reference_policy'] == 'verified_priority_handoff'
    assert status['priority_order'] == ORDER
    assert (q03 / 'content_checked').exists()
    assert not (q01 / 'summary.json').exists()
    assert status['protocol_sha256'] == json.loads((q03 / 'protocol.json').read_text())['protocol_sha256']


def test_imports_from_snapshot_and_restores_modules(adapter, prepared, monkeypatch):
    poisoned = ModuleType('scripts.helper')
    poisoned.ORIGIN = 'mutable_checkout'
    monkeypatch.setitem(sys.modules, 'scripts.helper', poisoned)
    original_path = list(sys.path)
    run(adapter, prepared)
    assert json.loads((prepared[1] / 'controller_status.json').read_text())['origin'] == 'snapshot'
    assert sys.modules['scripts.helper'] is poisoned
    assert sys.path == original_path


@pytest.mark.parametrize('change', ['source', 'plan_digest', 'adapter_hash', 'handoff_absent',
                                   'handoff_false', 'handoff_digest', 'data', 'reference_digest',
                                   'q03_digest', 'scope'])
def test_rejects_invalid_evidence_before_training(adapter, prepared, change):
    q01, q03, scheduler = prepared
    if change == 'source':
        (q03 / 'source_snapshot/scripts/helper.py').write_text('ORIGIN = "tampered"')
    elif change == 'handoff_absent':
        (scheduler / 'handoff.json').unlink()
    elif change == 'data':
        (q01 / 'records.json').write_text('[]')
    elif change == 'reference_digest':
        path = q01 / 'protocol.json'
        data = json.loads(path.read_text()); data['record_sha256'] = '9' * 64
        path.write_text(json.dumps(data))
    elif change in ('handoff_false', 'handoff_digest'):
        path = scheduler / 'handoff.json'
        data = json.loads(path.read_text()); data['active_q01_verified'] = False
        if change == 'handoff_false': write_signed(path, data, 'handoff_sha256')
        else: path.write_text(json.dumps(data))
    elif change in ('plan_digest', 'adapter_hash'):
        path = scheduler / 'plan.json'
        data = json.loads(path.read_text())
        data['source_files']['scripts/run_region_normalization_priority.py'] = '0' * 64
        if change == 'adapter_hash': write_signed(path, data, 'plan_sha256')
        else: path.write_text(json.dumps(data))
    else:
        path = q03 / 'protocol.json'; data = json.loads(path.read_text())
        data['config']['epochs'] = 81
        if change == 'q03_digest': path.write_text(json.dumps(data))
        else:
            data = write_signed(path, data, 'protocol_sha256')
            plan_path = scheduler / 'plan.json'; plan = json.loads(plan_path.read_text())
            plan['q03_protocol_sha256'] = data['protocol_sha256']
            plan = write_signed(plan_path, plan, 'plan_sha256')
            hp = scheduler / 'handoff.json'; handoff = json.loads(hp.read_text())
            handoff.update(q03_protocol_sha256=data['protocol_sha256'], plan_sha256=plan['plan_sha256'])
            write_signed(hp, handoff, 'handoff_sha256')
    with pytest.raises((ValueError, FileNotFoundError)):
        run(adapter, prepared)
    status_file = q03 / 'controller_status.json'
    if status_file.exists():
        assert json.loads(status_file.read_text()).get('state') != 'complete'
