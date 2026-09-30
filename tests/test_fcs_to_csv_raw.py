import csv
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from fcs_to_csv_raw import convert_one


def test_export_uses_unprocessed_values_and_original_channel_names(tmp_path, monkeypatch):
    class FakeFlowData:
        def __init__(self, path):
            assert path == tmp_path / 'sample.fcs'
            self.pnn_labels = ['FSC-A', 'Marker, A']
            self.event_count = 2
            self.channel_count = 2

        def as_array(self, preprocess=True):
            assert preprocess is False
            return np.array([[12.25, -0.5], [0.0, 1000000.0]], dtype=np.float64)

    monkeypatch.setitem(sys.modules, 'flowio', SimpleNamespace(FlowData=FakeFlowData))
    source = tmp_path / 'sample.fcs'
    source.write_bytes(b'not read by fake')
    target = tmp_path / 'out' / 'sample.csv'
    assert convert_one(source, target, chunk_rows=1, overwrite=False) == (2, 2)
    with target.open(encoding='utf-8', newline='') as handle:
        assert list(csv.reader(handle)) == [
            ['FSC-A', 'Marker, A'], ['12.25', '-0.5'], ['0', '1000000']]
    with pytest.raises(FileExistsError):
        convert_one(source, target, chunk_rows=1, overwrite=False)


def test_failure_removes_partial_csv(tmp_path, monkeypatch):
    class FakeFlowData:
        event_count = 2
        channel_count = 1
        pnn_labels = ['A']

        def __init__(self, _path):
            pass

        def as_array(self, preprocess=True):
            return np.array([[1.], [2.]])

    monkeypatch.setitem(sys.modules, 'flowio', SimpleNamespace(FlowData=FakeFlowData))
    source = tmp_path / 'sample.fcs'
    source.write_bytes(b'fake')
    target = tmp_path / 'sample.csv'
    with pytest.raises(ValueError):
        convert_one(source, target, chunk_rows=0, overwrite=False)
    assert not target.exists()
    assert not list(tmp_path.glob('*.tmp'))
