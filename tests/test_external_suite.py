"""Protocol safety and optimization semantics for Q04, independent of costly networks."""
import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from unet_moe.external_baseline_suite import optimization_step, objective
from unet_moe.external_baselines.protocol import assert_checkpoint, assert_matrix_complete
from unet_moe.fair_suite import common_loss, _rng_state, _restore_rng, _seed
from unet_moe.region import DOMAINS


class Queries:
    def __init__(self):
        gen = torch.Generator().manual_seed(123)
        self.images = torch.rand(7, 1, 8, 8, generator=gen)
        self.targets = (self.images > .4).float()
        self.targets[:, :, :2, :2] = -1
    def __getitem__(self, i):
        return self.images[i], self.targets[i], DOMAINS[i % len(DOMAINS)], ''


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)
    def forward(self, images, domains):
        return dict(logits=self.conv(images), auxiliary_logits=[])


def test_accumulated_loss_and_update_match_full_short_batch():
    torch.manual_seed(1)
    dataset = Queries()
    a, b = Tiny(), Tiny()
    b.load_state_dict(a.state_dict())
    optimizers = [torch.optim.SGD(m.parameters(), lr=.1) for m in (a,b)]
    values = [optimization_step(m, 'lanzhou', opt, dataset, list(range(7)), 'cpu', micro,
                                seed=42, epoch=1)[0]
              for m,opt,micro in zip((a,b),optimizers,(7,2))]
    assert values[0] == pytest.approx(values[1], abs=1e-6)
    for x,y in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(x,y,rtol=1e-6,atol=1e-7)


def test_masked_loss_has_no_unknown_pixel_gradient():
    logits = torch.randn(2,1,8,8,requires_grad=True)
    target = torch.zeros_like(logits)
    target[:,:,:3] = -1
    target[:,:,4:6,4:6] = 1
    common_loss(logits,target).backward()
    assert torch.count_nonzero(logits.grad[:,:,:3]) == 0
    assert torch.count_nonzero(logits.grad[:,:,3:]) > 0


def test_smart_secondary_head_is_supervised_with_declared_weight():
    main = torch.randn(2,1,8,8,requires_grad=True)
    aux = torch.randn_like(main,requires_grad=True)
    target = torch.zeros_like(main)
    target[:,:,3:6,3:6] = 1
    loss = objective(None,'smart',dict(logits=main,auxiliary_logits=[aux]),target,None)
    assert loss.item() == pytest.approx((common_loss(main,target)+.4*common_loss(aux,target)).item())
    loss.backward()
    assert main.grad.abs().sum() > 0 and aux.grad.abs().sum() > 0


def test_adamw_and_rng_checkpoint_resume_matches_uninterrupted(tmp_path):
    _seed(42)
    dataset = Queries()
    model = Tiny()
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    optimization_step(model,'lanzhou',optimizer,dataset,list(range(7)),'cpu',2,seed=42,epoch=1)
    path = tmp_path/'checkpoint.pt'
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=_rng_state()),path)
    draw = torch.rand(3)
    optimization_step(model,'lanzhou',optimizer,dataset,list(range(7)),'cpu',2,seed=42,epoch=2)
    restored = Tiny()
    opt = torch.optim.AdamW(restored.parameters(),lr=1e-4,weight_decay=1e-4)
    cp = torch.load(path,weights_only=True)
    restored.load_state_dict(cp['model']); opt.load_state_dict(cp['optimizer']); _restore_rng(cp['rng'])
    torch.testing.assert_close(torch.rand(3),draw,rtol=0,atol=0)
    optimization_step(restored,'lanzhou',opt,dataset,list(range(7)),'cpu',2,seed=42,epoch=2)
    for a,b in zip(model.parameters(),restored.parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_checkpoint_protocol_and_method_are_strict():
    protocol = dict(protocol_sha256='frozen')
    cp = dict(protocol_sha256='other',method='smart',seed=42)
    with pytest.raises(ValueError):
        assert_checkpoint(cp,protocol,'smart',42)
    cp['protocol_sha256']='frozen'
    with pytest.raises(ValueError):
        assert_checkpoint(cp,protocol,'lanzhou',42)


def test_test_barrier_checks_checkpoint_instead_of_summary_flags(tmp_path):
    protocol = dict(protocol_sha256='frozen',config=dict(methods=['lanzhou'],seeds=[42],epochs=80))
    folder = tmp_path/'lanzhou'/'seed_42'; folder.mkdir(parents=True)
    (folder/'run_summary.json').write_text(json.dumps(dict(epochs_completed=80,test_evaluated=True)))
    torch.save(dict(protocol_sha256='frozen',method='lanzhou',seed=42,epoch=79),folder/'last.pt')
    with pytest.raises(ValueError,match='incomplete'):
        assert_matrix_complete(tmp_path,protocol)


def test_epoch80_crash_before_summary_is_recovered_without_an_extra_epoch(tmp_path, monkeypatch):
    from unet_moe import external_baseline_suite as suite
    from unet_moe.external_baselines.protocol import write_json
    protocol = dict(protocol_sha256='frozen', config=dict(size=512, epochs=80, batch_size=16,
        micro_batch_sizes=dict(lanzhou=2), lr=1e-4, weight_decay=1e-4))
    write_json(tmp_path/'preflight/lanzhou.json', dict(method='lanzhou', passed=True,
        protocol_sha256='frozen', size=512, physical_batch=2, effective_batch=16,
        real_optimizer_updates=2, strict_model_optimizer_rng_reload=True,
        identical_reload_prediction=True))
    write_json(tmp_path/'sampling_plan.json', {'42': []})
    folder = tmp_path/'lanzhou/seed_42'; folder.mkdir(parents=True)
    model = Tiny()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    history = [dict(epoch=n) for n in range(1,81)]
    torch.save(dict(protocol_sha256='frozen', method='lanzhou', seed=42, epoch=80,
        model=model.state_dict(), optimizer=optimizer.state_dict(), history=history,
        best_score=90., best_epoch=80, elapsed_seconds=2., rng=_rng_state()), folder/'last.pt')
    monkeypatch.setattr(suite, 'make_model', lambda *args, **kwargs: Tiny())
    monkeypatch.setattr(suite, '_datasets', lambda *args: {})
    monkeypatch.setattr(suite, '_bank', lambda *args: None)
    monkeypatch.setattr(suite, 'optimization_step', lambda *args, **kwargs: pytest.fail('No 81st epoch'))
    restored = suite.train(tmp_path, protocol, [], 'lanzhou', 42, 'cpu')
    assert len(restored) == 80
    assert json.loads((folder/'run_summary.json').read_text())['epochs_completed'] == 80
    assert (folder/'best.pt').is_file()
    assert len((folder/'metrics.jsonl').read_text().splitlines()) == 80


def test_mofo_main_training_rejects_completed_short_smoke_prior(tmp_path):
    from unet_moe.external_baseline_suite import make_model
    from unet_moe.external_baselines.protocol import file_sha
    (tmp_path/'records.json').write_text('[]')
    (tmp_path/'support_manifest.json').write_text('{}')
    folder = tmp_path/'mofo_prior/seed_42'; folder.mkdir(parents=True)
    torch.save(dict(complete=True, epoch=1, protocol=dict(epochs=1, seed=42, size=512,
        records_sha256=file_sha(tmp_path/'records.json'),
        support_manifest_sha256=file_sha(tmp_path/'support_manifest.json'))), folder/'last.pt')
    protocol = dict(config=dict(size=512, mofo_prior=dict(epochs=80), assets_dir='/unused'))
    with pytest.raises(ValueError, match='80-epoch'):
        make_model(tmp_path, protocol, 'mofo', 42)


def complete_fixture(tmp_path):
    from unet_moe.external_baselines.protocol import write_json
    protocol = dict(protocol_sha256='frozen', config=dict(epochs=80))
    folder = tmp_path/'lanzhou/seed_42'; folder.mkdir(parents=True)
    history = [dict(epoch=n, sample_sha256=f'plan{n}', sample_count=4500, optimizer_steps=282,
                    validation=dict(macro_positive_dsc_percent=float(n))) for n in range(1,81)]
    cp = dict(protocol_sha256='frozen', method='lanzhou', seed=42, epoch=80,
              history=history, best_epoch=80, best_score=80.)
    for name in ('last.pt','best.pt'):
        torch.save(cp, folder/name)
    (folder/'metrics.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in history))
    write_json(folder/'run_summary.json', dict(protocol_sha256='frozen', method='lanzhou',
        seed=42, epochs_completed=80, best_epoch=80, test_evaluated=False))
    write_json(tmp_path/'sampling_plan.json', {'42': [dict(sample_sha256=f'plan{n}') for n in range(1,81)]})
    return protocol, folder


def test_verifier_rejects_modified_metrics_even_when_checkpoint_is_complete(tmp_path):
    from unet_moe.external_baseline_suite import verify_run
    protocol, folder = complete_fixture(tmp_path)
    verify_run(tmp_path, protocol, 'lanzhou', 42)
    with (folder/'metrics.jsonl').open('a') as stream:
        stream.write('{"epoch":81}\n')
    with pytest.raises(ValueError, match='Persisted metrics history'):
        verify_run(tmp_path, protocol, 'lanzhou', 42)


def test_verifier_rejects_best_epoch_inconsistent_with_validation_scores(tmp_path):
    from unet_moe.external_baseline_suite import verify_run
    protocol, folder = complete_fixture(tmp_path)
    cp = torch.load(folder/'last.pt', weights_only=True)
    cp['best_score'] = 99.
    torch.save(cp, folder/'last.pt')
    with pytest.raises(ValueError, match='selection disagrees'):
        verify_run(tmp_path, protocol, 'lanzhou', 42)


def test_reports_keep_undefined_distances_as_na_and_penalized_counts(tmp_path):
    from unet_moe.external_baseline_suite import write_reports
    from unet_moe.reevaluation_report import COHORTS, FIELDS
    import csv
    groups = [dict(method='lanzhou', selection='best', cohort=cohort, seeds=3,
        metrics={field: dict(mean=80. if field=='dsc_percent' else None,
                             sample_std=1. if field=='dsc_percent' else None) for field in FIELDS})
              for cohort in COHORTS]
    metrics = {cohort: dict(dsc_percent=80., hd95_pixel=None, assd_pixel=None,
        n_included=7, n_coverage_total=8, n_penalized=2,
        n_target_boundary_undefined_excluded=1, distance_status='undefined') for cohort in COHORTS}
    results = [dict(method='lanzhou', seed=42, selection='best', split='test',
        surface_metrics=[dict(view='union', groups=dict(AbdomenUS=metrics), abdomen_by_source={'CT1':metrics})])]
    write_reports(tmp_path, {'protocol_sha256':'frozen'}, results, groups)
    rows = list(csv.DictReader((tmp_path/'metrics.csv').open()))
    assert len(rows)==4 and all(row['hd95_pixel_mean']=='NA' for row in rows)
    per_domain = list(csv.DictReader((tmp_path/'per_domain_metrics.csv').open()))
    assert len(per_domain)==8 and all(row['n_penalized']=='2' for row in per_domain)
    text = (tmp_path/'REPORT.md').read_text()
    assert 'GPU preflight' in text and 'NA' in text and '80.00 ± 1.00' in text
