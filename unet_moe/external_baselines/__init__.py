"""Q04 source-preserving external segmentation baselines.

See docs/Q04_EXTERNAL_BASELINES_512.md for the retraining protocol. Imports
are lazy so auditing data and operating the queue need no model dependencies.
"""

METHODS = ('lanzhou', 'aimagelab', 'universeg', 'smart', 'mofo')


def build_model(method, *, assets_dir=None, prior_checkpoint=None,
                expected_seed=None, records_sha256=None, support_manifest_sha256=None):
    from importlib import import_module
    if method not in METHODS:
        raise ValueError(f'Unknown external method: {method}')
    module = import_module(f'{__name__}.{method}')
    if method == 'mofo':
        return module.build_model(assets_dir=assets_dir, prior_checkpoint=prior_checkpoint,
                                  expected_seed=expected_seed, records_sha256=records_sha256,
                                  support_manifest_sha256=support_manifest_sha256)
    return module.build_model()
