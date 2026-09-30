"""Deterministic per-domain image draws shared by the three model protocols."""
import random

from .region import DOMAINS


def domain_draws(groups, quota, seed, epoch, domain=None, weights=None):
    if weights is not None and len(weights) != len(groups):
        raise ValueError('Sampling weights must match image groups')
    candidates = {name: [] for name in DOMAINS}
    for index, rows in enumerate(groups):
        candidates[rows[0]['dataset']].append(index)
    names = [domain] if domain is not None else DOMAINS
    draws = {}
    for name in names:
        rng = random.Random(seed + 100003 * epoch + 997 * DOMAINS.index(name))
        if weights is None:
            draws[name] = [rng.choice(candidates[name]) for _ in range(quota)]
        else:
            draws[name] = rng.choices(candidates[name],
                                      weights=[weights[i] for i in candidates[name]], k=quota)
    if domain is not None:
        return draws[domain]
    combined = [i for name in DOMAINS for i in draws[name]]
    random.Random(seed + 100003 * epoch + 7919).shuffle(combined)
    return combined
