"""Read-only audit of the frozen three-way paper-five experiment records."""
import argparse
from collections import Counter
import json

from unet_moe.baseline_suite import frozen_records
from unet_moe.region import DOMAINS, PAPER_ALIGNED_TASK_IDS, RegionDataset
from unet_moe.region_train import abdomen_positive_weights, domain_sampling_weights
from unet_moe.sampling import domain_draws


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    args = parser.parse_args()
    records, _, digest = frozen_records(args.data_root, 42, PAPER_ALIGNED_TASK_IDS, 'ct7_2')
    datasets = {split: RegionDataset([r for r in records if r['split'] == split], 256,
                                     PAPER_ALIGNED_TASK_IDS)
                for split in ('train', 'val', 'test')}
    weights, balance = abdomen_positive_weights(datasets['train'])
    train = datasets['train']
    sampling = domain_sampling_weights(train, weights, .5)
    shared_draws = domain_draws(train.groups, 500, 42, 1, weights=sampling)
    assert len(shared_draws) == len(DOMAINS) * 500
    for domain in DOMAINS:
        selected = [train.groups[i][0]['image'] for i in shared_draws
                    if train.groups[i][0]['dataset'] == domain]
        local = [rows for rows in train.groups if rows[0]['dataset'] == domain]
        local_dataset = RegionDataset([r for rows in local for r in rows], 256,
                                      PAPER_ALIGNED_TASK_IDS)
        local_weights = domain_sampling_weights(local_dataset, weights, .5)
        independent = domain_draws(local_dataset.groups, 500, 42, 1,
                                   domain=domain, weights=local_weights)
        assert Counter(selected) == Counter(local_dataset.groups[i][0]['image']
                                            for i in independent), domain
    counts = {split: dict(Counter(rows[0]['dataset'] for rows in ds.groups))
              for split, ds in datasets.items()}
    assert len(PAPER_ALIGNED_TASK_IDS) == 16
    assert tuple(counts[s]['AbdomenUS'] for s in ('train', 'val', 'test')) == (477, 156, 293)
    print(json.dumps(dict(record_sha256=digest, classes=16, experts=9,
                          images=counts, abdomen_positive_balance=balance,
                          same_epoch_one_draws=True), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
