"""Small deterministic epoch-level learning-rate schedules for controlled comparisons."""
import math


def epoch_lr(initial_lr, epoch, epochs, schedule='constant', min_ratio=.1):
    if initial_lr <= 0 or epochs < 1 or not 1 <= epoch <= epochs:
        raise ValueError('Invalid learning-rate schedule position')
    if schedule == 'constant':
        return initial_lr
    if schedule != 'cosine' or not 0 <= min_ratio <= 1:
        raise ValueError('Unsupported schedule or minimum ratio')
    progress = (epoch - 1) / max(epochs - 1, 1)
    return initial_lr * (min_ratio + (1 - min_ratio) * .5 * (1 + math.cos(math.pi * progress)))
