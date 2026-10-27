"""MOFO paper-loss adaptation: nine domain unions, real CLIP and trained prior.

Visual weights are initialized from scratch.  The architecture itself is the
official CSWin-S [2,4,32,2], prompt decoder and dynamic convolution model.
"""
from pathlib import Path
import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F

from ..region import DOMAINS
from .vendor.mofo.network import MOFO
from .vendor.mofo.prior import pUNet


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prior_encode(prior, mask):
    """Only the original encoder is needed for the frozen prior constraint."""
    for block in (prior.down_tr32, prior.down_tr64, prior.down_tr128,
                  prior.down_tr256, prior.down_tr512):
        mask, feature = block(mask)
    return feature


class MOFOAdapter(nn.Module):
    def __init__(self, prompt_features, prior, *, image_size=512,
                 activation_checkpointing=True):
        super().__init__()
        if tuple(prompt_features.shape) != (len(DOMAINS), 512):
            raise ValueError('MOFO requires 9 real CLIP ViT-B/32 text vectors')
        if not torch.isfinite(prompt_features).all():
            raise ValueError('Non-finite text features')
        self.image_size = image_size
        self.network = MOFO(len(DOMAINS), 'word_embedding', image_size=image_size,
                            activation_checkpointing=activation_checkpointing)
        self.network.organ_embedding.copy_(prompt_features)
        self.prior = prior.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        self.prior.eval()
        return self

    def forward(self, images, domain_ids):
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, self.image_size, self.image_size):
            raise ValueError('Expected B,1,H,W grayscale input at frozen resolution')
        ids = torch.as_tensor(domain_ids, device=images.device, dtype=torch.long)
        if ids.shape != (len(images),) or ((ids < 0) | (ids >= len(DOMAINS))).any():
            raise ValueError('Require one valid domain ID per image')
        # The common loader provides [0,1] grayscale. No additional normalization
        # is imposed on a from-scratch backbone; RGB replication is lossless.
        all_logits, classification = self.network(images.expand(-1, 3, -1, -1))
        selected = all_logits[torch.arange(len(images), device=images.device), ids, None]
        return {'logits': selected, 'classification_logits': classification,
                'auxiliary_logits': []}

    def auxiliary_loss(self, output, target, domain_ids):
        """Paper Eq7: .5 CE + .2 mean Euclidean bottleneck distance.

        Domain-balanced query draws imply equal effective CE class weights.
        The prior uses the largest fully observed foreground/background ROI;
        unknown pixels are never encoded as anatomical background. Full masks
        keep the complete image. ROI-undefined/all-empty/all-foreground targets
        retain masked segmentation and domain-classification supervision.
        """
        ids = torch.as_tensor(domain_ids, device=target.device, dtype=torch.long)
        classification = F.cross_entropy(output['classification_logits'], ids)
        from .valid_region import valid_roi
        terms = []
        for index in range(len(target)):
            roi = valid_roi(target[index, 0])
            if roi is None:
                continue
            top, left, bottom, right = roi
            truth = target[index:index+1, :, top:bottom, left:right]
            if (truth < 0).any():
                raise ValueError('Prior ROI contains unknown labels')
            if not truth.gt(0).any() or not truth.eq(0).any():
                continue
            probability = output['logits'][index:index+1, :, top:bottom, left:right].sigmoid()
            if truth.shape[-2:] != (self.image_size, self.image_size):
                probability = F.interpolate(probability, size=self.image_size, mode='bilinear', align_corners=False)
                truth = F.interpolate(truth, size=self.image_size, mode='nearest')
            prediction = prior_encode(self.prior, probability)
            with torch.no_grad():
                reference = prior_encode(self.prior, truth)
            terms.append(torch.linalg.vector_norm(prediction - reference, dim=1).sum())
        # Sum over eligible images / entire batch keeps accumulation exact.
        prior_loss = sum(terms) / len(target) if terms else output['logits'].sum() * 0
        return .5 * classification + .2 * prior_loss


def build_model(assets_dir, prior_checkpoint, *, image_size=512,
                activation_checkpointing=True, expected_seed=None,
                records_sha256=None, support_manifest_sha256=None,
                expected_prior_epochs=80):
    """Fail closed unless text provenance and completed train-only prior exist."""
    assets_dir = Path(assets_dir)
    prompt_path = assets_dir / 'text_prompts.pt'
    manifest = json.loads((assets_dir / 'text_prompts.json').read_text())
    if manifest['domains'] != DOMAINS or manifest['sha256'] != sha256(prompt_path):
        raise ValueError('CLIP task order or asset SHA mismatch')
    if manifest['model_id'] != 'openai/clip-vit-base-patch32' or not manifest.get('revision'):
        raise ValueError('CLIP provenance absent or wrong encoder')
    prompt = torch.load(prompt_path, map_location='cpu', weights_only=True)
    state = torch.load(prior_checkpoint, map_location='cpu', weights_only=True)
    protocol = state.get('protocol', {})
    if not state.get('complete') or state.get('epoch') != protocol.get('epochs'):
        raise ValueError('Prior checkpoint is not a completed auxiliary stage')
    if protocol.get('epochs') != expected_prior_epochs:
        raise ValueError('Prior training budget differs from frozen auxiliary stage')
    if protocol.get('split') != 'train' or protocol.get('target_policy') != 'fully_observed_union_roi':
        raise ValueError('Prior must use fully observed train-only union ROIs')
    if protocol.get('domains') != DOMAINS or protocol.get('size') != image_size:
        raise ValueError('Prior domain/size mismatch')
    for name, expected in [('seed', expected_seed), ('records_sha256', records_sha256),
                           ('support_manifest_sha256', support_manifest_sha256)]:
        if expected is not None and protocol.get(name) != expected:
            raise ValueError(f'Prior {name} does not match current training protocol')
    prior = pUNet(n_class=1)
    prior.load_state_dict(state['model'], strict=True)
    model = MOFOAdapter(prompt, prior, image_size=image_size,
                        activation_checkpointing=activation_checkpointing)
    model.asset_provenance = {'prompt_sha256': manifest['sha256'],
                              'prior_sha256': sha256(prior_checkpoint),
                              'prior_protocol': protocol}
    return model
