"""Generate immutable nine-domain prompts with the official pretrained CLIP."""
import argparse
import json
from pathlib import Path

import torch

from ..region import DOMAINS
from .mofo import sha256

MODEL_ID = 'openai/clip-vit-base-patch32'
REVISION = '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
WEIGHTS_SHA256 = 'a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f'
PROMPTS = [
    'An ultrasound image of a breast lesion.',
    'An ultrasound image of an ovarian lesion.',
    'An ultrasound image of a thyroid nodule.',
    'An ultrasound image of the liver and a liver mass.',
    'An ultrasound image of a muscle aponeurosis.',
    'An ultrasound image of a fetal head contour.',
    'An ultrasound image of a carotid artery.',
    'An ultrasound image of the left ventricle cavity, myocardium and left atrium.',
    'A medical image of the liver, kidneys, blood vessels, gallbladder and spleen.',
]


def generate(output, local_model=None):
    import transformers
    from transformers import CLIPTextConfig, CLIPTextModelWithProjection, CLIPTokenizer
    from huggingface_hub import hf_hub_download
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    kwargs = {'local_files_only': True} if local_model else {'revision': REVISION, 'cache_dir': str(output / 'clip')}
    source = str(local_model) if local_model else MODEL_ID
    tokenizer = CLIPTokenizer.from_pretrained(source, **kwargs)
    configuration_hashes = {}
    for filename in ('config.json', 'tokenizer_config.json', 'special_tokens_map.json',
                     'vocab.json', 'merges.txt', 'tokenizer.json'):
        path = Path(local_model) / filename if local_model else Path(hf_hub_download(
            MODEL_ID, filename, revision=REVISION, cache_dir=str(output / 'clip')))
        configuration_hashes[filename] = sha256(path)
    weights = Path(local_model) / 'pytorch_model.bin' if local_model else Path(hf_hub_download(
        MODEL_ID, 'pytorch_model.bin', revision=REVISION, cache_dir=str(output / 'clip')))
    weight_hash = sha256(weights)
    if weight_hash != WEIGHTS_SHA256:
        raise ValueError('Official pinned CLIP checkpoint SHA256 does not match')
    # Explicit weights-only loading is compatible with the frozen torch 2.5
    # environment. It never unpickles arbitrary third-party Python objects.
    state = torch.load(weights, map_location='cpu', weights_only=True)
    text_state = {key: value for key, value in state.items()
                  if key.startswith(('text_model.', 'text_projection.'))
                  and key != 'text_model.embeddings.position_ids'}
    model = CLIPTextModelWithProjection(CLIPTextConfig.from_pretrained(source, **kwargs))
    model.load_state_dict(text_state, strict=True)
    model.eval().requires_grad_(False)
    tokens = tokenizer(PROMPTS, padding=True, truncation=False, return_tensors='pt')
    with torch.inference_mode():
        # Preserve CLIP encode_text's raw projected embeddings (not L2 normalized).
        features = model(**tokens).text_embeds.float().cpu()
    if features.shape != (9, 512) or not torch.isfinite(features).all():
        raise RuntimeError('Unexpected CLIP encoding')
    path = output / 'text_prompts.pt'
    torch.save(features, path)
    manifest = {'model_id': MODEL_ID, 'revision': REVISION, 'domains': DOMAINS,
                'prompts': PROMPTS, 'shape': [9, 512], 'normalized': False,
                'sha256': sha256(path), 'source': source, 'weights_sha256': weight_hash,
                'configuration_sha256': configuration_hashes,
                'environment': {'torch': str(torch.__version__), 'transformers': transformers.__version__},
                'note': 'Official pretrained CLIP text encoder only; visual network trained from scratch.'}
    (output / 'text_prompts.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--local-model', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    generate(args.output, args.local_model)


if __name__ == '__main__':
    main()
