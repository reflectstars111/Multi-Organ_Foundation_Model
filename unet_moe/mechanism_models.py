"""Q06/Q07 factories with a common Q05-refined initialization.

Q06 changes its objective only. Q07 constructs the complete nine-expert FiLM
model before removing branches, so the encoder, shared decoder, head, task
embedding and FiLM begin with identical values and identical RNG consumption.
The one-expert arm retains expert zero; this is fixed before any results are
examined, not selected from trained expert performance.
"""
from collections import OrderedDict

from torch import nn

from .refinement_models import create_model as create_refined_model


REGION_ARMS = ('control', 'routing')
TASK_ARMS = ('experts0', 'experts1', 'experts9')


def expert_count_for_arm(kind, arm):
    if kind == 'region' and arm in REGION_ARMS:
        return 9
    if kind == 'task_id' and arm in TASK_ARMS:
        return int(arm[len('experts'):])
    raise ValueError(f'Unknown Q06/Q07 architecture/arm: {kind}/{arm}')


class SharedFiLMDecoder(nn.Module):
    """The same shared-convolution-then-FiLM path, without any expert or router.

Unlike the historical ConditionalDecoder, this does not add a new condition
projection before the convolution. It uses the exact existing Q05 FiLM module
after the shared output, preserving the conditioning mechanism for Q07.
"""

    def __init__(self, original):
        super().__init__()
        self.shared = original.shared
        self.film = original.film
        self.experts = nn.ModuleList()
        self.top_k = 0

    def forward(self, x, context):
        output = self.shared(x)
        gamma, beta = self.film(context).chunk(2, dim=1)
        output = (1 + gamma[:, :, None, None]) * output + beta[:, :, None, None]
        return output, output.new_zeros(()), output.new_zeros(()), output.new_empty((0,))


def _retain_expert_zero(decoder):
    """Prune a fully initialized FiLM decoder without sampling any parameters.

The remaining router has one output, making softmax exactly one and its
selection gradient identically zero. Its existing balance is the constant one;
the Q07 objective explicitly excludes that inactive regularizer. The complete
single-expert design is compared, not a capacity-matched routing-only ablation.
"""
    decoder.experts = nn.ModuleList([decoder.experts[0]])
    router = decoder.router
    router.weight = nn.Parameter(router.weight[:1].detach().clone(), requires_grad=router.weight.requires_grad)
    if router.bias is not None:
        router.bias = nn.Parameter(router.bias[:1].detach().clone(), requires_grad=router.bias.requires_grad)
    router.out_features = 1
    decoder.top_k = 1
    return decoder


def create_model(kind, base=24, top_k=1, experts=9, arm='control'):
    """Create Q06 Region or Q07 Task-ID, always from a nine-expert reference.

``experts`` specifies the common initialization reference and must be nine.
The Q07 arm selects the actual expert count. This avoids accidentally changing
common initial parameters by constructing smaller networks from the same seed.
"""
    count = expert_count_for_arm(kind, arm)
    if type(experts) is not int or experts != 9 or type(top_k) is not int or top_k != 1:
        raise ValueError('Q06/Q07 require a common nine-expert top1 initialization')
    model = create_refined_model(kind, base=base, top_k=top_k, experts=9, arm='refined')
    if kind == 'task_id':
        if count == 0:
            model.decoders = nn.ModuleList(SharedFiLMDecoder(layer) for layer in model.decoders)
        elif count == 1:
            model.decoders = nn.ModuleList(_retain_expert_zero(layer) for layer in model.decoders)
    return model


def common_state_dict(model):
    """Project the state onto parameters shared across all expert-count arms.

Returned tensors are views of state_dict, not independent mutable copies.
Region paired factories also have exactly matching *full* states; this helper
is mainly intended for Q07's shared-initialization provenance.
"""
    return OrderedDict((key, value) for key, value in model.state_dict().items()
                       if '.experts.' not in key and '.router.' not in key)
