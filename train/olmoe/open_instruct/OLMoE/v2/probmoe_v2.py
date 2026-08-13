"""ProbMoE v2: OLMoE sparse MoE block with SIMPLE routing on a compiled circuit.

Same routing semantics as probmoe_v1/probmoe_v1.py:

    raw router logit -> sigmoid -> independent Bernoullis conditioned on
    exactly top_k successes.  Training uses a straight-through estimator
    (exact k-hot sample forward, exact conditional marginals backward)
    multiplied by softmax(router_logits); inference uses deterministic
    softmax top-k.

The difference is how sampling and marginals are computed: v1 runs an O(n*k)
dynamic program per token plus an autograd probe for the marginals; v2 runs
one upward pass of a compiled exactly-k probabilistic circuit
(simple_v2.Simple_Layer) that serves ancestral sampling and top-down
marginals in the same pass.

Gradient semantics follow the SIMPLE paper (arXiv:2210.01941): the marginal
Jacobian equals the covariance of the conditioned distribution.  The circuit's
top-down flows realize the paper's inner derivative (the complement weight is
a constant circuit parameter there), while log(1 - p) stays connected to the
router logits so the outer derivative is exact -- see the note in
simple_v2.Simple_Layer.upward and probmoe_v1/toy_detach_scope.py.

Simple_Layer is an nn.Module whose circuit index tables are non-persistent
integer buffers: they follow the block through .to()/DDP/FSDP placement, add
no parameters, and stay out of state_dict, so checkpoints remain plain OLMoE.
"""

import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from transformers.models.olmoe.modeling_olmoe import (
    OlmoeSparseMoeBlock as OriginalOlmoeSparseMoeBlock,
)

from simple_v2 import Simple_Layer


def log_sigmoid(logits):
    return torch.clamp(F.logsigmoid(logits), min=-float('inf'), max=-1e-7)


class ProbMoEOlmoeSparseMoeBlock(OriginalOlmoeSparseMoeBlock):
    def __init__(self, config):
        super().__init__(config)
        self.use_gradient_clipping = getattr(config, 'use_gradient_clipping', False)
        # One circuit per (num_experts, top_k); loaded from (or built into)
        # exactly_k_{n}_{k}.pkl next to simple_v2.py.
        self.simple = Simple_Layer(self.num_experts, self.top_k)

    def probmoe_routing(self, router_logits):
        """
        ProbMoE stochastic routing for training (circuit-based SIMPLE).
        """
        router_logits = router_logits.float()
        log_p = log_sigmoid(router_logits)

        # One upward pass serves exact ancestral sampling and marginals.
        data, theta = self.simple.upward(log_p)
        samples = self.simple.sample_subset(theta)  # no_grad, exact k-hot
        marginals = self.simple.marginals_from_upward(data, theta).exp().permute(1, 0)

        # during training, use the softmax for the router weights
        softmax_probs = F.softmax(router_logits, dim=1, dtype=torch.float)

        g_full = (samples - marginals).detach() + marginals  # Straight-through estimator

        w_full = g_full * softmax_probs  # Combine with softmax probabilities

        # sparse compute
        _, selected_experts = torch.topk(samples, self.top_k, dim=-1)

        router_weights = torch.gather(w_full, 1, selected_experts)

        return router_weights, selected_experts

    def deterministic_routing(self, router_logits):
        """
        Deterministic top-k routing for inference
        """
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)  # (batch_size*sequence_length, num_experts)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        return routing_weights, selected_experts

    def forward(self, hidden_states: torch.Tensor):
        """
        custom forward function for OLMoE sparse block
        """
        dtype = hidden_states.dtype
        device = hidden_states.device

        batch_size, sequence_length, hidden_dim = hidden_states.size()
        hidden_states = hidden_states.view(-1, hidden_dim)  # (batch_size*sequence_length, hidden_dim)

        # router logits (batch_size*sequence_length, num_experts)
        router_logits = self.gate(hidden_states)

        if self.training:
            routing_weights, selected_experts = self.probmoe_routing(router_logits)
        else:
            # Use deterministic top-k routing
            routing_weights, selected_experts = self.deterministic_routing(router_logits)

        routing_weights = routing_weights.to(hidden_states.dtype)

        # Initialize output
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=dtype, device=device
        )

        # create expert mask for routing
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # process through each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


# Alias so callers can be explicit about the version they import.
ProbMoEOlmoeSparseMoeBlockV2 = ProbMoEOlmoeSparseMoeBlock
