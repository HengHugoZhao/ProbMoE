"""
OLMoE sparse MoE block with circuit-based stochastic top-k routing.

Replaces the chain-DP SIMPLE routing with an SDD-compiled "exactly-k" circuit.
The circuit is built once at __init__ and reused for every forward pass.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from transformers.models.olmoe.modeling_olmoe import (
    OlmoeSparseMoeBlock as OriginalOlmoeSparseMoeBlock,
)

from .circuit_layer import CircuitLayer
from .create_simple_constraint import create_exactly_k


class ProbMoEOlmoeSparseMoeBlock(OriginalOlmoeSparseMoeBlock):
    """
    Drop-in replacement for OlmoeSparseMoeBlock.

    During training: samples top_k experts from the constrained distribution
    P(subset | exactly top_k selected) using an SDD, with a straight-through
    estimator backed by exact marginals.

    During eval: standard deterministic top-k softmax routing.

    """

    def __init__(self, config):
        super().__init__(config)
        self.use_gradient_clipping = getattr(config, "use_gradient_clipping", False)
        # Default to exact ancestral sampling: ProbMoE's straight-through
        # estimator requires E[samples] = marginals, which only holds when
        # samples are drawn from the constrained distribution P(S | exactly k).
        # `gumbel_topk_sample` samples from a Plackett-Luce surrogate instead
        # and biases the gradient (training loss inflates by ~2-3× in
        # practice). Set `use_exact_circuit_sampling=False` only for
        # inference / debugging.
        self.use_exact_circuit_sampling = getattr(
            config, "use_exact_circuit_sampling", True
        )

        # Build the "exactly top_k of num_experts" SDD once.
        # NOTE: create_exactly_k requires num_experts to be a power of 2.
        # If your config uses non-power-of-2 expert counts, see the note below.
        root = create_exactly_k(self.num_experts, self.top_k)[0][-1]
        self.circuit = CircuitLayer(root, device="cuda")

    # ----------------------------------------------------------------------
    # Routing
    # ----------------------------------------------------------------------

    def probmoe_routing(self, router_logits: torch.Tensor):
        """
        Stochastic constrained routing for training.

        router_logits : (B*T, num_experts)
        returns
            router_weights  : (B*T, top_k)
            selected_experts: (B*T, top_k)
        """
        router_logits = router_logits.float()
        log_p = F.logsigmoid(router_logits)  # log P(expert i = 1)

        # Straight-through samples + exact marginals through the SDD.
        # Both are (B*T, num_experts).
        samples, marginals = self.circuit(
            log_p, k=self.top_k, exact=self.use_exact_circuit_sampling
        )

        if self.use_gradient_clipping:
            sigmoid_probs = torch.sigmoid(router_logits)
            blended = 0.99 * marginals + 0.01 * sigmoid_probs
            blended = torch.clamp(blended, min=1e-3, max=1 - 1e-3)
            # Re-derive the ST tensor with the clipped marginals
            samples = (samples.detach() - blended).detach() + blended

        softmax_probs = F.softmax(router_logits, dim=1, dtype=torch.float)
        w_full = samples * softmax_probs

        # Pick which top_k positions are "on" for each token.
        # samples is 0/1 with exactly top_k ones per row, so topk recovers them.
        _, selected_experts = torch.topk(samples, self.top_k, dim=-1)
        router_weights = torch.gather(w_full, 1, selected_experts)

        return router_weights.to(router_logits.dtype), selected_experts

    def deterministic_routing(self, router_logits: torch.Tensor):
        """Standard deterministic top-k softmax routing (for eval)."""
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(
                dim=-1, keepdim=True
            )
        return routing_weights, selected_experts

    # ----------------------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------------------

    def forward(self, hidden_states: torch.Tensor):
        dtype = hidden_states.dtype
        device = hidden_states.device

        batch_size, sequence_length, hidden_dim = hidden_states.size()
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states)

        if self.training:
            routing_weights, selected_experts = self.probmoe_routing(router_logits)
        else:
            routing_weights, selected_experts = self.deterministic_routing(
                router_logits
            )

        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=dtype,
            device=device,
        )

        expert_mask = F.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = (
                expert_layer(current_state)
                * routing_weights[top_x, idx, None]
            )
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(dtype)
            )

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states, router_logits
