import torch
import torch.nn.functional as F
from typing import Tuple
import torch.nn as nn

from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock as OriginalQwen2MoeSparseMoeBlock

#helper function t

def log_sigmoid(logits):
    return torch.clamp(F.logsigmoid(logits), min=-float('inf'), max=-1e-7)

def log1mexp(x):
    """
    Compute log(1 - exp(-|x|)) in a numerically stable way
    """
    x = -x.abs()
    x = torch.where(
        x > -0.693147180559945309417232121458,  # log(0.5)
        torch.log(-torch.expm1(x)),
        torch.log1p(-torch.exp(x))
    )

    return x

def torch_logaddexp_tfstyle(x1, x2):
    delta = torch.where(x1 == x2, 0., x1 - x2)
    return torch.maximum(x1, x2) + torch.nn.functional.softplus(-torch.abs(delta))



class ProbMoEQwen2MoeSparseMoeBlock(OriginalQwen2MoeSparseMoeBlock):
    def __init__(self, config):
        super().__init__(config)
        self.use_gradient_clipping = getattr(config, 'use_gradient_clipping', False)
    
    def log_pr_exactly_k(self, log_p, log_q, k):
        """
        compute the log probability of excatly k experts being selected using dp
        """

        batch_size, n_experts = log_p.shape
        log_p = log_p.float()

        # NEG_INF = -80 #float('-inf')
        NEG_INF = -300.0


        # Initialize the DP table
        state = torch.full((batch_size, k + 2), NEG_INF, device=log_p.device, dtype=log_p.dtype)
        state[:, 1] = 0.0  # P (sum = 0 from 0 experts) = 1

        all_states = [state.clone()]
        
        #loop from 1 to n_experts + 1 
        for i in range(1, n_experts + 1):
            new_state = torch.cat([
                torch.full([batch_size, 1], NEG_INF, device=log_p.device, dtype=log_p.dtype),
                torch_logaddexp_tfstyle(
                    state[:, :-1] + log_p[:, i-1:i], # select expert i-1
                    state[:, 1:] + log_q[:, i-1:i]  # do not select expert i-1
                )
            ], dim=1)
            
            state = new_state
            all_states.append(state.clone())
        return torch.stack(all_states, dim=1)  # (batch_size, n_experts + 1, k + 2)
    
    # def compute_marginals(self, router_logits, k):
    #     # print("Computing marginals with k =", k)
    #     # Clamp router logits to prevent extreme values
    #     router_logits_f32 = router_logits.float()

    #     log_p = log_sigmoid(router_logits_f32)
    #     log_p.requires_grad_(True)
    #     log_q = log1mexp(log_p.detach())
    #     a = self.log_pr_exactly_k(log_p, log_q, k)
    #     log_pr = a[:, -1, k + 1 : k+2]
    #     marginals = torch.autograd.grad(
    #         outputs=log_pr.sum(),
    #         inputs=log_p,
    #         create_graph=True,
    #     )[0]
    #     return marginals, a
    
    def compute_marginals(self, router_logits, k):
        # print("Computing marginals with k =", k)
        # Clamp router logits to prevent extreme values
        router_logits_f32 = router_logits.float()
        if not router_logits_f32.requires_grad:
            router_logits_f32.requires_grad_(True)

        with torch.enable_grad():
            log_p = log_sigmoid(router_logits_f32)
            log_q = log1mexp(log_p.detach())
            a = self.log_pr_exactly_k(log_p, log_q, k)
            log_pr = a[:, -1, k + 1 : k+2]
            marginals = torch.autograd.grad(
                outputs=log_pr.sum(),
                inputs=log_p,
                create_graph=True,
            )[0]
        return marginals, a
    
    def sample_k_subset(self, a, log_p, k):
       batch_size, n_experts = log_p.shape
       log_p = log_p.float()
       j = torch.full((batch_size,), k, device=a.device, dtype=torch.long) #if k =2, j will be [2,2,2,...]
       samples = []
    
       for i in range(n_experts, 0, -1):
           batch_idx = torch.arange(batch_size, device=a.device)
           
           # Handle batch dimension properly
           mask_j_zero = (j == 0)
           
           if mask_j_zero.all():
               # All batches need 0 more experts
               selection = torch.zeros(batch_size, device=a.device)
           else:
               # Get DP values
               p_vals = a[batch_idx, i - 1, j]
               z_vals = a[batch_idx, i, j + 1]
               
               p = (p_vals + log_p[:, i-1]) - z_vals
               
               # Force probability to 0 for batches that need no more experts
               p = torch.where(mask_j_zero, torch.tensor(-300.0, device=p.device), p)
               
               q = log1mexp(p)
               log_odds = p - q
               prob_select = torch.sigmoid(log_odds)
               selection = torch.bernoulli(prob_select)
               
               # Ensure no selection when j=0
            #    mask_j_zero = (j == 0)
               selection = torch.where(mask_j_zero, torch.zeros_like(selection), selection)
           
           # Update remaining count
           j = torch.where(selection > 0, j - 1, j)
           if (j < 0).any():
                print(f"j went negative: {j}, iteration i={i}")
           samples.append(selection)
       
       samples = torch.stack(samples[::-1], dim=1)
       return samples
   
    def simoe_routing(self, router_logits):
        """
        SIMoE-based stochastic routing for trainig
        """
        # print("Using SIMoE routing")
        router_logits = router_logits.float()
        log_p = log_sigmoid(router_logits)
        log_q = log1mexp(log_p.detach())  # stop gradient through log_q
        a = self.log_pr_exactly_k(log_p, log_q, self.top_k)
        samples = self.sample_k_subset(a, log_p, self.top_k).detach()
        #compute the exact marginals and discrete samples
        marginals, _ = self.compute_marginals(router_logits, self.top_k)
        
        #during training, use the softmax for the router weights
        softmax_probs = F.softmax(router_logits, dim=1, dtype=torch.float)  # (batch_size*sequence_length, num_experts)
        
        if self.use_gradient_clipping:
            sigmoid_probs = torch.sigmoid(router_logits)
            regularized_marginals = 0.99 * marginals + 0.01 * sigmoid_probs
            regularized_marginals = torch.clamp(regularized_marginals, min=1e-3, max=1-1e-3)
            marginals = regularized_marginals
        
        g_full = (samples - marginals).detach() + marginals  # Straight-through estimator
        
        w_full = g_full * softmax_probs  # (batch_size*sequence_length, num_experts)
        
        _, selected_experts = torch.topk(samples, self.top_k, dim=-1)
        
        router_weights = torch.gather(w_full, 1, selected_experts)
        
        router_weights = router_weights.to(router_logits.dtype)
        return router_weights, selected_experts
   
    def deterministic_routing(self, router_logits):
        """
        Deterministic top-k routing for inference
        """
        # print("Using deterministic routing")
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        return routing_weights, selected_experts



    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)
        
        if self.training:
            # stochastic
            routing_weights, selected_experts = self.simoe_routing(router_logits)
        else:
            # determinstic     
            routing_weights, selected_experts = self.deterministic_routing(router_logits)

        routing_weights = routing_weights.to(hidden_states.dtype)
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits