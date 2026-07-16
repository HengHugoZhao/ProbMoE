import torch
import torch.nn.functional as F
from typing import Tuple

from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock as OriginalOlmoeSparseMoeBlock
import random

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

class ProbMoEOlmoeSparseMoeBlock(OriginalOlmoeSparseMoeBlock):
    def __init__(self, config):
        super().__init__(config) # initialize the original OlmoeSparseMoeBlock
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
    
    def compute_marginals(self, router_logits, k):
        # print("Computing marginals with k =", k)
        # Clamp router logits to prevent extreme values
        router_logits_f32 = router_logits.float()
        
        # router_logits_f32 = torch.clamp(router_logits_f32, min=-50.0, max=500.0)# do
        log_p = log_sigmoid(router_logits_f32)
        log_p = log_p.requires_grad_(True)
        # log_q = torch_log1mexp_tfstyle(log_p)
        log_q = log1mexp(log_p.detach())  # stop gradient through log_q
        # log_p = log_p.detach().requires_grad_(True)
        a = self.log_pr_exactly_k(log_p, log_q, k)
        log_pr = a[:, -1, k + 1 : k+2]

        marginals = torch.autograd.grad(
            outputs=log_pr.sum(),
            # outputs=log_pr,
            inputs=log_p,
            # grad_outputs=torch.ones_like(log_pr),
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
    
    def probmoe_routing(self, router_logits):
        """
        ProbMoE stochastic routing for training.
        """
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
        
        w_full = g_full * softmax_probs  # Combine with softmax probabilities
        
        # sparse compute
        _, selected_experts = torch.topk(samples, self.top_k, dim=-1)
        
        router_weights = torch.gather(w_full, 1, selected_experts)
        
        #cast back to the input dtype
        router_weights = router_weights.to(router_logits.dtype)
        return router_weights, selected_experts
    
    
    def deterministic_routing(self, router_logits):
        """
        Deterministic top-k routing for inference
        """
        #deterministic top-k routing
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)  # (batch_size*sequence_length, num_experts)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        #cast back to the input dtype

        return routing_weights, selected_experts

    def forward(self, hidden_states: torch.Tensor):
        """
        cumstom forward function for OLMoE sparse block
        """
        dtype = hidden_states.dtype
        device = hidden_states.device


        batch_size, sequence_length, hidden_dim = hidden_states.size()
        hidden_states = hidden_states.view(-1, hidden_dim)  # (batch_size*sequence_length, hidden_dim)

        #router logits (batch_size*squence_length, num_experts)
        router_logits = self.gate(hidden_states)

        if self.training:
            routing_weights, selected_experts = self.probmoe_routing(router_logits)

        else:
            # Use deterministic top-k routing
            routing_weights, selected_experts = self.deterministic_routing(router_logits)

        routing_weights = routing_weights.to(hidden_states.dtype)
        
        #Initialize output
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=dtype, device=device
        )

        #crate expert mask for routing
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        #process through each expert
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
