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
        
        self.k_max = getattr(config, 'k_max', self.top_k)
        self.k_min = getattr(config, 'k_min', 2)
    
    def log_pr_upto_k(self, log_p, log_q, k_max):
        """
        compute the log probability of less or equal to k experts being selected using dp
        """

        batch_size, n_experts = log_p.shape
        log_p = log_p.float()
        # NEG_INF = -80 #float('-inf')
        NEG_INF = -300.0
        
        # Initialize the DP table
        state = torch.full((batch_size, k_max + 2), NEG_INF, device=log_p.device, dtype=log_p.dtype)
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

    def sample_band_K(self, router_logits, a, k_min, k_max):
        #compute the probability of selecting between k_min and k_max experts
        B = a.size(0)
        
        logps = a[:, -1, (k_min + 1):(k_max + 2)]  # (batch_size, k_max - k_min + 1)
        probs = torch.softmax(logps, dim=-1)  # (batch_size, k_max - k_min + 1) P(S = k_min + idx | k_min ≤ S ≤ k_max)
        idx = torch.multinomial(probs, num_samples=1).squeeze(1) # [0, band_len-1]
        ks = idx + k_min  # sampled k values
        return ks
    
    def compute_marginals_band(self, router_logits, k_min, k_max):
        # print("Computing marginals with k =", k)
        # Clamp router logits to prevent extreme values
        router_logits = router_logits.float()
        # router_logits_f32 = torch.clamp(router_logits_f32, min=-50.0, max=500.0)# do
        
        log_p = log_sigmoid(router_logits)
        log_p.requires_grad_(True)
        # log_q = torch_log1mexp_tfstyle(log_p)
        log_q = log1mexp(log_p.detach())  # stop gradient through log_q
        # log_p = log_p.detach().requires_grad_(True)
        a = self.log_pr_upto_k(log_p, log_q, k_max)
        band_logps = a[:, -1, (k_min + 1):(k_max + 2)]  # (batch_size, k_max - k_min + 1)
        Log_P_band = torch.logsumexp(band_logps, dim=-1).sum()  # (batch_size, )
        marginals = torch.autograd.grad(
            outputs=Log_P_band,
            # outputs=log_pr,
            inputs=log_p,
            # grad_outputs=torch.ones_like(log_pr),
            create_graph=True,
        )[0]
        
        return marginals, a
    
    def sample_k_subset_dynamic(self, a, log_p, ks):
        batch_size, n_experts = log_p.shape
        log_p = log_p.float()
        j = ks.clone().to(device=a.device) # ks is of shape (batch_size,)
        samples = []
       
        for i in range(n_experts, 0, -1):
           batch_idx = torch.arange(batch_size, device=a.device)
           mask_j_zero = (j == 0)
           
           p_vals = a[batch_idx, i-1, j]
           z_vals = a[batch_idx, i, j+1]
           p = (p_vals + log_p[:, i-1]) - z_vals
           p = torch.where(mask_j_zero, torch.tensor(-300.0, device=p.device), p)
           
           q = log1mexp(p)
           log_odds = p - q
           prob_select = torch.sigmoid(log_odds)
           selection = torch.bernoulli(prob_select)
           selection = torch.where(mask_j_zero, torch.zeros_like(selection), selection)
           
           j = torch.where(selection > 0, j - 1, j)
           samples.append(selection)
           
        samples = torch.stack(samples[::-1], dim=1)  # (batch_size, n_experts)
        return samples
   
    def simoe_routing(self, router_logits):
        """
        simoe-based stochastic routing for trainig
        """
        # print("Using simoe routing")
        router_logits = router_logits.float()
        log_p = log_sigmoid(router_logits)
        log_q = log1mexp(log_p.detach())  # stop gradient through log_q
        a = self.log_pr_upto_k(log_p, log_q, self.k_max)
        
        ks = self.sample_band_K(router_logits, a, self.k_min, self.k_max)
        # print("Sampled ks:", ks)
        samples = self.sample_k_subset_dynamic(a, log_p, ks).detach()
        #compute the exact marginals and discrete samples
        marginals, _ = self.compute_marginals_band(router_logits, self.k_min, self.k_max)
        
        #during training, use the softmax for the router weights
        softmax_probs = F.softmax(router_logits, dim=1, dtype=torch.float)  # (batch_size*sequence_length, num_experts)
        
        if self.use_gradient_clipping:
            sigmoid_probs = torch.sigmoid(router_logits)
            regularized_marginals = 0.99 * marginals + 0.01 * sigmoid_probs
            regularized_marginals = torch.clamp(regularized_marginals, min=1e-3, max=1-1e-3)
            marginals = regularized_marginals
            
        g_full = (samples - marginals).detach() + marginals  # Straight-through estimator
        w_full = g_full * softmax_probs  # (batch_size, n_experts)
        values, selected_experts = torch.topk(samples, self.k_max, dim=-1)
        
        router_weights = torch.gather(w_full, 1, selected_experts)
        
        router_weights = router_weights * values  # zero out unselected experts
        
    
        #cast back to the input dtype
        router_weights = router_weights.to(router_logits.dtype)
        return router_weights, selected_experts
   
    def deterministic_routing(self, router_logits):
        """
        Deterministic top-k routing for inference
        """
        #deterministic top-k routing
        router_logits = router_logits.float() #[B,E]
        B, E = router_logits.shape
        
        #sort the logits by router logits decending
        sorted_logits, sorted_indices = torch.sort(router_logits, dim=1, descending=True)
        
        #Prefix sums of sorted logits
        cumsums = torch.cumsum(sorted_logits, dim=1)  #[B,E] calculates the sum of elements up to the current position along the specified dim
        
        #choose k in the range, map 
        #slice the range and take find the argmax per row
        candidate_scores = cumsums[:, (self.k_min - 1):self.k_max] #[B, k_max - k_min + 1]
        best_k_offset = torch.argmax(candidate_scores, dim=1)  #[B,]
        best_ks = best_k_offset + self.k_min  #[B,]
        
        #always take the  k_max for padding 
        topk_idx = sorted_indices[:, :self.k_max]  #(B, k_max)
        
        #softmax
        soft_routing = torch.softmax(router_logits, dim=1, dtype=torch.float)  # (batch_size*sequence_length, num_experts)
        batch_idx = torch.arange(B, device=router_logits.device).unsqueeze(1)
        routing_weights = soft_routing[batch_idx, topk_idx]  #(B, k_max)
        
        #mask out the unselected experts based on best_ks
        pos = torch.arange(self.k_max, device=router_logits.device).unsqueeze(0)  #(1, k_max)
        keep_mask = (pos < best_ks.unsqueeze(1)).float()  #[B, kmax], True for selected positions
        
        #zero out the unselected experts
        routing_weights = routing_weights * keep_mask.to(router_logits.dtype)

        if self.norm_topk_prob:
            sums = routing_weights.sum(dim=1, keepdim=True)  # [B, 1]
            # Avoid divide-by-zero if k_star == 0 (not expected with k_min>=1)
            routing_weights = torch.where(
                sums > 0,
                routing_weights / sums,
                routing_weights
            )
            
        #cast back to the input dtype
        routing_weights = routing_weights.to(router_logits.dtype)
        selected_experts = topk_idx  # [B, kmax] (first k* real, rest padded/zero-weighted)
        # print("Deterministic ks:", selected_experts)
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