import os
import sys
import time
import torch
import torch._dynamo as dynamo
import logging

# torch.set_float32_matmul_precision("high")
from typing import Dict, List, Tuple
import pickle

# Make `node` (needed both for the import below and for unpickling circuits)
# importable no matter where simple_v2 is imported from.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from node import *

@torch.compile(fullgraph=True)
def log1mexp(x):
    x = -x.abs()
    x = torch.where(
        x > -0.6931471805599453094,
        torch.log(-torch.expm1(x)),
        torch.log1p(-torch.exp(x)),
    )
    return x

@torch.compile(fullgraph=True)
def levelwiseSL(levels: List[torch.Tensor], idx2primesub: torch.Tensor, data: torch.Tensor, theta: torch.Tensor):
    for level in levels:
        theta[level] = data[idx2primesub[level]].sum(-2)
        data[level] = theta[level].logsumexp(-2)
        theta[level] -= data[level].unsqueeze(1)
    return data[levels[-1]]

@torch.compile(fullgraph=True)
def levelwiseMars(levels: List[torch.Tensor], idx2primesub: torch.Tensor, data: torch.Tensor, theta: torch.Tensor, parents: torch.Tensor):
    for level in reversed(levels):
        data[level] = (theta[parents[level].unbind(-1)] + data[parents[level].unbind(-1)[0]]).logsumexp(-2)


def levelOrder(beta):
    seen = dict()
    nodes = [beta]
    level = []
    answer =[]
    result = [[beta]]
    
    while len(nodes) !=0:
        for a in nodes:
            if not a.is_decomposition():
                continue
            for element in a.elements:
                for e in element:
                    if not e.is_decomposition():
                        continue
                    if seen.get(e) != None:
                        continue
                    seen[e] = True
                    level.append(e)
        nodes = level
        for i in level:
            answer.append(i)
        level = []
        answer = list(dict.fromkeys(answer))
        result.append(answer)
        answer = []
    
    return result[:-1]

class Simple_Layer(torch.nn.Module):
    """Exactly-k circuit layer.

    The circuit's index tensors are registered as NON-PERSISTENT buffers:
    they follow the module through .to()/.cuda()/DDP placement like weights,
    but stay out of state_dict, so checkpoints remain plain OLMoE. They are
    all integer tensors, so dtype casts (.half()/.bfloat16()) never touch
    them and no gradients ever flow into them.
    """

    def __init__(self, n_experts: int = 4, k: int = 2, device: str = "cuda"):
        super().__init__()
        self.n_experts = n_experts
        self.k = k
        self.device = device

        circuit_path = os.path.join(HERE, f"exactly_k_{n_experts}_{k}.pkl")
        if not os.path.exists(circuit_path):
            from create_simple_constaint import create_exactly_k
            alpha = create_exactly_k(n_experts, k)[0][-1]
            with open(circuit_path, "wb") as f:
                pickle.dump(alpha, f)
            print(f"Built and saved {circuit_path}")
        with open(circuit_path, "rb") as f:
            beta = pickle.load(f)

        max_elements = 0
        
        for node in beta.positive_iter():
            if node.is_decomposition():
                max_elements = max(max_elements, len(node.elements))
        
        # print(f"max_elements: {max_elements}")
        
        levels_nodes = levelOrder(beta)
        # print(f"number of levels: {len(levels_nodes)}")
        # print(levels_nodes)
        
        nodes = [node for node in beta.positive_iter()]
        nodes = list(dict.fromkeys(nodes))        
        
        id = 0
        for e in nodes:
            e.id = id
            id += 1
        # Sentinel index / workspace height. NOT named `id`: DeepSpeed ZeRO-3
        # stamps its own `module.id` counter onto every submodule it walks,
        # which would silently overwrite this and mis-size `data`/`theta`.
        self.num_nodes = id

        
        from collections import defaultdict
        parents_dict = defaultdict(list)
        for node in beta.positive_iter():
            if node.is_decomposition():
                for i, (p,s) in enumerate(node.elements):
                    parents_dict[p.id] += [[node.id,i]]
                    parents_dict[s.id] += [[node.id,i]]
        
        # print(f"parents_dict: {parents_dict}")
                    
        #backward pass
        max_parents = 0
        for p in parents_dict.values():
            max_parents = max(max_parents, len(p))

        parents = torch.empty((id, max_parents, 2), dtype=torch.int, device=self.device).fill_(id)#.to(device)
        # print(f"parents: {parents.shape}")
        # print(parents)
        for k,v in parents_dict.items():
            parents[k] = torch.tensor(v + [[id, 0]]*(max_parents-len(v)), dtype=torch.int, device=self.device)
        self.register_buffer("parents", parents, persistent=False)
        
        # print(f"parents: {self.parents.shape}")
        # print(self.parents)
        
        levels = []
        for level in levels_nodes:
            # print(f"current level: {level}")
            levels.append(torch.tensor([l.id for l in level], dtype=torch.int, device=self.device))
        levels.reverse()
        # Lists cannot be registered as buffers directly: one buffer per
        # level; the `levels` property rebuilds the (bottom-up) list.
        self.num_levels = len(levels)
        for i, level in enumerate(levels):
            self.register_buffer(f"level_{i}", level, persistent=False)
        # print(levels)
        
        # print("number of levels: ", len(levels))
        
        
        true_indicies = torch.tensor(
            [
                node.id for node in nodes if node.is_true()
            ],
            dtype=torch.int,
            device=self.device
        )
        self.register_buffer("true_indicies", true_indicies, persistent=False)
        
        #literal indicies
        literal_indicies = torch.tensor(
            [
                [node.id, node.literal] for node in nodes if node.is_literal()
            ],
            dtype=torch.int,
            device=self.device
        )
        literal_indicies, literal_mask = literal_indicies.unbind(-1)
        literal_mask = literal_mask.abs() -1, (literal_mask > 0).long()
        self.register_buffer("literal_indicies", literal_indicies, persistent=False)
        # Tuples cannot be registered either; the `literal_mask` property
        # reassembles (variable index, is-positive) from the two buffers.
        self.register_buffer("literal_mask_var", literal_mask[0], persistent=False)
        self.register_buffer("literal_mask_pos", literal_mask[1], persistent=False)

        order = self.literal_mask[0][self.literal_mask[1].bool()].sort().indices
        pos_literals = self.literal_indicies[self.literal_mask[1].bool()][order]
        self.register_buffer("pos_literals", pos_literals, persistent=False)
        
        #Map nodes to their prime/subs
        idx2primesub = torch.zeros((id, max_elements, 2), dtype=torch.int, device=self.device)
        for node in nodes:
            if node.is_decomposition():
                tmp = torch.tensor([[p.id, s.id] for p, s in node.elements], dtype=torch.int)
                idx2primesub[node.id] = torch.nn.functional.pad(tmp, (0, 0, 0, max_elements - len(tmp)), value=id)
        self.register_buffer("idx2primesub", idx2primesub, persistent=False)

    @property
    def levels(self):
        return [getattr(self, f"level_{i}") for i in range(self.num_levels)]

    @property
    def literal_mask(self):
        return self.literal_mask_var, self.literal_mask_pos

    def upward(self, log_probs):
        """Run the circuit upward once and return its runtime workspaces."""
        # log(1 - p) must stay CONNECTED to log_probs. The top-down flow pass
        # already realizes the paper's inner derivative (complement treated as
        # a constant circuit parameter when extracting marginals), but the
        # outer SIMPLE derivative d(marginals)/d(logits) has to flow through
        # both log p and log(1 - p). A global detach here rescales the
        # marginal Jacobian by sigmoid(-logit) columnwise and breaks shift
        # invariance (see probmoe_v1/toy_detach_scope.py).
        literal_weights = torch.stack(
            (
                log1mexp(-log_probs),
                log_probs
                ),
            dim=-1
            ).permute(1, 2, 0)

        data = torch.empty(self.num_nodes+1, log_probs.size(0), device=log_probs.device, dtype=log_probs.dtype)
        theta = torch.zeros(self.num_nodes+1, self.idx2primesub.size(1), log_probs.size(0), device=log_probs.device, dtype=log_probs.dtype)
        
        data[self.true_indicies] = 0
        data[self.num_nodes] = -float(1000)
        data[self.literal_indicies] = literal_weights[self.literal_mask[0], self.literal_mask[1]]
        
        levelwiseSL(self.levels, self.idx2primesub, data, theta)
        return data, theta

    def marginals_from_upward(self, data, theta):
        """Reuse an upward pass for the top-down marginal computation."""
        data[self.levels[-1]] -= data[self.levels[-1]] 
        levelwiseMars([self.literal_indicies] + self.levels[:-1], self.idx2primesub, data, theta, self.parents)
        return data[self.pos_literals]

    def log_pr(self, log_probs):
        data, theta = self.upward(log_probs)
        return self.marginals_from_upward(data, theta)

    @torch.no_grad()
    def sample_subset(self, theta):
        """Sample an exact k-hot mask using the existing circuit upward pass."""
        batch_size = theta.size(-1)
        device = theta.device

        # A node is active when an active parent selected an element containing
        # that node. Counts (rather than booleans) handle shared DAG children.
        active_count = torch.zeros(
            self.num_nodes + 1,
            batch_size,
            dtype=torch.long,
            device=device,
        )
        active_count[self.levels[-1]] = 1

        # self.levels is bottom-up; ancestral sampling must run root-to-leaves.
        for level in reversed(self.levels):
            parent_active = active_count[level] > 0
            element_logits = theta[level].detach()

            # An inactive node's random choice is discarded. Give those rows a
            # harmless finite distribution so an unreachable zero-mass node
            # cannot make torch.multinomial fail because of NaNs.
            element_logits = torch.where(
                parent_active.unsqueeze(1),
                element_logits,
                torch.zeros_like(element_logits),
            )

            # Padded elements point to the sentinel node. Mask them explicitly
            # so they have exactly zero sampling probability.
            element_table = self.idx2primesub[level]
            valid_element = element_table[..., 0] != self.num_nodes
            element_logits = element_logits.masked_fill(
                ~valid_element.unsqueeze(-1),
                -torch.inf,
            )
            element_probabilities = element_logits.softmax(dim=-2)

            num_nodes, num_elements, _ = element_probabilities.shape
            flat_probabilities = element_probabilities.permute(0, 2, 1).reshape(
                num_nodes * batch_size,
                num_elements,
            )
            selected_element = torch.multinomial(
                flat_probabilities,
                num_samples=1,
            ).reshape(num_nodes, batch_size)

            # Gather [prime_id, sub_id] for the sampled element of every node.
            gather_index = selected_element.unsqueeze(-1).expand(-1, -1, 2)
            selected_children = element_table.gather(1, gather_index)

            # Choices made for inactive nodes are ignored.
            batch_index = torch.arange(device=device, end=batch_size)
            batch_index = batch_index.unsqueeze(0).expand(num_nodes, -1).reshape(-1)
            activation_vote = parent_active.long().reshape(-1)

            for child_position in range(2):
                child_id = selected_children[:, :, child_position].reshape(-1).long()
                active_count.index_put_(
                    (child_id, batch_index),
                    activation_vote,
                    accumulate=True,
                )

        return (active_count[self.pos_literals] > 0).T.to(theta.dtype)

    def sample(self, theta):
        return self.sample_subset(theta)

    def forward(self, log_probs):
        # One upward pass supplies both exact ancestral sampling and marginals.
        data, theta = self.upward(log_probs)
        samples = self.sample(theta)
        marginals = self.marginals_from_upward(data, theta).exp().permute(1, 0)
        return (samples - marginals).detach() + marginals
  
        
if __name__ == "__main__":
    layer = Simple_Layer()        
