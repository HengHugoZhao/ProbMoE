"""
Probabilistic-circuit (SDD) layer for computing exact marginals and samples
under a logical constraint (e.g. "exactly k of n variables are true").

This is a refactor of simple.py's `Layer` that:
  - takes the SDD root node directly (no pickle file)
  - exposes `log_pr` (marginals) and `sample` (Gumbel top-k approx)
  - adds `exact_sample` for true ancestral sampling through the circuit
"""

from __future__ import annotations

from collections import defaultdict
from typing import List

import torch

from .node import Node


# ---------------------------------------------------------------------------
# Compiled inner kernels (same as simple.py)
# ---------------------------------------------------------------------------

@torch.compile(fullgraph=True)
def _levelwise_sl(
    levels: List[torch.Tensor],
    idx2primesub: torch.Tensor,
    data: torch.Tensor,
    theta: torch.Tensor,
):
    """Upward pass: log-sum-exp semiring. Caches normalized child log-probs in theta."""
    for level in levels:
        theta[level] = data[idx2primesub[level]].sum(-2)
        data[level] = theta[level].logsumexp(-2)
        theta[level] -= data[level].unsqueeze(1)
    return data[levels[-1]]


@torch.compile(fullgraph=True)
def _levelwise_mars(
    levels: List[torch.Tensor],
    idx2primesub: torch.Tensor,
    data: torch.Tensor,
    theta: torch.Tensor,
    parents: torch.Tensor,
):
    """Downward pass: propagate marginals from root to leaves."""
    for level in reversed(levels):
        data[level] = (
            theta[parents[level].unbind(-1)]
            + data[parents[level].unbind(-1)[0]]
        ).logsumexp(-2)


def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable log(1 - exp(-|x|))."""
    x = -x.abs()
    return torch.where(
        x > -0.6931471805599453094,
        torch.log(-torch.expm1(x)),
        torch.log1p(-torch.exp(x)),
    )


class _CircuitLogZ(torch.autograd.Function):
    """log Z(constraint) as a single autograd op.

    forward  : SDD upward pass (in-place, fast). Returns log Z.
    backward : SDD downward pass (in-place, fast). Returns marginals.

    First-order only: backward is decorated with ``@once_differentiable`` so
    autograd will not try to differentiate the in-place downward kernel.
    Use the autograd-traced ``CircuitLayer.log_pr`` (functional `_log_Z` +
    ``autograd.grad``) when double-backward is required (SIMOE training).
    """

    @staticmethod
    def forward(ctx, log_probs: torch.Tensor, layer):
        # log_probs: (n_vars, batch). log(1-p) is treated as a constant
        # (the marginal-trick stop-grad) by being computed inside the Function.
        lit_weights = torch.stack(
            (log1mexp(-log_probs.detach()), log_probs.detach()), dim=-1
        ).permute(0, 2, 1)  # (n_vars, 2, batch)
        batch = log_probs.size(-1)
        data = torch.empty(layer.id + 1, batch, device=layer.device, dtype=log_probs.dtype)
        theta = torch.zeros(
            layer.id + 1, layer.idx2primesub.size(1), batch,
            device=layer.device, dtype=log_probs.dtype,
        )
        data[layer.true_indices] = 0.0
        data[layer.id] = -1000.0
        data[layer.literal_indices] = lit_weights[
            layer.literal_mask[0], layer.literal_mask[1]
        ]
        _levelwise_sl(layer.levels, layer.idx2primesub, data, theta)
        log_Z = data[layer.levels[-1]].squeeze(0).clone()  # (batch,)

        ctx.save_for_backward(data, theta)
        ctx.layer = layer
        return log_Z

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_log_Z: torch.Tensor):
        data, theta = ctx.saved_tensors
        layer = ctx.layer
        # save_for_backward returns shared tensors; clone before in-place mutation.
        data = data.clone()
        data[layer.levels[-1]] -= data[layer.levels[-1]]   # zero root → conditional
        _levelwise_mars(
            [layer.literal_indices] + layer.levels[:-1],
            layer.idx2primesub, data, theta, layer.parents,
        )
        marginals = data[layer.pos_literals].exp()  # (n_vars, batch)
        grad_log_probs = grad_log_Z.unsqueeze(0) * marginals
        return grad_log_probs, None  # no grad for `layer`


def _level_order(beta: Node) -> List[List[Node]]:
    """BFS by depth over decomposition nodes, deduplicating shared subgraphs."""
    seen: dict = {}
    nodes = [beta]
    result = [[beta]]
    while nodes:
        level = []
        for a in nodes:
            if not a.is_decomposition():
                continue
            for prime, sub in a.elements:
                for child in (prime, sub):
                    if not child.is_decomposition():
                        continue
                    if child in seen:
                        continue
                    seen[child] = True
                    level.append(child)
        if not level:
            break
        result.append(list(dict.fromkeys(level)))
        nodes = level
    return result


# ---------------------------------------------------------------------------
# CircuitLayer
# ---------------------------------------------------------------------------

class CircuitLayer:
    """
    A compiled SDD layer.

    Parameters
    ----------
    beta : Node
        The root of the SDD (e.g. from `create_exactly_k(n, k)[0][-1]`).
    device : str
        CUDA device. CPU is not supported by the compiled kernels.
    """

    def __init__(self, beta: Node, device: str = "cuda"):
        self.device = device

        # ---- Walk the circuit, assign sequential IDs ----
        nodes = list(dict.fromkeys(beta.positive_iter()))
        for new_id, n in enumerate(nodes):
            n.id = new_id
        self.id = len(nodes)  # sentinel index = self.id

        # ---- Find the widest decomposition (for padding idx2primesub) ----
        max_elements = max(
            (len(n.elements) for n in nodes if n.is_decomposition()),
            default=1,
        )

        # ---- Collect parents for the backward (marginal) pass ----
        parents_dict: dict = defaultdict(list)
        for n in nodes:
            if n.is_decomposition():
                for i, (p, s) in enumerate(n.elements):
                    parents_dict[p.id].append([n.id, i])
                    parents_dict[s.id].append([n.id, i])
        max_parents = max((len(v) for v in parents_dict.values()), default=1)

        parents = torch.full(
            (self.id, max_parents, 2),
            self.id,
            dtype=torch.int,
            device=device,
        )
        for k, v in parents_dict.items():
            padded = v + [[self.id, 0]] * (max_parents - len(v))
            parents[k] = torch.tensor(padded, dtype=torch.int, device=device)
        self.parents = parents

        # ---- Levels, leaves first ----
        levels_nodes = _level_order(beta)
        levels = [
            torch.tensor([l.id for l in lvl], dtype=torch.int, device=device)
            for lvl in levels_nodes
        ]
        levels.reverse()
        self.levels = levels

        # ---- TRUE node indices ----
        self.true_indices = torch.tensor(
            [n.id for n in nodes if n.is_true()],
            dtype=torch.int,
            device=device,
        )

        # ---- Literal indices and signs ----
        if any(n.is_literal() for n in nodes):
            lit = torch.tensor(
                [[n.id, n.literal] for n in nodes if n.is_literal()],
                dtype=torch.int,
                device=device,
            )
            literal_indices, signed_var = lit.unbind(-1)
            # literal_mask = (var_index, is_positive)
            literal_mask = (signed_var.abs() - 1, (signed_var > 0).long())
            self.literal_indices = literal_indices
            self.literal_mask = literal_mask

            # Positive-literal node IDs ordered by variable index — these are
            # the rows we read out as per-variable marginals.
            order = self.literal_mask[0][self.literal_mask[1].bool()].sort().indices
            self.pos_literals = self.literal_indices[
                self.literal_mask[1].bool()
            ][order]
        else:
            self.literal_indices = torch.empty(0, dtype=torch.int, device=device)
            self.literal_mask = (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
            )
            self.pos_literals = torch.empty(0, dtype=torch.int, device=device)

        # ---- Decomposition map: node -> [(prime_id, sub_id), ...] ----
        idx2primesub = torch.full(
            (self.id, max_elements, 2),
            self.id,
            dtype=torch.int,
            device=device,
        )
        for n in nodes:
            if n.is_decomposition():
                tmp = torch.tensor(
                    [[p.id, s.id] for p, s in n.elements],
                    dtype=torch.int,
                )
                idx2primesub[n.id] = torch.nn.functional.pad(
                    tmp, (0, 0, 0, max_elements - len(tmp)), value=self.id
                )
        self.idx2primesub = idx2primesub

    # ----- forward (marginals) -----------------------------------------
    def _log_Z(self, log_probs: torch.Tensor) -> torch.Tensor:
        """Functional upward pass returning log Z(constraint) of shape (batch,).
        No in-place ops, no torch.compile — keeps the autograd graph intact so
        autograd.grad can recover the marginals.

        The negative-literal weights are detached (stop-grad on log(1-p)). This
        makes d log Z / d log_p_i = P(x_i = 1 | constraint) exactly (the marginal
        trick); without the detach autograd would also flow through the
        log(1-p) path and give P(x=1) - (p/(1-p))·P(x=0) instead.
        """
        lit_weights = torch.stack(
            (log1mexp(-log_probs.detach()), log_probs), dim=-1
        ).permute(0, 2, 1)  # (n_vars, 2, batch)
        batch = log_probs.size(-1)

        # Build a (id+1, batch) tensor functionally: leaves first, then each
        # level's logsumexp via out-of-place index_put.
        base = torch.full(
            (self.id + 1, batch), -1000.0, device=self.device, dtype=log_probs.dtype
        )
        # true nodes -> log 1 = 0
        base = base.index_put(
            (self.true_indices.long(),),
            torch.zeros(self.true_indices.numel(), batch, device=self.device, dtype=log_probs.dtype),
        )
        # literal nodes -> their log weights
        lit_vals = lit_weights[self.literal_mask[0], self.literal_mask[1]]  # (num_lits, batch)
        data = base.index_put((self.literal_indices.long(),), lit_vals)

        for level in self.levels:
            level_l = level.long()
            primesub = self.idx2primesub[level_l]  # (level_size, n_elem, 2)
            gathered = data[primesub.long()]       # (level_size, n_elem, 2, batch)
            theta_level = gathered.sum(-2)         # (level_size, n_elem, batch)
            new_level = theta_level.logsumexp(-2)  # (level_size, batch)
            data = data.index_put((level_l,), new_level)

        return data[self.levels[-1].long()].squeeze(0)  # (batch,)

    def log_pr(self, log_probs: torch.Tensor) -> torch.Tensor:
        """
        Per-variable conditional log-marginals under the constraint.

        Uses the marginal trick:
            P(var_i = 1 | constraint) = d log Z(constraint) / d log_p_i
        which gives the correct first- and second-order autograd behaviour
        for free, instead of a hand-rolled downward pass that breaks the
        backward graph.

        Parameters
        ----------
        log_probs : (n_vars, batch) tensor of log P(var_i = 1)

        Returns
        -------
        (n_vars, batch) tensor of log P(var_i = 1 | constraint satisfied)
        """
        needs_local_grad = not log_probs.requires_grad
        with torch.enable_grad():
            if needs_local_grad:
                log_probs = log_probs.detach().requires_grad_(True)

            log_Z = self._log_Z(log_probs)
            marginals = torch.autograd.grad(
                log_Z.sum(),
                log_probs,
                create_graph=not needs_local_grad,
            )[0]

        return marginals.clamp_min(1e-30).log()

    # ----- first-order fast path (custom autograd.Function) ------------
    def log_pr_fast(self, log_probs: torch.Tensor) -> torch.Tensor:
        """First-order-only fast path for log marginals.

        Uses an in-place upward pass in forward and an in-place downward
        pass in backward (one circuit traversal each — half the work of
        the autograd-traced `log_pr`). The downward pass directly produces
        marginals via the SDD's structured backward.

        WARNING: only first-order gradients. Sufficient for inference and
        for losses that don't backprop through the marginals' values
        (e.g. `log_Z`-only losses). For SIMOE training — where the loss
        depends on `(samples - marginals).detach() + marginals` and so
        must backprop into the marginals — use `log_pr` instead, since
        that path supports the required double-backward.
        """
        log_Z = _CircuitLogZ.apply(log_probs, self)  # (batch,)
        marginals = torch.autograd.grad(log_Z.sum(), log_probs)[0]
        return marginals.clamp_min(1e-30).log()

    # ----- approximate sampling (Gumbel top-k) -------------------------
    @torch.no_grad()
    def gumbel_topk_sample(self, log_probs: torch.Tensor, k: int) -> torch.Tensor:
        """
        Fast approximate sample of a k-subset (NOT exact under the constraint —
        this is Gumbel top-k on the raw weights, ignoring the circuit).

        Parameters
        ----------
        log_probs : (batch, n_vars) tensor

        Returns
        -------
        (batch, n_vars) one-hot mask with exactly k ones per row.
        """
        gumbel = -torch.log(-torch.log(torch.rand_like(log_probs)))
        topk = (log_probs + gumbel).topk(k, dim=-1).indices
        out = torch.zeros_like(log_probs)
        out.scatter_(1, topk, 1.0)
        return out

    # ----- exact ancestral sampling through the circuit ----------------
    @torch.no_grad()
    def exact_sample(self, log_probs: torch.Tensor) -> torch.Tensor:
        """
        Draw an exact sample from P(assignment | constraint) by walking the
        circuit top-down, sampling one (prime, sub) element per decomposition
        according to the cached `theta` values.

        Parameters
        ----------
        log_probs : (n_vars, batch) tensor of log P(var_i = 1)

        Returns
        -------
        (batch, n_vars) 0/1 tensor.
        """
        # Recompute theta with the upward pass (we need the cached conditionals).
        lit_weights = torch.stack(
            (log1mexp(-log_probs), log_probs), dim=-1
        ).permute(0, 2, 1)  # (n_vars, 2, batch)
        batch = log_probs.size(-1)
        data = torch.empty(self.id + 1, batch, device=self.device)
        theta = torch.zeros(
            self.id + 1, self.idx2primesub.size(1), batch, device=self.device
        )
        data[self.true_indices] = 0.0
        data[self.id] = -1000.0
        data[self.literal_indices] = lit_weights[
            self.literal_mask[0], self.literal_mask[1]
        ]
        _levelwise_sl(self.levels, self.idx2primesub, data, theta)

        # Walk down. `active_count[node, b] = number of active parents that
        # selected `node` for batch b`. A node is "in the sampled sub-circuit"
        # iff active_count > 0.
        #
        # We CANNOT use bool + `|=` here: SDD subgraphs are shared, so multiple
        # parents on the same level can select the same child node. PyTorch's
        # index_put on duplicate indices without accumulate=True is
        # non-deterministic — a True write can be overwritten by a False
        # write, silently losing activation. Using int with accumulate=True
        # is order-independent and correct.
        active_count = torch.zeros(
            self.id + 1, batch, dtype=torch.long, device=self.device
        )
        active_count[self.levels[-1]] = 1  # root active

        # Process levels from root toward leaves.
        for level in reversed(self.levels):
            # log-probs over elements for each node on this level
            probs = theta[level].softmax(dim=-2)  # (level_size, n_elem, batch)
            # For each node, draw one element index per batch
            # Reshape for multinomial: (level_size * batch, n_elem)
            ls, ne, b = probs.shape
            flat = probs.permute(0, 2, 1).reshape(ls * b, ne)
            choice = torch.multinomial(flat, 1).view(ls, b)  # (level_size, batch)

            # Activate the chosen prime and sub
            chosen_pairs = idx_gather(self.idx2primesub[level], choice)  # (level_size, 2, batch)
            # Only propagate where the parent node is active.
            parent_active = (active_count[level] > 0).long()  # (level_size, batch)
            batch_idx = (
                torch.arange(batch, device=self.device)
                .unsqueeze(0)
                .expand(ls, -1)
                .reshape(-1)
            )
            for kk in range(2):  # prime, sub
                child_ids = chosen_pairs[:, kk, :].reshape(-1).long()  # (ls*batch,)
                vote = parent_active.reshape(-1)                       # (ls*batch,)
                active_count.index_put_(
                    (child_ids, batch_idx), vote, accumulate=True,
                )

        # Read out: variable i is selected iff its positive-literal node ended
        # up with at least one active parent that selected it.
        out = (active_count[self.pos_literals] > 0).T.float()  # (batch, n_vars)
        return out

    # ----- straight-through call ---------------------------------------
    def __call__(self, log_probs: torch.Tensor, k: int, exact: bool = False):
        """
        Straight-through estimator:
          forward = discrete sample, backward = marginals.

        Parameters
        ----------w
        log_probs : (batch, n_vars) tensor (NOTE: batch-first here for MoE convenience)
        k : subset size
        exact : if True use exact ancestral sampling, else Gumbel top-k

        Returns
        -------
        (samples, marginals) both (batch, n_vars). `samples` is the ST tensor
        (carries gradients of marginals); raw discrete samples are detached.
        """
        # log_pr expects (n_vars, batch)
        marginals_log = self.log_pr(log_probs.T)  # (n_vars, batch)
        marginals = marginals_log.exp().T  # (batch, n_vars)

        if exact:
            samples = self.exact_sample(log_probs.T)
        else:
            samples = self.gumbel_topk_sample(log_probs, k)

        st = (samples - marginals).detach() + marginals
        return st, marginals


# ---------------------------------------------------------------------------
# Helper used by exact_sample
# ---------------------------------------------------------------------------

def idx_gather(idx2primesub_level: torch.Tensor, choice: torch.Tensor) -> torch.Tensor:
    """
    Gather chosen (prime, sub) pairs.

    idx2primesub_level : (level_size, n_elem, 2) int
    choice             : (level_size, batch) long, value in [0, n_elem)

    Returns (level_size, 2, batch) int.
    """
    ls, ne, _ = idx2primesub_level.shape
    b = choice.shape[1]
    # Expand to (level_size, batch, 2) by gathering along n_elem axis
    choice_e = choice.unsqueeze(-1).expand(ls, b, 2)  # index along dim=1 of source
    src = idx2primesub_level.unsqueeze(1).expand(ls, b, ne, 2)
    out = src.gather(2, choice_e.unsqueeze(2)).squeeze(2)  # (ls, b, 2)
    return out.permute(0, 2, 1)  # (ls, 2, b)