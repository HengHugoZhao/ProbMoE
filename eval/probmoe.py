"""Load and verify the framework-local ProbMoE blocks used for evaluation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from torch import nn


ModelType = Literal["olmoe", "qwen"]
RoutingMode = Literal["exact", "band", "dynamic"]

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OLMOE_SOURCE = _REPO_ROOT / "train" / "olmoe" / "open_instruct"
_QWEN_SOURCE = _REPO_ROOT / "train" / "qwen" / "LLaMA-Factory" / "src" / "llamafactory" / "model"


def _add_framework_source(source: Path) -> None:
    """Make an in-repository framework package importable without user paths."""
    source_string = str(source)
    if source_string not in sys.path:
        sys.path.insert(0, source_string)


def patch_probmoe_block(model_type: ModelType, mode: RoutingMode) -> type[nn.Module]:
    """Patch the selected Transformers MoE block and return the expected class."""
    dynamic = mode in {"band", "dynamic"}

    if model_type == "olmoe":
        _add_framework_source(_OLMOE_SOURCE)
        import transformers.models.olmoe.modeling_olmoe as model_module

        if dynamic:
            from OLMoE.v1.ProbMoE_V1_olmoe_dynamic import ProbMoEOlmoeSparseMoeBlock
        else:
            from OLMoE.v1.ProbMoE_V1_olmoe_exact import ProbMoEOlmoeSparseMoeBlock

        model_module.OlmoeSparseMoeBlock = ProbMoEOlmoeSparseMoeBlock
        return ProbMoEOlmoeSparseMoeBlock

    if model_type == "qwen":
        _add_framework_source(_QWEN_SOURCE)
        import transformers.models.qwen2_moe.modeling_qwen2_moe as model_module

        if dynamic:
            from Qwen.ProbMoE_V1_qwen_dynamic import ProbMoEQwen2MoeSparseMoeBlock
        else:
            from Qwen.ProbMoE_V1_qwen_exact import ProbMoEQwen2MoeSparseMoeBlock

        model_module.Qwen2MoeSparseMoeBlock = ProbMoEQwen2MoeSparseMoeBlock
        return ProbMoEQwen2MoeSparseMoeBlock

    raise ValueError(f"Unsupported model type: {model_type!r}.")


def verify_probmoe_block(model: nn.Module, expected_class: type[nn.Module]) -> tuple[str, nn.Module]:
    """Return the first loaded ProbMoE block, or fail if the patch did not apply."""
    for name, module in model.named_modules():
        if isinstance(module, expected_class):
            if not hasattr(module, "probmoe_routing"):
                raise RuntimeError(f"ProbMoE block {name} is missing probmoe_routing().")
            print(f"Found ProbMoE block: {name} - Type: {type(module)}")
            return name, module

    raise RuntimeError(
        f"No {expected_class.__name__} was found after loading the model; the ProbMoE patch did not apply."
    )
