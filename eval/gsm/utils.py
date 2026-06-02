import torch
import tqdm
import os
from importlib import import_module
from transformers import (
    StoppingCriteria,
    StoppingCriteriaList,
    LogitsProcessorList,
    NoBadWordsLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor
)
import sys
import transformers.models.olmoe.modeling_olmoe as olmoe_module
import transformers.models.qwen2_moe.modeling_qwen2_moe as Qwen2_module

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def set_model_eval_routing_method(model, method):
    try:
        for m in model.modules():
                if hasattr(m, "set_eval_routing_method"):
                    # logger.info(f"Setting eval routing method to {method}")
                    m.set_eval_routing_method(method)
    except Exception as e:
        logger.warning(f"Failed to set eval routing method: {e}")


class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            sequence_should_be_stopped = False
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequence_should_be_stopped = True
                    break
            sequences_should_be_stopped.append(sequence_should_be_stopped)
        return all(sequences_should_be_stopped)


@torch.inference_mode()
def generate_completions(
    model,
    tokenizer,
    prompts,
    batch_size=1,
    stop_id_sequences=None,
    banned_id_sequences=None,
    banned_begin_ids=None,
    add_special_tokens=True,
    disable_tqdm=False,
    temperature=1.0,
    top_p=1.0,
    **generation_kwargs
):
    generations = []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Generating Completions")

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        tokenized_prompts = tokenizer(
            batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=add_special_tokens
        )
        batch_input_ids = tokenized_prompts['input_ids']
        attention_mask = tokenized_prompts['attention_mask']

        if model.device.type == "cuda":
            if isinstance(batch_input_ids, dict):
                for k in batch_input_ids:
                    batch_input_ids[k] = batch_input_ids[k].cuda()
                    attention_mask[k] = attention_mask[k].cuda()
            else:
                batch_input_ids = batch_input_ids.cuda()
                attention_mask = attention_mask.cuda()

        stopping_criteria = StoppingCriteriaList([KeyWordsCriteria(stop_id_sequences)]) if stop_id_sequences else None

        # create logit processors
        if banned_id_sequences or banned_begin_ids:
            logit_processors = []
            if banned_id_sequences:
                logit_processors.append(
                    NoBadWordsLogitsProcessor(banned_id_sequences, eos_token_id=tokenizer.eos_token_id)
                )
            if banned_begin_ids:
                logit_processors.append(
                    SuppressTokensAtBeginLogitsProcessor(banned_begin_ids, begin_index=batch_input_ids.shape[1])
                )
            logits_processor = LogitsProcessorList(logit_processors)
        else:
            logits_processor = None

        batch_outputs = model.generate(
            input_ids=batch_input_ids,
            attention_mask=attention_mask,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            temperature=temperature,
            top_p=top_p,
            **generation_kwargs
        )

        # to support the logits processing below when using DExperts with mixed tokenizers
        if isinstance(batch_input_ids, dict):
            batch_input_ids = batch_input_ids['llama']

        # the stopping criteria is applied at batch level, so if other examples are not stopped,
        # the entire batch will continue to generate. so some outputs still have the stop sequence,
        # which we need to remove.
        if stop_id_sequences:
            for output_idx in range(batch_outputs.shape[0]):
                for token_idx in range(batch_input_ids.shape[1], batch_outputs.shape[1]):
                    if any(batch_outputs[output_idx, token_idx: token_idx+len(stop_sequence)].tolist() == stop_sequence for stop_sequence in stop_id_sequences):
                        batch_outputs[output_idx, token_idx:] = tokenizer.pad_token_id
                        break

        # remove the prompt from the output
        # we need to re-encode the prompt because we need to make sure the special tokens are treated
        # the same way as in the outputs. we changed our previous way of truncating the output token ids
        # directly because some tokenizer (e.g., llama) won't add space token before the first token.
        # space is important for some tasks (e.g., code completion).
        batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
        batch_prompts = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)

        # duplicate the prompts to match the number of return sequences
        batch_prompts = [prompt for prompt in batch_prompts for _ in range(num_return_sequences)]
        batch_generations = [
            output[len(prompt):] for prompt, output in zip(batch_prompts, batch_outputs)
        ]

        generations += batch_generations

        if not disable_tqdm:
            progress.update(len(batch_prompts)//num_return_sequences)

    assert len(generations) == len(prompts) * num_return_sequences, "number of generations should be equal to number of prompts * num_return_sequences"
    return generations


def load_lm_and_tokenizer(
    model_name_or_path,
    tokenizer_name_or_path=None,
    device_map="auto",
    load_in_8bit=False,
    convert_to_half=False,
    use_fast_tokenizer=True,
    padding_side="left",
    mode="exact",
    custom_moe_path="",
    model_type="olmoe",
):
    if custom_moe_path is not None and custom_moe_path != "":
        sys.path.append(custom_moe_path)
    
    if model_type == "olmoe":
        if mode == "band":
            print("Using dynamic mode olmoe block")
            from OLMoE.v1.SIMoE_V1_olmoe_dynamic import SIMoEOlmoeSparseMoeBlock
            olmoe_module.OlmoeSparseMoeBlock = SIMoEOlmoeSparseMoeBlock
        else:
            print("Using exact mode olmoe block")
    elif model_type == "qwen":
        if mode == "band":
            print("Using dynamic mode qwen2 moe block")
            from Qwen.SIMoE_V1_qwen_dynamic import SIMoEQwen2MoeSparseMoeBlock
            Qwen2_module.Qwen2MoeSparseMoeBlock = SIMoEQwen2MoeSparseMoeBlock
        else:
            print("Using exact mode qwen2 moe block")
        
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_kwargs = {
        'device_map': device_map,
        'offload_folder': 'offload_folder',
        'torch_dtype': torch.float16,
        'offload_state_dict': True,
        'load_in_8bit': load_in_8bit
    }
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    if convert_to_half:
        model = model.half()
    model.eval()

    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=use_fast_tokenizer)
    tokenizer = add_pad_token(tokenizer, padding_side)

    if eval_routing_method == "simple_eval":
        set_model_eval_routing_method(model, "simple_eval")
        print("Set model eval routing method to simple_eval")
    else:
        set_model_eval_routing_method(model, "deterministic")
        print("Set model eval routing method to deterministic")
    
    return model, tokenizer


def add_pad_token(tokenizer, padding_side="left"):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = padding_side
    return tokenizer


def load_dexperts_model_and_tokenizer(
    base_model_name_or_path: str,
    expert_model_name_or_path: str,
    antiexpert_model_name_or_path: str = None,
    device_map: str = "auto",
    system_prompt: str = None,
    alpha: float = 1.0,
    chat_response_prefix: str = None,
    load_in_8bit: bool = False,
    use_fast_tokenizer: bool = True,
    padding_side: str = "left",
):
    from transformers import AutoTokenizer
    from modeling.dexperts import DExpertsLlama

    model_kwargs = {
        'device_map': device_map,
        'offload_folder': 'offload_folder',
        'torch_dtype': torch.float16,
        'offload_state_dict': True,
        'load_in_8bit': load_in_8bit,
    }

    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast_tokenizer=use_fast_tokenizer)
    tokenizer = add_pad_token(tokenizer, padding_side)
    if not antiexpert_model_name_or_path:
        antiexpert_model_name_or_path = 'meta-llama/Llama-2-7b-hf'

    model = DExpertsLlama(
        base_model_name_or_path=base_model_name_or_path,
        expert_model_name_or_path=expert_model_name_or_path,
        antiexpert_model_name_or_path=antiexpert_model_name_or_path,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        alpha=alpha,
        chat_response_prefix=chat_response_prefix,
        model_kwargs=model_kwargs,
    )

    return model, tokenizer


def dynamic_import_function(function_path):
    '''
    Dynamically import a function from a path string (e.g., "module.submodule.my_function")
    '''
    module_path, function_name = function_path.rsplit(".", 1)
    module = import_module(module_path)
    function = getattr(module, function_name)
    return function