import torch
import json
import time
import re
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

from eval.probmoe import patch_probmoe_block, verify_probmoe_block


EXPECTED_PROBMOE_CLASS = patch_probmoe_block("olmoe", "band")

# ==========================================
# 2. WRAPPER CLASS
# ==========================================

class HFWrapper:
    def __init__(self, model_path, gpu_count=1, max_new_tokens=128, temperature=0.0, top_p=1.0, device_map="auto"):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.device_map = device_map

        print(f"Loading tokenizer from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        
        # Ensure pad token exists
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        print(f"Loading model from {model_path}...")
        # trust_remote_code=False ensures it uses the local 'transformers' library 
        # (which we just patched above) instead of downloading code from the hub.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=False, 
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=device_map,
        )
        verify_probmoe_block(self.model, EXPECTED_PROBMOE_CLASS)
        self.model.eval()

    @torch.inference_mode()
    def generate(self, prompts, batch_size=8):
        outs = []
        # Batch generation loop
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )

            # Move inputs to the correct device
            if hasattr(self.model, "device"):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Handle temperature=0 edge case (Transformers requires do_sample=False for temp=0)
            do_sample = (self.temperature > 0)
            
            gen_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if do_sample:
                gen_kwargs["temperature"] = self.temperature
                gen_kwargs["top_p"] = self.top_p

            gen_ids = self.model.generate(**inputs, **gen_kwargs)

            # Extract new tokens only
            prompt_len = inputs["input_ids"].shape[1]
            new_tokens = gen_ids[:, prompt_len:]
            texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            outs.extend(texts)
            
        return outs

# ==========================================
# 3. EVALUATOR CLASSES
# ==========================================

class BaseEvaluator:
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.max_new_tokens = config['max_new_tokens']
        self.batch_size = config['eval_batch_size']
        
        # Initialize the Hugging Face Wrapper
        self.generator = HFWrapper(
            config["model_path"],
            config.get("gpu_count", 1),
            max_new_tokens=self.max_new_tokens,
            temperature=0.0, # Greedy decoding for benchmarks is usually standard
            top_p=1.0,
            device_map="auto",
        )

    def infer(self):
        input_text = [i for i in self.dataset["prompt"]]
        print(f"Generating responses for {len(input_text)} prompts...")
        
        responses = self.generator.generate(input_text, batch_size=self.batch_size)

        output = [{
            "prompt": input_text[i], 
            "raw_prediction": responses[i], 
            "raw_answers": self.dataset['completion'][i]
        } for i in range(len(responses))]

        return output

    def eval_metric(self, results):
        scores = []
        for sample in results:
            raw_prediction, raw_answers = sample["raw_prediction"], sample["raw_answers"]
            prediction, answers = self.post_process(raw_prediction, raw_answers)
            score = self._metrics(prediction, answers[0])
            scores.append(score)
        return scores
    
    def post_process(self, raw_prediction, raw_answers):
        pred = raw_prediction.strip()
        if pred == "":
            pred = "None"
        pred = pred.strip(".。")
        ground_truth = raw_answers
        return pred, [ground_truth]
    
    def _metrics(self, prediction, ground_truth):
        raise NotImplementedError

    def evaluate(self):
        print("Running inference on evaluation dataset...")
        results = self.infer()
        print("Evaluating results...")
        metrics = self.eval_metric(results)
        print("Evaluation complete.")
        if metrics:
             print(f"Average score: {sum(metrics) / len(metrics)}")
        return results, metrics


class GPT4Evaluator(BaseEvaluator):
    def __init__(self, dataset, config):
        super().__init__(dataset, config)
        import openai
        self.client = openai.OpenAI(api_key=config['openai_api_key'])

    def query_gpt4(self, text):
        MAX_TRIAL = 5
        response_text = ""
        for i in range(MAX_TRIAL):
            try:
                chat_completion = self.client.chat.completions.create(
                    model="gpt-4o-mini-2024-07-18",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Follow the user's instructions carefully. Respond using markdown."},
                        {"role": "user", "content": text}
                    ],
                    max_tokens=80
                )
                response_text = chat_completion.choices[0].message.content
                break
            except Exception as e:
                print(f"Error connecting to OpenAI (Attempt {i+1}/{MAX_TRIAL}):", e)
                time.sleep(10)
        return response_text

    def parse_gpt4(self, response_text):
        score = re.findall(self.pattern, response_text)
        if score:
            score = float(score[0]) / 10
        else:
            score = 0.0
            print("GPT4 did not provide a valid score:", response_text)
        return score

    @property
    def template(self):
        raise NotImplementedError
    
    @property
    def pattern(self):
        raise NotImplementedError

    def _metrics(self, prediction, ground_truth):
        text = self.template.format(prediction=prediction, ground_truth=ground_truth)
        response_text = self.query_gpt4(text)
        score = self.parse_gpt4(response_text)
        return score

# --- IMPLEMENTATION OF SPECIFIC EVALUATORS ---

class IntentEvaluator(BaseEvaluator):
    def post_process(self, raw_prediction, ground_truths):
        pred = raw_prediction.strip()
        if pred == "":
            pred = "None"
        pred = pred.strip('.。')
        if "```json" in pred:
            try:
                pred = pred[pred.index("```json") + 7:]
                pred = pred[:pred.index("```")]
            except:
                pred = "{}"
        if "\n" in pred:
            pred = [i for i in pred.split("\n") if i][0]
        pred = pred.strip('.。')
        return pred, [ground_truths]

    def _metrics(self, prediction, ground_truth):
        ground_truth = json.loads(ground_truth)
        try:
            prediction = json.loads(prediction) 
        except:
            return 0.0

        try:
            intent_em = prediction.get('intent', '') == ground_truth.get('intent', '')
            gt_slots = {(k, str(tuple(sorted([str(i) for i in v]))) if isinstance(v, list) else v) for k, v in ground_truth.get('slots', {}).items()}
        except:
            return 0.0
        try:
            pred_slots = {(k, str(tuple(sorted([str(i).replace(" ", "") for i in v]))) if isinstance(v, list) else v.replace(" ", "")) for k, v in prediction.get('slots', {}).items()}
        except:
            return 0.0  
        correct_slots = pred_slots.intersection(gt_slots)
        slots_em = (len(correct_slots) == len(pred_slots)) and (len(correct_slots) == len(gt_slots))
        return int(intent_em and slots_em)

# Use original templates from your file
SummaryTemplate = """
请你进行以下电话总结内容的评分。请依据以下标准综合考量，以确定预测答案与标准答案之间的一致性程度。满分为10分，根据预测答案的准确性、完整性和相关性来逐项扣分。请先给每一项打分并给出总分，再给出打分理由。总分为10分减去每一项扣除分数之和，最低可扣到0分。请以"内容准确性扣x分，详细程度/完整性扣x分，...，总分是：x分"为开头。
(Template Truncated for brevity, assuming existing template strings)
预测答案：{prediction}
参考答案：{ground_truth}
"""

class SummaryEvaluator(GPT4Evaluator):
    @property
    def pattern(self):
        return r"总分是：(\d+\.\d+|\d+)分"
    @property
    def template(self):
        return SummaryTemplate

LawTemplate = """
请你进行以下法案判决预测内容的评分。请依据以下标准综合考量，以确定预测答案与标准答案之间的一致性程度。满分为10分，根据预测答案的准确性、完整性和相关性来逐项扣分。请先给每一项打分并给出总分，再给出打分理由。总分为10分减去每一项扣除分数之和，最低可扣到0分。请以"相关性扣x分，完整性扣x分，...，总分是：x分"为开头。
预测答案：{prediction}
参考答案：{ground_truth}
"""

class LawEvaluator(GPT4Evaluator):
    @property
    def pattern(self):
        return r"总分是：(\d+\.\d+|\d+)分"
    @property
    def template(self):
        return LawTemplate

TranslationTemplate = """
You are an expert master in machine translation. Please score the predicted answer against the standard answer out of 10 points based on the following criteria:
Content accuracy: Does the predicted answer accurately reflect the key points of the reference answer?
Level of detail/completeness: Does the predicted answer cover all important points from the standard answer?
Content redundancy: Is the predicted answer concise and consistent with the style of the standard answer?
Respond following the format:"Content accuracy x points, level of detail/completeness x points, ..., total score: x points". The total score is the average of all the scores. Do not give reasons for your scores.
Predicted answer: {prediction}
Reference answer: {ground_truth}
"""

class TranslationEvaluator(GPT4Evaluator):
    @property
    def pattern(self):
        return r"score: *?(\d+\.\d+|\d+) *?point"
    @property
    def template(self):
        return TranslationTemplate
    def post_process(self, raw_prediction, raw_answers):
        pred = raw_prediction.strip().split("\n\n")[0]
        if pred == "": pred = "None"
        pred.strip(".。")
        ground_truth = raw_answers if isinstance(raw_answers, str) else (raw_answers[0] if raw_answers else "")
        return pred, [ground_truth]

__all__ = ["IntentEvaluator", "SummaryEvaluator", "LawEvaluator", "TranslationEvaluator"]
