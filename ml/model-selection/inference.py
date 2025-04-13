from unsloth import FastLanguageModel
from transformers import AutoTokenizer
import torch


alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


class MentalHealthAssistant:
    def __init__(self, model_path, max_seq_length=2048):
        self.instruction = """You are Aider, a compassionate, empathetic, and knowledgeable AI assistant specialized in mental health support.
    Your primary role is to provide thoughtful, supportive, and informative responses to users seeking guidance, information,
    or emotional support related to mental health and well-being."""
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=torch.float16,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)

    def generate_response_llama(self, user_message, max_new_tokens=150):
        messages = [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": user_message}
        ]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")

        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, num_return_sequences=1)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        response = text.split("assistant")[-1].strip()
        return response

    def extract_response(self, context_array):
        response_marker = "### Response:"
        context_text = context_array[0]
        response_start = context_text.find(response_marker)

        if response_start == -1:
            return None

        response_text = context_text[response_start + len(response_marker):].strip()
        return response_text

    def generate_response_qwen(self, user_message, max_new_tokens=150):
        inputs = self.tokenizer(
            [
                alpaca_prompt.format(
                    self.instruction,  # instruction
                    user_message,  # input
                    "",  # output - leave this blank for generation!
                )
            ], return_tensors="pt").to("cuda")

        outputs = self.model.generate(**inputs, max_new_tokens=64, use_cache=True)
        context_array = self.tokenizer.batch_decode(outputs)
        response = self.extract_response(context_array)
        return response
