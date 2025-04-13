from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from huggingface_hub import HfApi
import torch

model_path = "../models/Llama-3.2-3B-Instruct"
max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=max_seq_length,
    dtype=torch.float16,
    load_in_4bit=True,
)

api = HfApi()

api.create_repo(repo_id="cheoly7/mental-health-assistant", exist_ok=True)
model.push_to_hub("cheoly7/mental-health-assistant")
tokenizer.push_to_hub("cheoly7/mental-health-assistant")
