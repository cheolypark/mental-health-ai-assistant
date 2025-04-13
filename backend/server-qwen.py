from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model and tokenizer once at startup
model_path = "../ml/fine-tuning/Llama-3.2-3B-Instruct"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=True,
)

# Enable faster inference
FastLanguageModel.for_inference(model)

instruction = """You are Aider, a compassionate, empathetic, and knowledgeable AI assistant specialized in mental health support.
    Your primary role is to provide thoughtful, supportive, and informative responses to users seeking guidance, information,
    or emotional support related to mental health and well-being."""

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def extract_response(context_array):
    response_marker = "### Response:"
    context_text = context_array[0]
    response_start = context_text.find(response_marker)

    if response_start == -1:
        return None

    response_text = context_text[response_start + len(response_marker):].strip()
    return response_text


@app.get("/get_response")
async def get_response(user_message: str = Query(..., description="User's message to the assistant")):
    print('Received user message:', user_message)
    FastLanguageModel.for_inference(model)
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                instruction,  # instruction
                user_message,  # input
                "",  # output - leave this blank for generation!
            )
        ], return_tensors="pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
    context_array = tokenizer.batch_decode(outputs)
    response = extract_response(context_array)

    print('response with message:', response)

    return {"response": response}
