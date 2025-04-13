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
model_path = "../ml/models/Llama-3.2-3B-Instruct"
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

@app.get("/get_response")
async def get_response(user_message: str = Query(..., description="User's message to the assistant")):
    print('Received user message:', user_message)
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_message}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=2000, num_return_sequences=1)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    response = text.split("assistant")[-1].strip()

    print('response with message:', response)

    return {"response": response}
