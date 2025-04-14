import pandas as pd
from utils.openai import request_chat_completion
import re

response_file_qwen = "../data/response-Qwen2.5-7B.csv"
response_file_llama = "../data/response-Llama-3.2-3B-Instruct.csv"

df_qwen = pd.read_csv(response_file_qwen)
df_llama = pd.read_csv(response_file_llama)

# Initialize counters
qwen_count = 0
llama_count = 0
tie_count = 0

# Iterate through responses
for idx in range(min(len(df_qwen), len(df_llama))):
    context_qwen = df_qwen.loc[idx, 'Context']
    response_qwen = df_qwen.loc[idx, 'Response']

    context_llama = df_llama.loc[idx, 'Context']
    response_llama = df_llama.loc[idx, 'Response']

    # Check if contexts match
    if context_qwen != context_llama:
        print(f"\nWarning: Context mismatch at row {idx}!")
        continue

    prompt = f"""
    You are tasked with evaluating two AI-generated responses based on the provided context. Your evaluation should focus on two criteria:

    1. Accuracy: How correctly does the response address the context provided?
    2. Contextual Relevance: How relevant, coherent, and suitable is the response given the context?

    Please provide a brief comparative analysis and clearly state which response (Qwen or Llama) is superior overall, along with a concise justification.

    Context:
    {context_qwen}

    Qwen Response:
    {response_qwen}

    Llama Response:
    {response_llama}

    Your evaluation should follow this format:

    Accuracy:
    - Qwen: [Brief evaluation]
    - Llama: [Brief evaluation]

    Contextual Relevance:
    - Qwen: [Brief evaluation]
    - Llama: [Brief evaluation]

    Overall Superior Response: [Qwen or Llama]
    Reason: [Brief justification]
    """

    compared = request_chat_completion(prompt)
    print(f"Evaluation [{idx}]:\n{compared}\n{'-'*80}")

    # Extract the Overall Superior Response
    match = re.search(r"Overall Superior Response:\s*(Qwen|Llama)", compared, re.IGNORECASE)
    if match:
        winner = match.group(1).strip().lower()
        if winner == "qwen":
            qwen_count += 1
        elif winner == "llama":
            llama_count += 1
    else:
        print(f"Could not determine winner clearly at index {idx}. Counting as tie.")
        tie_count += 1

# Display final scores
print("\nFinal Scores:")
print(f"Qwen: {qwen_count}")
print(f"Llama: {llama_count}")
print(f"Ties/Undetermined: {tie_count}")
