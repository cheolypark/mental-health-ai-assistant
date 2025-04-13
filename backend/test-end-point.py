import requests

API_URL = "https://u74rik2yanw1qhf8.us-east-1.aws.endpoints.huggingface.cloud"
headers = {"Authorization": "Bearer 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"}

payload = {
    "inputs": "I'm feeling anxious and overwhelmed. Can you help?",
    "max_new_tokens": 150
}

response = requests.post(API_URL, headers=headers, json=payload)
print(response.json())
