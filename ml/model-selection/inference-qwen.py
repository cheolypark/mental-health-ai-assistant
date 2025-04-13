import pandas as pd
from inference import MentalHealthAssistant

# Load the dataset directly using pandas
test_data_path = "../data/test.csv"
df = pd.read_csv(test_data_path)

# Initialize the assistant
model_path = "../models/Qwen2.5-7B"
assistant = MentalHealthAssistant(model_path=model_path)

# Generate responses and store them in the 'Response' column
responses = []
for idx, row in df.iterrows():
    context = row['Context']
    print(f"Context: {context}")
    generated_response = assistant.generate_response_qwen(context)
    responses.append(generated_response)

# Update the DataFrame with generated responses
df['Response'] = responses

# Save the updated DataFrame to a CSV file
output_file = "../data/response-Qwen2.5-7B.csv"
df.to_csv(output_file, index=False)

print(f"Responses saved to {output_file}")
