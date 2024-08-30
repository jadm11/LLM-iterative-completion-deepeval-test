from openai import OpenAI
import os
from deepeval.test_case import LLMTestCase
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to fetch expected responses using the updated OpenAI API
def fetch_expected_responses(prompt, context):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"Context: {context}"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )
    return [response.choices[0].message.content.strip()]

# Example prompt and context
prompt = "Why did the chicken cross the road?"
context = "Cultural"

# Fetch expected responses from the API
expected_responses = fetch_expected_responses(prompt, context)

# Add manually crafted expected responses
expected_responses.extend([
    "To get to the other side.",
    ("Because its dopaminergic neurons fired synchronously across the synapses of its caudate nucleus, "
     "triggering motor contractions propelling the organism forward, to a goal predetermined by its hippocampal road mappings.")
])

# Simulated actual model output
model_completion = "To get to the other side."

# Create a DeepEval test case
test_case = LLMTestCase(
    input=prompt,
    actual_output=model_completion,
    expected_output=expected_responses
)

# Manual evaluation as a fallback
passed = model_completion in expected_responses

# Print the result with a basic report
report = {
    "input": prompt,
    "expected_output": expected_responses,
    "actual_output": model_completion,
    "result": "Pass" if passed else "Fail"
}

print("Test Report:")
for key, value in report.items():
    print(f"{key}: {value}")

