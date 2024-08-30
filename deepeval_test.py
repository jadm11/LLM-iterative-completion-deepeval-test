from openai import OpenAI  # Import the OpenAI class to interact with the OpenAI API
import os  # Import the os module to access environment variables
from deepeval.test_case import LLMTestCase  # Import the LLMTestCase class from deepeval for setting up the test case
import warnings  # Import warnings to suppress runtime warnings

# Suppress runtime warnings that might clutter the output
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Initialize the OpenAI client using the API key from environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to fetch expected responses using the OpenAI API
def fetch_expected_responses(prompt, context):
    # Create a chat completion using the provided prompt and context
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Specify the model to use
        messages=[
            {"role": "system", "content": f"Context: {context}"},  # System message to set context
            {"role": "user", "content": prompt}  # User message with the prompt
        ],
        max_tokens=100  # Limit the number of tokens in the response
    )
    # Return the stripped content of the first choice from the response
    return [response.choices[0].message.content.strip()]

# Example prompt and context to be tested
prompt = "Why did the chicken cross the road?"
context = "Cultural"  # Context can be 'Cultural', 'Scientific', etc.

# Fetch expected responses from the API based on the prompt and context
expected_responses = fetch_expected_responses(prompt, context)

# Manually add additional expected responses for comprehensive testing
expected_responses.extend([
    "To get to the other side.",  # Traditional response
    ("Because its dopaminergic neurons fired synchronously across the synapses of its caudate nucleus, "
     "triggering motor contractions propelling the organism forward, to a goal predetermined by its hippocampal road mappings.")  # Scientific response
])

# Simulate the actual output from the model
model_completion = "To get to the other side."

# Create a DeepEval test case using the prompt, actual output, and expected outputs
test_case = LLMTestCase(
    input=prompt,
    actual_output=model_completion,
    expected_output=expected_responses
)

# Manually evaluate if the model's output is in the expected responses
passed = model_completion in expected_responses

# Create a simple report dictionary to display the results
report = {
    "input": prompt,  # The input prompt used
    "expected_output": expected_responses,  # The list of expected responses
    "actual_output": model_completion,  # The actual output from the model
    "result": "Pass" if passed else "Fail"  # The result of the test
}

# Print the test report in a readable format
print("Test Report:")
for key, value in report.items():
    print(f"{key}: {value}")

