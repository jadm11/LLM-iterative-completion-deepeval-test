from openai import OpenAI
import os
from deepeval.test_case import LLMTestCase
from sentence_transformers import SentenceTransformer, util
import warnings
import unittest

# Suppress runtime warnings that might clutter the output
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Initialize the OpenAI client using the API key from environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize the sentence transformer model for semantic similarity
# https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to fetch a response from the OpenAI API
def fetch_response(prompt, context):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"Context: {context}"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )
    return response.choices[0].message.content.strip()

# Test Configuration
context = "Cultural"
use_dynamic_responses = False  # Option to switch between dynamic and hardcoded expected responses
threshold = 0.4  # Define the threshold variable
prompt = "Why did the chicken cross the road?" # Input prompt

if use_dynamic_responses:
    # Fetch expected responses from the API
    expected_responses = [
        fetch_response(prompt, context),
        fetch_response(prompt, context),
        fetch_response(prompt, context)
    ]
else:
    # Use hardcoded expected responses
    expected_responses = [
        "To get to the other side.",
        ("Because its dopaminergic neurons fired synchronously across the synapses of its caudate nucleus, "
         "triggering motor contractions propelling the organism forward, to a goal predetermined by its hippocampal road mappings.")
    ]

# Fetch the actual output from the model using the same API call
model_completion = fetch_response(prompt, context)

# Define a custom similarity function 
def semantic_similarity(actual_output, expected_outputs, threshold=threshold):
    actual_embedding = embedding_model.encode(actual_output, convert_to_tensor=True)
    expected_embeddings = embedding_model.encode(expected_outputs, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(actual_embedding, expected_embeddings)
    return any(score >= threshold for score in cosine_scores[0])

# Define the test class
class SemanticSimilarityTest(unittest.TestCase):
    def test_semantic_similarity(self):
        # Given I have a response from the OpenAI API
        response = fetch_response(prompt, context)
        
        # When the response is semantically similar to any of the expected responses
        is_similar = semantic_similarity(response, expected_responses)

        # Then the response should be similar to at least one expected response
        self.assertTrue(is_similar, f"Response '{response}' is not semantically similar to any expected responses.")

if __name__ == "__main__":
    # Evaluate the test case directly using the custom similarity function
    passed = semantic_similarity(model_completion, expected_responses)

    ###### Report on results
    # ANSI escape codes for color and style formatting
    BOLD = "\033[1m"  # Bold text
    DIM = "\033[2m"  # Dimmed text for separators
    GREEN = "\033[32m"  # Green text (used for Pass result)
    RED = "\033[31m"  # Red text (used for Fail result)
    BLUE = "\033[34m"  # Blue text (used for headers)
    CYAN = "\033[36m"  # Cyan text (used for labels like Input, Expected, etc.)
    RESET = "\033[0m"  # Reset to default terminal formatting
    SEPARATOR = f"{DIM}{'-' * 50}{RESET}"  # Separator line with dimmed dashes

    # Choosing the color based on the test result
    result_color = GREEN if passed else RED  # Green for Pass, Red for Fail

    # Preparing a formatted and elegant report with additional context
    report = (
        f"{BOLD}{BLUE}Test Report{RESET}\n"  # Test Report header in bold blue text
        f"{SEPARATOR}\n"  # Separator line below the header
        f"{CYAN}Context:{RESET} {BOLD}{context}{RESET}\n"  # Context of the test in bold
        f"{CYAN}Dynamic Responses Enabled:{RESET} {BOLD}{use_dynamic_responses}{RESET}\n"  # Dynamic responses flag in bold
        f"{CYAN}Similarity Threshold:{RESET} {BOLD}{threshold}{RESET}\n\n"  # Similarity threshold in bold
        f"{CYAN}Input:{RESET} {BOLD}{prompt}{RESET}\n\n"  # Input label in cyan, followed by the actual input in bold
        f"{CYAN}Expected Responses:{RESET}\n"  # Expected Responses label in cyan
        f"  1. {expected_responses[0]}\n"  # First expected response
        f"  2. {expected_responses[1] if len(expected_responses) > 1 else ''}\n\n"  # Second expected response if available
        f"{CYAN}Actual Output:{RESET}\n"  # Actual Output label in cyan
        f"  {BOLD}{model_completion}{RESET}\n"  # The model's actual output in bold
        f"{SEPARATOR}\n"  # Separator line before the result
        f"{BOLD}Result: {result_color}{'✔ Pass' if passed else '✘ Fail'}{RESET}\n"  # Result label in bold, showing Pass or Fail in the appropriate color
    )

    print(report)  # Print the report

    # Run the unittest framework to discover and run tests
    unittest.main(argv=[''], exit=False)