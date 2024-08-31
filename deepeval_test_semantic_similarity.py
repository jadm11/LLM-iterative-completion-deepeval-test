import os  # Standard library module for interacting with the operating system
import warnings  # Standard library module for managing warnings in Python
import unittest  # Standard library module for creating and running unit tests

from openai import OpenAI  # Import the OpenAI class to interact with the OpenAI API
from deepeval.test_case import LLMTestCase  # Import the LLMTestCase from the deepeval package for test case creation
from sentence_transformers import SentenceTransformer, util  # Import SentenceTransformer and utility functions for semantic similarity

# Suppress runtime warnings that might clutter the output
warnings.simplefilter("ignore", category=RuntimeWarning)

# Initialize the OpenAI client using an API key from environment variables
# This client is used to make API calls to the OpenAI model
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize the SentenceTransformer model for semantic similarity
# The model is loaded from the Hugging Face model hub; used to compute embeddings for text comparison
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to fetch a response from the OpenAI API
# Takes a prompt and context, sends them to the model, and returns the generated response as a string
def fetch_response(prompt, context):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Specify the model version to use
        messages=[
            {"role": "system", "content": f"Context: {context}"},  # Provide context to guide the model's behavior
            {"role": "user", "content": prompt}  # Provide the actual user prompt
        ],
        max_tokens=100  # Limit the response length to 100 tokens
    )
    return response.choices[0].message.content.strip()  # Return the generated response as a string

# Function to compute semantic similarity between actual output and expected outputs
# Uses cosine similarity between embeddings to evaluate how closely the actual output matches the expected outputs
def semantic_similarity(actual_output, expected_outputs, threshold=0.4):
    actual_embedding = embedding_model.encode(actual_output, convert_to_tensor=True)  # Convert the actual output to an embedding
    expected_embeddings = embedding_model.encode(expected_outputs, convert_to_tensor=True)  # Convert expected outputs to embeddings
    cosine_scores = util.pytorch_cos_sim(actual_embedding, expected_embeddings)  # Compute cosine similarity scores
    return any(score >= threshold for score in cosine_scores[0])  # Return True if any similarity score meets or exceeds the threshold

# Unit test class for semantic similarity testing
# Contains a single test case that verifies if the model's response is similar to any expected response
class SemanticSimilarityTest(unittest.TestCase):
    def test_semantic_similarity(self):
        # Given that I ahve a response from the OpenAI model
        response = fetch_response(prompt, context)
        
        # When the response is semantically similar to the expected responses
        is_similar = semantic_similarity(response, expected_responses)
        
        # Then the respons should be similar to at least one of the expected responses
        self.assertTrue(is_similar, f"Response '{response}' is not semantically similar to any expected responses.")

# Configuration settings for the test
context = "Cultural"  # Context provided to the model to influence its response
use_dynamic_responses = False  # Toggle for using dynamic or hardcoded expected responses
threshold = 0.4  # Similarity threshold for passing the test
prompt = "Why did the chicken cross the road?"  # The prompt provided to the model

# Generate expected responses based on the configuration
if use_dynamic_responses:
    # Dynamically generate expected responses using the model
    expected_responses = [
        fetch_response(prompt, context),
        fetch_response(prompt, context),
        fetch_response(prompt, context)
    ]
else:
    # Use predefined hardcoded responses
    expected_responses = [
        "To get to the other side.",  # Traditional simple answer
        ("Because its dopaminergic neurons fired synchronously across the synapses of its caudate nucleus, "
         "triggering motor contractions propelling the organism forward, to a goal predetermined by its hippocampal road mappings.")  # Complex, scientific explanation
    ]

# Fetch the actual output from the model using the same API call
model_completion = fetch_response(prompt, context)

# Main execution block to evaluate the test case and generate a report
if __name__ == "__main__":
    # Evaluate whether the model's actual output matches the expected responses based on the similarity threshold
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

    # Choose the color based on the test result
    result_color = GREEN if passed else RED  # Green for Pass, Red for Fail

    # Prepare a formatted and elegant report with additional context
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