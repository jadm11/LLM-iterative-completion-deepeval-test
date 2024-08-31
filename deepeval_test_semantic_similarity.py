import os  # Standard library module for interacting with the operating system
import warnings  # Standard library module for managing warnings in Python
import unittest  # Standard library module for creating and running unit tests

from openai import OpenAI  # Import the OpenAI class to interact with the OpenAI API
from deepeval.test_case import LLMTestCase  # Import the LLMTestCase from the deepeval package for test case creation
from sentence_transformers import SentenceTransformer, util  # Import SentenceTransformer and utility functions for semantic similarity
from colorama import init, Fore, Style  # Import colorama for colored terminal output

# Initialize colorama for cross-platform colored output
init(autoreset=True)

# Suppress runtime warnings that might clutter the output
warnings.simplefilter("ignore", category=RuntimeWarning)

# Initialize the OpenAI client using an API key from environment variables
# The client will be used to make API calls to the OpenAI model.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize the SentenceTransformer model for semantic similarity
# The model is loaded from the Hugging Face model hub and used to compute embeddings for text comparison.
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to fetch a response from the OpenAI API
# Takes a prompt and context, sends them to the model, and returns the generated response as a string.
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
# Uses cosine similarity between embeddings to evaluate how closely the actual output matches the expected outputs.
def semantic_similarity(actual_output, expected_outputs, threshold=0.4):
    actual_embedding = embedding_model.encode(actual_output, convert_to_tensor=True)  # Convert the actual output to an embedding
    expected_embeddings = embedding_model.encode(expected_outputs, convert_to_tensor=True)  # Convert expected outputs to embeddings
    cosine_scores = util.pytorch_cos_sim(actual_embedding, expected_embeddings)  # Compute cosine similarity scores
    return any(score >= threshold for score in cosine_scores[0])  # Return True if any similarity score meets or exceeds the threshold

# Define a custom unittest.TestCase class that uses both LLMTestCase structure and custom similarity checks
class CustomSemanticSimilarityTest(unittest.TestCase):
    def setUp(self):
        """Setup the test case with prompt, context, and expected output."""
        self.prompt = prompt  # Set the prompt for the test case
        self.context = context  # Set the context for the test case
        self.threshold = threshold  # Set the similarity threshold
        self.actual_output = fetch_response(self.prompt, self.context)  # Fetch the actual output from the model
        self.expected_output = expected_responses  # Set the expected responses

    def test_validate_output(self):
        """Run the custom semantic similarity test."""
        passed = semantic_similarity(self.actual_output, self.expected_output, self.threshold)  # Perform the semantic similarity check
        self.generate_report(passed)  # Generate a report based on the test result
        self.assertTrue(passed, "The output did not pass the semantic similarity test.")  # Assert that the output passed the test

    def generate_report(self, passed):
        """Generates and prints a formatted report for the test result."""
        # ANSI escape codes for color and style formatting
        BOLD = Style.BRIGHT  # Bold text
        DIM = Style.DIM  # Dimmed text for separators
        GREEN = Fore.GREEN  # Green text (used for Pass result)
        RED = Fore.RED  # Red text (used for Fail result)
        BLUE = Fore.BLUE  # Blue text (used for headers)
        CYAN = Fore.CYAN  # Cyan text (used for labels like Input, Expected, etc.)
        RESET = Style.RESET_ALL  # Reset to default terminal formatting
        SEPARATOR = f"{DIM}{'-' * 50}{RESET}"  # Separator line with dimmed dashes

        result_color = GREEN if passed else RED  # Choose the color based on the test result

        # Prepare a formatted and elegant report with additional context
        report = (
            f"{BOLD}{BLUE}Test Report{RESET}\n"
            f"{SEPARATOR}\n"
            f"{CYAN}Context:{RESET} {BOLD}{self.context}{RESET}\n"
            f"{CYAN}Dynamic Responses Enabled:{RESET} {BOLD}{use_dynamic_responses}{RESET}\n"
            f"{CYAN}Similarity Threshold:{RESET} {BOLD}{self.threshold}{RESET}\n\n"
            f"{CYAN}Input:{RESET} {BOLD}{self.prompt}{RESET}\n\n"
            f"{CYAN}Expected Responses:{RESET}\n"
            f"  1. {self.expected_output[0]}\n"
            f"  2. {self.expected_output[1] if len(self.expected_output) > 1 else ''}\n\n"
            f"{CYAN}Actual Output:{RESET}\n"
            f"  {BOLD}{self.actual_output}{RESET}\n"
            f"{SEPARATOR}\n"
            f"{BOLD}Result: {result_color}{'✔ Pass' if passed else '✘ Fail'}{RESET}\n"
        )

        print(report)  # Print the report to the console

# Configuration settings for the test
context = "Scientific"  # Context provided to the model to influence its response
use_dynamic_responses = False  # Toggle for using dynamic or hardcoded expected responses
threshold = 0.1  # Similarity threshold for passing the test
prompt = "Why did the chicken cross the road?"  # The prompt provided to the model

# Generate expected responses based on the configuration
if use_dynamic_responses:
    expected_responses = [
        fetch_response(prompt, context),  # Dynamically fetch expected responses
        fetch_response(prompt, context),
        fetch_response(prompt, context)
    ]
else:
    expected_responses = [
        "To get to the other side.",  # Use predefined hardcoded responses
        ("Because its dopaminergic neurons fired synchronously across the synapses of its caudate nucleus, "
         "triggering motor contractions propelling the organism forward, to a goal predetermined by its hippocampal road mappings.")
    ]

# Main execution block to run the test case
if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)  # Run the unittest framework to discover and run tests