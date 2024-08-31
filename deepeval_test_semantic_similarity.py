import os  # For interacting with the operating system, particularly for environment variables
import warnings  # For managing warnings in Python, specifically to suppress runtime warnings
from openai import OpenAI  # To interact with the OpenAI API, used for generating responses from a language model
from deepeval.test_case import LLMTestCase  # Import LLMTestCase from the DeepEval package for testing LLM outputs
from sentence_transformers import SentenceTransformer, util  # For performing semantic similarity checks
from colorama import init, Fore, Style  # For colored terminal output to enhance readability of the report

# Initialize colorama to ensure colored output works across different platforms
init(autoreset=True)

# Suppress runtime warnings to focus only on critical information during execution
warnings.simplefilter("ignore", category=RuntimeWarning)

# Initialize the OpenAI client using the API key retrieved from environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load the SentenceTransformer model for computing semantic similarity between text embeddings
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def fetch_response(prompt, context):
    """Fetches a response from the OpenAI API based on the provided prompt and context."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Specify the OpenAI model to be used
        messages=[
            {"role": "system", "content": f"Context: {context}"},  # Set the system context for the model's behavior
            {"role": "user", "content": prompt}  # Provide the user's prompt to the model
        ],
        max_tokens=100  # Limit the response length to 100 tokens
    )
    return response.choices[0].message.content.strip()  # Return the model's response, stripped of any leading/trailing whitespace

def semantic_similarity(actual_output, expected_outputs, threshold=0.4):
    """Calculates semantic similarity between the actual output and expected outputs."""
    # Encode the actual output and expected outputs into embeddings for comparison
    actual_embedding = embedding_model.encode(actual_output, convert_to_tensor=True)
    expected_embeddings = embedding_model.encode(expected_outputs, convert_to_tensor=True)
    
    # Compute cosine similarity scores between the actual output and each expected output
    cosine_scores = util.pytorch_cos_sim(actual_embedding, expected_embeddings)
    
    # Return True if any of the similarity scores meet or exceed the threshold, along with the scores for reporting
    return any(score >= threshold for score in cosine_scores[0]), cosine_scores

class CustomLLMEvaluation:
    """Combines LLMTestCase and SentenceTransformer for comprehensive evaluation of LLM outputs."""
    
    def __init__(self, prompt, context, threshold=0.4, use_dynamic_responses=False):
        self.prompt = prompt  # The input prompt to be tested
        self.context = context  # The context guiding the LLM's response
        self.threshold = threshold  # The similarity threshold for semantic similarity
        self.use_dynamic_responses = use_dynamic_responses  # Flag to determine if expected responses should be dynamically generated
        
        # Fetch the actual output from the model based on the provided prompt and context
        self.actual_output = fetch_response(self.prompt, self.context)

        # Generate expected responses either dynamically or use predefined hardcoded responses
        if self.use_dynamic_responses:
            self.expected_responses = [
                fetch_response(self.prompt, self.context),
                fetch_response(self.prompt, self.context),
                fetch_response(self.prompt, self.context)
            ]
        else:
            self.expected_responses = [
                "To get to the other side.",  # Common, simple response
                ("Because its dopaminergic neurons fired synchronously across the synapses of its caudate nucleus, "
                 "triggering motor contractions propelling the organism forward, to a goal predetermined by its hippocampal road mappings.")  # Complex, scientific explanation
            ]

        # Initialize LLMTestCase with the provided inputs, actual output, and expected output
        self.llm_test_case = LLMTestCase(
            input=self.prompt,
            actual_output=self.actual_output,
            expected_output=self.expected_responses[0]  # Use the first expected response for exact match checking
        )

    def run_evaluation(self):
        """Runs the evaluation using both semantic similarity and LLMTestCase."""
        # Check if the actual output matches the expected output exactly
        llm_test_passed = self.llm_test_case.actual_output.strip() == self.llm_test_case.expected_output.strip()
        
        # Check if the actual output is semantically similar to any of the expected outputs
        semantic_passed, cosine_scores = semantic_similarity(self.actual_output, self.expected_responses, self.threshold)

        # Determine the overall result based on both checks
        overall_passed = llm_test_passed or semantic_passed
        
        # Generate a detailed report of the evaluation
        self.generate_report(overall_passed, cosine_scores)
        return overall_passed

    def generate_report(self, passed, cosine_scores):
        """Generates a detailed report of the evaluation results."""
        # Define styles and colors for the report
        BOLD = Style.BRIGHT
        DIM = Style.DIM
        GREEN = Fore.GREEN
        RED = Fore.RED
        BLUE = Fore.BLUE
        CYAN = Fore.CYAN
        RESET = Style.RESET_ALL
        SEPARATOR = f"{DIM}{'-' * 50}{RESET}"

        # Determine the result color based on whether the test passed or failed
        result_color = GREEN if passed else RED

        # Prepare the detailed report
        report = (
            f"{BOLD}{BLUE}Test Report{RESET}\n"
            f"{SEPARATOR}\n"
            f"{CYAN}Context:{RESET} {BOLD}{self.context}{RESET}\n"
            f"{CYAN}Dynamic Responses Enabled:{RESET} {BOLD}{self.use_dynamic_responses}{RESET}\n"
            f"{CYAN}Similarity Threshold:{RESET} {BOLD}{self.threshold}{RESET}\n"
            f"{CYAN}Cosine Scores:{RESET} {BOLD}{cosine_scores}{RESET}\n\n"
            f"{CYAN}Input:{RESET} {BOLD}{self.prompt}{RESET}\n\n"
            f"{CYAN}Expected Responses:{RESET}\n"
            f"  1. {self.expected_responses[0]}\n"
            f"  2. {self.expected_responses[1] if len(self.expected_responses) > 1 else ''}\n\n"
            f"{CYAN}Actual Output:{RESET}\n"
            f"  {BOLD}{self.actual_output}{RESET}\n"
            f"{SEPARATOR}\n"
            f"{BOLD}Result: {result_color}{'✔ Pass' if passed else '✘ Fail'}{RESET}\n"
        )

        # Print the report to the console
        print(report)

# Configuration settings for running the evaluation
context = "Scientific"  # Set the context for the model's response
use_dynamic_responses = False  # Disable dynamic generation of expected responses
threshold = 0.5  # Set a low similarity threshold for testing
prompt = "Why did the chicken cross the road?"  # The input prompt to be evaluated

# Run the evaluation
evaluation = CustomLLMEvaluation(prompt, context, threshold, use_dynamic_responses)
evaluation.run_evaluation()
