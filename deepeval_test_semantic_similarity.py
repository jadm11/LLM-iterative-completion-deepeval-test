# Standard Library Imports
import os  # For interacting with the operating system, particularly for accessing environment variables
import warnings  # For managing and suppressing warnings that might clutter output
import logging  # For logging events, errors, and other significant runtime information

# Third-Party Imports
from openai import OpenAI  # To interact with the OpenAI API, used to generate responses from a language model
from deepeval.test_case import LLMTestCase  # Import LLMTestCase from the DeepEval package for structured testing of LLM outputs
from sentence_transformers import SentenceTransformer, util  # For performing semantic similarity checks using embeddings
from colorama import init, Fore, Style  # For enhancing terminal output with colors and styles for better readability

# Initialize colorama to ensure colored output works across different platforms
init(autoreset=True)

# Suppress runtime warnings to keep the output clean and focused on critical information
warnings.simplefilter("ignore", category=RuntimeWarning)

# Configure logging to record events, errors, and information to a log file with timestamps
logging.basicConfig(
    filename="llm_evaluation.log",  # The log file where messages will be stored
    level=logging.DEBUG,  # Log level set to INFO to capture general events and errors
    format="%(asctime)s - %(levelname)s - %(message)s"  # Include timestamp, log level, and message in each log entry
)

# Initialize the OpenAI client using an API key from environment variables, with error handling
try:
    api_key = os.getenv("OPENAI_API_KEY")  # Attempt to retrieve the API key from environment variables
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set")  # Raise an error if the API key is missing
    client = OpenAI(api_key=api_key)  # Initialize the OpenAI client with the retrieved API key
except Exception as e:
    logging.error(f"Failed to initialize OpenAI client: {e}")  # Log the error if initialization fails
    raise  # Re-raise the exception to halt execution if the client can't be initialized

# Load the SentenceTransformer model for computing semantic similarity between text embeddings
try:
    embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Load a pre-trained sentence transformer model
except Exception as e:
    logging.error(f"Failed to load SentenceTransformer model: {e}")  # Log the error if model loading fails
    raise  # Re-raise the exception to halt execution if the model can't be loaded

def fetch_response(prompt, context):
    """Fetches a response from the OpenAI API based on the provided prompt and context."""
    try:
        # Generate a response from the OpenAI API using the specified prompt and context
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Specify the OpenAI model to be used
            messages=[
                {"role": "system", "content": f"Context: {context}"},  # Set the system context for the model's behavior
                {"role": "user", "content": prompt}  # Provide the user's prompt to the model
            ],
            max_tokens=100  # Limit the response length to 100 tokens to control verbosity
        )
        return response.choices[0].message.content.strip()  # Return the model's response, stripped of any leading/trailing whitespace
    except Exception as e:
        logging.error(f"Error fetching response from OpenAI: {e}")  # Log the error if the response fetch fails
        raise  # Re-raise the exception to halt execution if the response can't be fetched

def semantic_similarity(actual_output, expected_outputs, threshold=0.4):
    """Calculates semantic similarity between the actual output and expected outputs."""
    try:
        # Log the number of expected outputs and the actual expected outputs before encoding
        logging.debug(f"Number of expected outputs: {len(expected_outputs)}")
        logging.debug(f"Expected outputs before encoding: {expected_outputs}")
        
        # Encode the actual output and expected outputs into embeddings for comparison
        actual_embedding = embedding_model.encode(actual_output, convert_to_tensor=True)
        expected_embeddings = embedding_model.encode(expected_outputs, convert_to_tensor=True)
        
        # Log the shape of the embeddings to verify their structure
        logging.debug(f"Actual embedding shape: {actual_embedding.shape}")
        logging.debug(f"Expected embeddings shape: {expected_embeddings.shape}")
        
        # Compute cosine similarity scores between the actual output and each expected output
        cosine_scores = util.pytorch_cos_sim(actual_embedding, expected_embeddings)

        # Log the shape of the cosine scores tensor to ensure it matches expectations
        logging.debug(f"Cosine scores tensor shape: {cosine_scores.shape}")

        # Return True if any of the similarity scores meet or exceed the threshold, along with the scores for reporting
        return any(score >= threshold for score in cosine_scores[0]), cosine_scores
    except Exception as e:
        logging.error(f"Error calculating semantic similarity: {e}")
        raise

class CustomLLMEvaluation:
    """A class that combines LLMTestCase and SentenceTransformer for a comprehensive evaluation of LLM outputs."""
    
    def __init__(self, prompt, context, threshold=0.4, use_dynamic_responses=False):
        self.prompt = prompt  # Store the input prompt for the evaluation
        self.context = context  # Store the context that guides the LLM's response
        self.threshold = threshold  # Store the similarity threshold for the semantic similarity check
        self.use_dynamic_responses = use_dynamic_responses  # Store the flag indicating whether to dynamically generate expected responses
        
        try:
            # Fetch the actual output from the model based on the provided prompt and context
            self.actual_output = fetch_response(self.prompt, self.context)
        except Exception as e:
            logging.error(f"Error fetching actual output: {e}")  # Log any error that occurs while fetching the actual output
            raise  # Re-raise the exception to halt execution if the actual output can't be fetched

        # Generate expected responses dynamically using the same LLM, or use predefined hardcoded responses
        try:
            if self.use_dynamic_responses:
                self.expected_responses = [
                    fetch_response(self.prompt, self.context),  # Fetch the first expected response dynamically
                    fetch_response(self.prompt, self.context),  # Fetch the second expected response dynamically
                ]
            else:
                self.expected_responses = [
                    "To get to the other side.",  # A common, straightforward response often associated with the joke
                    ("Because its dopaminergic neurons fired synchronously across the synapses of its caudate nucleus, "
                     "triggering motor contractions propelling the organism forward, to a goal predetermined by its hippocampal road mappings.")  # A more complex, scientific explanation
                ]
        except Exception as e:
            logging.error(f"Error generating expected responses: {e}")  # Log any error that occurs while generating expected responses
            raise  # Re-raise the exception to halt execution if expected responses can't be generated

        # Initialize an LLMTestCase with the provided inputs, actual output, and expected output
        try:
            self.llm_test_case = LLMTestCase(
                input=self.prompt,  # The prompt used for generating the response
                actual_output=self.actual_output,  # The actual output generated by the LLM
                expected_output=self.expected_responses[0]  # Use the first expected response for exact match checking
            )
        except Exception as e:
            logging.error(f"Error initializing LLMTestCase: {e}")  # Log any error that occurs during LLMTestCase initialization
            raise  # Re-raise the exception to halt execution if the LLMTestCase can't be initialized

    def run_evaluation(self):
        """Runs the evaluation using both semantic similarity and LLMTestCase."""
        try:
            # Check if the actual output matches the expected output exactly
            llm_test_passed = self.llm_test_case.actual_output.strip() == self.llm_test_case.expected_output.strip()
            
            # Check if the actual output is semantically similar to any of the expected outputs
            semantic_passed, cosine_scores = semantic_similarity(self.actual_output, self.expected_responses, self.threshold)

            # Determine whether the overall test passed based on either exact match or semantic similarity
            overall_passed = llm_test_passed or semantic_passed
            
            # Generate and print a detailed report of the evaluation
            self.generate_report(overall_passed, cosine_scores)
            return overall_passed
        except Exception as e:
            logging.error(f"Error during evaluation run: {e}")  # Log any error that occurs during the evaluation process
            raise  # Re-raise the exception to halt execution if the evaluation fails

    def generate_report(self, passed, cosine_scores):
        """Generates a detailed report of the evaluation results."""
        try:
            # Define styles and colors for the report to enhance readability
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

            # Prepare the detailed report with all the relevant information
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

            # Print the report to the console for immediate feedback
            print(report)
        except Exception as e:
            logging.error(f"Error generating report: {e}")  # Log any errors encountered during report generation
            raise  # Re-raise the exception to ensure any issues are properly flagged

# Configuration settings for running the evaluation
context = "Humor"  # Set the context to guide the LLM's response
use_dynamic_responses = True  # Enable dynamic generation of expected responses
threshold = 0.7  # Set the similarity threshold for semantic similarity testing
prompt = "Why did the chicken cross the road?"  # Define the prompt to be tested

# Run the evaluation
try:
    # Initialize the evaluation class with the specified parameters
    evaluation = CustomLLMEvaluation(prompt, context, threshold, use_dynamic_responses)
    # Execute the evaluation process
    evaluation.run_evaluation()
except Exception as e:
    logging.critical(f"Critical failure in running the evaluation: {e}")  # Log any critical failures during execution
    raise  # Re-raise the exception to halt execution in case of a critical error
