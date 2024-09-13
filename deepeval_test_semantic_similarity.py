# Standard Library Imports
import os
import warnings
import logging

# Third-Party Imports
from openai import OpenAI
from deepeval.test_case import LLMTestCase
from sentence_transformers import SentenceTransformer, util
from colorama import init, Fore, Style

# Initialize colorama and suppress warnings
init(autoreset=True)
warnings.simplefilter("ignore", category=RuntimeWarning)

# Configure logging
logging.basicConfig(
    filename="llm_evaluation.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize OpenAI client
try:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set")
    client = OpenAI(api_key=api_key)
except Exception as e:
    logging.error(f"Failed to initialize OpenAI client: {e}")
    raise

# Load SentenceTransformer model
try:
    embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
except Exception as e:
    logging.error(f"Failed to load SentenceTransformer model: {e}")
    raise

def fetch_response(prompt, context):
    """Fetch a response from the OpenAI API based on the prompt and context."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"Context: {context}"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error fetching response from OpenAI: {e}")
        raise

def semantic_similarity(actual_output, expected_outputs, threshold=0.4):
    """Calculate semantic similarity and ensure all scores meet or exceed the threshold."""
    try:
        actual_embedding = embedding_model.encode(actual_output, convert_to_tensor=True)
        expected_embeddings = embedding_model.encode(expected_outputs, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(actual_embedding, expected_embeddings)
        return all(score >= threshold for score in cosine_scores[0]), cosine_scores
    except Exception as e:
        logging.error(f"Error calculating semantic similarity: {e}")
        raise

class CustomLLMEvaluation:
    """Evaluates LLM outputs using LLMTestCase and semantic similarity checks."""
    
    def __init__(self, prompt, context, threshold=0.4, use_dynamic_responses=False):
        self.prompt = prompt
        self.context = context
        self.threshold = threshold
        self.use_dynamic_responses = use_dynamic_responses
        
        self.actual_output = self._fetch_actual_output()
        self.expected_responses = self._generate_expected_responses()
        self.llm_test_case = self._initialize_test_case()

    def _fetch_actual_output(self):
        return fetch_response(self.prompt, self.context)

    def _generate_expected_responses(self):
        if self.use_dynamic_responses:
            return [
                fetch_response(self.prompt, self.context),
                fetch_response(self.prompt, self.context)
            ]
        return [
            "To get to the other side.",
            ("Because its dopaminergic neurons fired synchronously across the synapses "
             "of its caudate nucleus, triggering motor contractions propelling the "
             "organism forward, to a goal predetermined by its hippocampal road mappings.")
        ]

    def _initialize_test_case(self):
        return LLMTestCase(
            input=self.prompt,
            actual_output=self.actual_output,
            expected_output=self.expected_responses[0]
        )

    def run_evaluation(self):
        try:
            llm_test_passed = self.llm_test_case.actual_output.strip() == self.llm_test_case.expected_output.strip()
            semantic_passed, cosine_scores = semantic_similarity(self.actual_output, self.expected_responses, self.threshold)
            overall_passed = llm_test_passed and semantic_passed
            self._generate_report(overall_passed, cosine_scores)
            return overall_passed
        except Exception as e:
            logging.error(f"Error during evaluation run: {e}")
            raise

    def _generate_report(self, passed, cosine_scores):
        try:
            BOLD = Style.BRIGHT
            GREEN = Fore.GREEN
            RED = Fore.RED
            BLUE = Fore.BLUE
            CYAN = Fore.CYAN
            RESET = Style.RESET_ALL
            SEPARATOR = f"{Style.DIM}{'-' * 50}{RESET}"

            result_color = GREEN if passed else RED
            report = (
                f"{BOLD}{BLUE}Test Report{RESET}\n{SEPARATOR}\n"
                f"{CYAN}Context:{RESET} {BOLD}{self.context}{RESET}\n"
                f"{CYAN}Dynamic Responses Enabled:{RESET} {BOLD}{self.use_dynamic_responses}{RESET}\n"
                f"{CYAN}Similarity Threshold:{RESET} {BOLD}{self.threshold}{RESET}\n"
                f"{CYAN}Cosine Scores:{RESET} {BOLD}{cosine_scores}{RESET}\n\n"
                f"{CYAN}Input:{RESET} {BOLD}{self.prompt}{RESET}\n\n"
                f"{CYAN}Expected Responses:{RESET}\n"
                f"  1. {self.expected_responses[0]}\n"
                f"  2. {self.expected_responses[1] if len(self.expected_responses) > 1 else ''}\n\n"
                f"{CYAN}Actual Output:{RESET}\n  {BOLD}{self.actual_output}{RESET}\n"
                f"{SEPARATOR}\n{BOLD}Result: {result_color}{'✔ Pass' if passed else '✘ Fail'}{RESET}\n"
            )
            print(report)
        except Exception as e:
            logging.error(f"Error generating report: {e}")
            raise

# Configuration and Evaluation
context = "Science"
use_dynamic_responses = False
threshold = 0.5
prompt = "Why did the chicken cross the road?"

try:
    evaluation = CustomLLMEvaluation(prompt, context, threshold, use_dynamic_responses)
    evaluation.run_evaluation()
except Exception as e:
    logging.critical(f"Critical failure in running the evaluation: {e}")
    raise
