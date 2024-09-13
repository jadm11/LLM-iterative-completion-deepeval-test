Feature: Custom LLM Evaluation Script

  Background:
    Given the standard libraries are imported:
      | Library   |
      | os        |
      | warnings  |
      | logging   |
    And the third-party libraries are imported:
      | Library                                      |
      | openai.OpenAI                                |
      | deepeval.test_case.LLMTestCase               |
      | sentence_transformers.SentenceTransformer    |
      | sentence_transformers.util                   |
      | colorama.init                                |
      | colorama.Fore                                |
      | colorama.Style                               |
    And colorama is initialized with auto-reset enabled
    And warnings of category RuntimeWarning are suppressed
    And logging is configured with the following parameters:
      | Parameter | Value                                           |
      | filename  | "llm_evaluation.log"                            |
      | level     | logging.DEBUG                                   |
      | format    | "%(asctime)s - %(levelname)s - %(message)s"     |
    And the OpenAI API key is obtained from the environment variable "OPENAI_API_KEY"
    And the OpenAI client is initialized with the API key
    And the SentenceTransformer model "paraphrase-MiniLM-L6-v2" is loaded into embedding_model

  Scenario: Defining the fetch_response function
    Given a function fetch_response that accepts a prompt and context
    When the function calls client.chat.completions.create with the following parameters:
      | Parameter   | Value                                                                                                                      |
      | model       | "gpt-4o"                                                                                                                   |
      | messages    | [{"role": "system", "content": f"Context: {context}"}, {"role": "user", "content": prompt}]                                |
      | max_tokens  | 100                                                                                                                        |
    Then the function returns the stripped content of the first message in response.choices

  Scenario: Defining the semantic_similarity function
    Given a function semantic_similarity that accepts actual_output, expected_outputs, and an optional threshold (default is 0.4)
    When the function encodes actual_output and expected_outputs using embedding_model.encode with convert_to_tensor=True
    And calculates cosine similarity scores using util.pytorch_cos_sim(actual_embedding, expected_embeddings)
    Then the function returns a tuple containing a boolean indicating if all scores meet or exceed the threshold and the cosine_scores

  Scenario: Creating the CustomLLMEvaluation class
    Given a class CustomLLMEvaluation with an __init__ method that accepts prompt, context, threshold (default 0.4), and use_dynamic_responses (default False)
    When the class is initialized
    Then it sets self.prompt, self.context, self.threshold, and self.use_dynamic_responses
    And it calls self._fetch_actual_output to obtain self.actual_output
    And it calls self._generate_expected_responses to obtain self.expected_responses
    And it calls self._initialize_test_case to create self.llm_test_case

  Scenario: Implementing _fetch_actual_output method
    Given the method self._fetch_actual_output
    When called
    Then it returns the result of fetch_response with self.prompt and self.context

  Scenario: Implementing _generate_expected_responses method
    Given the method self._generate_expected_responses
    When self.use_dynamic_responses is True
    Then it returns a list containing two responses obtained by calling fetch_response with self.prompt and self.context twice
    Else
    Then it returns a list containing the predefined responses:
      | Expected Response                                                                                                                                                     |
      | "To get to the other side."                                                                                                                                           |
      | "Because its dopaminergic neurons fired synchronously across the synapses of its caudate nucleus, triggering motor contractions propelling the organism forward, to a goal predetermined by its hippocampal road mappings." |

  Scenario: Implementing _initialize_test_case method
    Given the method self._initialize_test_case
    When called
    Then it creates an instance of LLMTestCase with input=self.prompt, actual_output=self.actual_output, and expected_output=self.expected_responses[0]

  Scenario: Running the evaluation
    Given the method run_evaluation in CustomLLMEvaluation
    When called
    Then it checks if the LLM test passed by comparing stripped actual_output and expected_output
    And it calls semantic_similarity with actual_output, expected_responses, and threshold to obtain semantic_passed and cosine_scores
    And it determines overall_passed as the logical AND of llm_test_passed and semantic_passed
    And it calls self._generate_report with overall_passed and cosine_scores
    And it returns overall_passed

  Scenario: Generating the evaluation report
    Given the method _generate_report that accepts passed and cosine_scores
    When called
    Then it defines formatting styles using colorama constants
    And it constructs a report string containing:
      | Element                     | Content                                   |
      | Test Report Title           | Styled with BOLD and BLUE                 |
      | Separator Line              | Styled with DIM                           |
      | Context                     | Displayed with CYAN label and BOLD value  |
      | Dynamic Responses Enabled   | Displayed with CYAN label and BOLD value  |
      | Similarity Threshold        | Displayed with CYAN label and BOLD value  |
      | Cosine Scores               | Displayed with CYAN label and BOLD value  |
      | Input Prompt                | Displayed with CYAN label and BOLD value  |
      | Expected Responses          | Listed with indices                       |
      | Actual Output               | Displayed with CYAN label and BOLD value  |
      | Result                      | Displayed with BOLD and colored based on pass/fail |
    And it prints the report

  Scenario: Executing the evaluation script
    Given the variables are set:
      | Variable               | Value                                        |
      | context                | "Humor"                                      |
      | use_dynamic_responses  | True                                         |
      | threshold              | 0.5                                          |
      | prompt                 | "Why did the chicken cross the road?"        |
    When an instance of CustomLLMEvaluation is created with these variables
    And run_evaluation is called on the instance
    Then the evaluation is performed
    And the report is generated and displayed 