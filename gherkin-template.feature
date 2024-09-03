Feature: Custom LLM Evaluation
  # This feature file is designed to help evaluate LLM responses by checking both exact matches and semantic similarity.
  # It serves as a template to generate test cases for various scenarios involving LLM evaluations for this tool.

  # Instructions:
  # 1. **Setup Environment with Background:**
  #    - Ensure that the environment variable "OPENAI_API_KEY" is set with a valid API key.
  #    - Initialize the necessary libraries such as colorama, OpenAI, SentenceTransformer, and logging.
  #    - The Background section automates this setup.

  # 2. **Customize Scenario Outline:**
  #    - Use the Scenario Outline to define multiple test cases by modifying the Examples table.
  #    - Populate the table with different prompts, contexts, thresholds, and expected outcomes.
  #    - The example values provided can be replaced with specific test cases relevant to your testing needs.

  # 3. **Handling Dynamic Responses:**
  #    - The dynamic responses option controls whether the expected responses are generated dynamically.
  #    - Set "use_dynamic_responses" to "True" or "False" depending on whether you want to use dynamic or static responses.

  # 4. **Adjusting Thresholds:**
  #    - Modify the "threshold" value to set the minimum semantic similarity score required for a pass.
  #    - Lower thresholds are more lenient, while higher thresholds require more precise matches.

  # 5. **Running the Test Cases:**
  #    - Execute the test cases defined in the Examples table.
  #    - The evaluation results are generated based on exact matches and semantic similarity checks.
  #    - Review the generated report to determine if the test passed or failed.

  # 6. **Exception Handling:**
  #    - The "Handle exceptions during evaluation" scenario demonstrates how to manage cases where initialization fails.
  #    - This scenario ensures that errors are logged, and the process terminates if a critical failure occurs.

  # 7. **Customizing Failure Scenarios:**
  #    - Use the "Generate report with failure due to low semantic similarity" scenario to test failure cases.
  #    - Modify the prompt, context, and threshold to simulate situations where the semantic similarity is too low.

  # 8. **Extending the Template:**
  #    - Add additional scenarios as needed to cover more complex or specific cases.
  #    - Consider including scenarios that test edge cases, like handling large inputs or varying contexts.

  As an engineer,
  I want to evaluate LLM responses against expected outputs using exact matches and semantic similarity,
  So that I can ensure the model's outputs are both accurate and contextually relevant.

  Background:
    Given the OpenAI API is initialized with a valid API key
    And the SentenceTransformer model "paraphrase-MiniLM-L6-v2" is loaded
    And warnings of type "RuntimeWarning" are ignored
    And logging is configured to file "llm_evaluation.log" with level "DEBUG" and format "%(asctime)s - %(levelname)s - %(message)s"

  Scenario Outline: Test model response with exact match and semantic similarity
    Given I have set the prompt "<prompt>"
    And I have set the context "<context>"
    And the dynamic responses option is "<use_dynamic_responses>"
    And the similarity threshold is set to "<threshold>"
    When I fetch the actual output from the AI model
    Then the actual output should be "<expected_output_1>"
    And the semantic similarity score between the actual output and the expected responses should be above "<threshold>"
    And the evaluation should generate a report with context "<context>", dynamic responses "<use_dynamic_responses>", and cosine scores "<cosine_scores>"
    And the test result should be "<result>"

    Examples:
      | prompt                               | context | use_dynamic_responses | threshold | expected_output_1                                                                                                    | expected_output_2 | cosine_scores | result |
      | "Why did the chicken cross the road?" | Humor   | True                  | 0.5       | "To get to the other side."                                                                                          | "Because its dopaminergic neurons fired synchronously across the synapses of its caudate nucleus, triggering motor contractions propelling the organism forward, to a goal predetermined by its hippocampal road mappings." | [<cosine_score_1>, <cosine_score_2>] | Pass   |
      | "Explain quantum entanglement"        | Science | False                 | 0.6       | "Quantum entanglement is a physical phenomenon where particles become interconnected, affecting each other’s state." | ""                | [<cosine_score_1>] | Pass   |
      | "Tell me a joke about computers"      | Humor   | True                  | 0.7       | "Why don’t programmers like nature? It has too many bugs."                                                            | "Computers and nature don't mix well, bugs everywhere!" | [<cosine_score_1>, <cosine_score_2>] | Pass   |

  Scenario: Test model response failure due to low similarity
    Given I have set the prompt "Define the term 'artificial intelligence'"
    And I have set the context "Technology"
    And the dynamic responses option is "False"
    And the similarity threshold is set to "0.9"
    When I fetch the actual output from the AI model
    Then the actual output should not exactly match the expected output
    And the semantic similarity score between the actual and expected output should be below "0.9"
    And the evaluation should generate a report with context "Technology", dynamic responses "False", and cosine scores "<cosine_scores>"
    And the test result should be "Fail"

  Scenario: Handle exceptions during evaluation
    Given the OpenAI API key is invalid or missing
    When I attempt to initialize the OpenAI client
    Then an error should be logged with the message "Failed to initialize OpenAI client"
    And the process should be terminated with a critical failure
    And the evaluation should not proceed further
