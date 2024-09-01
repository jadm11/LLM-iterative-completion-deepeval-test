Feature: Evaluate AI model responses

  ### This template provides a structured approach to creating test cases for AI model evaluation. It is not executable.
  
  As an engineer
  I want to evaluate the AI model's response against expected outputs
  So that I can ensure the model's outputs are both accurate and contextually relevant

  Background:
    Given the OpenAI API is initialized with a valid API key
    And the SentenceTransformer model "paraphrase-MiniLM-L6-v2" is loaded
    And logging is configured to capture any errors or issues

  Scenario Outline: Test model response with exact match and semantic similarity
    Given I have set the prompt "<prompt>"
    And I have set the context "<context>"
    And the dynamic responses option is "<use_dynamic_responses>"
    And the similarity threshold is set to "<threshold>"
    When I fetch the actual output from the AI model
    Then the actual output should be "<expected_output>"
    And the semantic similarity score between the actual and expected output should be above "<threshold>"
    And the test result should be "<result>"

    Examples:
      | prompt                                | context | use_dynamic_responses | threshold | expected_output                                                                                                     | result |
      | "Why did the chicken cross the road?"  | Humor   | True                  | 0.5       | "To get to the other side."                                                                                          | Pass   |
      | "What is the capital of France?"       | Geography | False               | 0.8       | "The capital of France is Paris."                                                                                    | Pass   |
      | "Explain quantum entanglement"         | Science | False                | 0.6       | "Quantum entanglement is a physical phenomenon where particles become interconnected, affecting each other’s state." | Pass   |
      | "Tell me a joke about computers"       | Humor   | True                  | 0.7       | "Why don’t programmers like nature? It has too many bugs."                                                            | Pass   |

  Scenario: Test model response failure due to low similarity
    Given I have set the prompt "Define the term 'artificial intelligence'"
    And I have set the context "Technology"
    And the dynamic responses option is "False"
    And the similarity threshold is set to "0.9"
    When I fetch the actual output from the AI model
    Then the actual output should not exactly match the expected output
    And the semantic similarity score between the actual and expected output should be below "0.9"
    And the test result should be "Fail"


## Explanation of the Template
# Feature: Describes the overall purpose of the test. 
# Background: Sets up the necessary preconditions.
# Scenario Outline: Provides a template for testing various prompts and contexts. The scenario outline allows you to define multiple test cases with 
# different inputs and expected outcomes.
# Examples: Lists specific test cases that can be run. You can add as many rows as needed to cover different scenarios.
# Scenario: Provides a specific case where the test is expected to fail due to low semantic similarity, ensuring that the model is properly evaluated 
# for scenarios where precision is critical.

## How to Use
# Modify the Examples Section: Add different prompts, contexts, and thresholds to create a comprehensive set of test cases.
# Adjust the Threshold: Set the threshold according to the desired strictness for each scenario.
# Run the Scenarios: Execute the test cases, and evaluate the model's performance based on both exact matches and semantic similarity scores.

