# DeepEval Test Script

## Overview

This script evaluates an AI model's response to a specific prompt by comparing it against predefined expected outcomes.

## Features

### 1. Library Imports
- Imports necessary modules, including `OpenAI`, `os`, and `LLMTestCase` from `deepeval`.
- Suppresses warnings to keep the output clean.

### 2. Client Initialization
- Initializes the OpenAI client using an API key stored in an environment variable.

### 3. Fetching Expected Responses
- Uses `fetch_expected_responses` to call the OpenAI API with a prompt and context, returning model outputs.

### 4. Defining the Test Case
- Defines a prompt and manually adds additional expected responses for comprehensive testing.

### 5. Simulating Model Output
- Simulates the model's response.

### 6. Test Case Creation
- Creates an `LLMTestCase` object encapsulating all necessary information for evaluation.

### 7. Evaluation
- Manually checks if the model’s output matches any of the expected responses.

### 8. Reporting
- Generates a simple report showing the input, expected output, actual output, and the test result (Pass/Fail).


