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

# Instructions to Run the Script from Scratch in a BASH Shell

## 1. Set Up Environment

Install Python if not already installed:

```bash
sudo apt-get update
sudo apt-get install python3 python3-pip
```

## 2. Install Required Python Packages

Install the necessary packages using pip:

```bash
pip3 install openai deepeval
```

## 3. Set OpenAI API Key

Export your OpenAI API key as an environment variable (please be secure with your keys, do not put them in code):

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## 4. Create the Script

Create a new Python file, paste the code, and save it as `deepeval_test.py`:

```bash
nano deepeval_test.py
```

## 5. Run the Script

Run the script using Python:

```bash
python3 deepeval_test.py
```

## 6. View Results

The script will print a test report with the input, expected output, actual output, and test result (Pass/Fail).

![Sample Report Output](https://github.com/jadm11/deepeval_test/blob/main/report.png)
Here is a screenshot of the report from the test case run from a shell in Terminator on Ubuntu. This is only an example, it's not pretty, but it works. 
