# DeepEval Test Script

## Overview

This script (`deepeval_test_semantic_similarity.py`) is designed to test and evaluate the responses of a Language Model (LLM) using the OpenAI API and custom semantic similarity comparisons.

## Features
**Similarity Threshold**

In this test, the similarity threshold determines how closely the model's output must match the expected responses in meaning. By adjusting this threshold (line 50), you can control the strictness of the test. A higher threshold (e.g., 0.8) requires very close matches, while a lower threshold (e.g., 0.7) allows for more variation in wording. Lowering the threshold might make the test pass when the outputs are similar in intent but differ in phrasing, ensuring meaningful yet flexible evaluation. Make adjustments and run the test noting the threshold and how a pass or fail can be determined with this change.

**Context**

To assess the model's adaptability, you can set the context (line 33) to different domains like "Scientific," "Historical," "Technical," or "Humorous," which tests how well the model handles various subject matters. For specialized fields, use contexts like "Customer Support," "Education," or "Medical" to evaluate performance. Additionally, by changing the context to reflect "Western," "Eastern," or "Global" viewpoints, you can examine the model’s ability to adapt to different cultural nuances. This approach helps you gain insights into the model's versatility and effectiveness across diverse scenarios.

**Dynamic vs Static Toggle**

This test includes a toggle (use_dynamic_responses) that controls how expected responses are generated:

* True: When set to True, the script dynamically generates expected responses using the OpenAI API based on the prompt and context. This allows you to evaluate the model's real-time adaptability and response generation.

* False: When set to False, the script uses predefined hardcoded responses. This enables a controlled test environment where the model's output is compared against specific, known answers.

Important: The SentenceTransformer model (which is used for semantic similarity) will be downloaded the first time this is run. This includes the model weights, configuration files, and tokenizer data. This download is necessary only the first time you use the model. Once downloaded, it will be cached locally, so subsequent runs should be faster.

## Script Overview

1. **Library Imports**
   * Imports essential libraries, including `OpenAI`, `os`, and `LLMTestCase` from `deepeval`.
   * Additionally, imports `SentenceTransformer` and `util` from `sentence_transformers` to handle semantic similarity comparisons.
   * Warnings are suppressed for cleaner output.

2. **Client Initialization**
   * The OpenAI client is initialized using an API key stored in an environment variable.

3. **Fetching Expected Responses**
   * The `fetch_response` function sends a prompt and context to the OpenAI API and retrieves model responses dynamically.

4. **Defining the Test Case**
   * A prompt ("Why did the chicken cross the road?") is defined, and multiple expected responses are fetched to be used in comparison.

5. **Simulating Model Output**
   * The same `fetch_response` function is used to simulate the model's actual response.

6. **Test Case Creation**
   * An `LLMTestCase` object is not explicitly created; instead, a custom similarity function (`semantic_similarity`) is defined to directly compare the actual and expected responses.

7. **Evaluation**
   * The custom `semantic_similarity` function compares the model's output to the expected responses using cosine similarity of embeddings.

8. **Reporting**
   * A simple report is generated, displaying the input, expected responses, actual output, and whether the test passed or failed.

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

Create a new Python file, paste the code, and save it as `deepeval_test_semantic_similarity.py`:

```bash
nano deepeval_test_semantic_similarity.py
```

## 5. Run the Script

Run the script using Python:

```bash
python3 deepeval_test_semantic_similarity.py
```

## 6. View Results

The script will print a test report with the input, expected output, actual output, and test result (Pass/Fail).

![Report Output](https://github.com/jadm11/deepeval_test/blob/main/report.png)
Here is a screenshot of the report from the test case run from a shell in Terminator on Ubuntu. This is only an example, it's not pretty, but it works. 
