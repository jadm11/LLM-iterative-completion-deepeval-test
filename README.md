# DeepEval Test Script

## Overview
For the article see: https://www.linkedin.com/pulse/testing-ai-models-example-iterative-completion-jacob-adm-yhvpe

This script (`deepeval_test_semantic_similarity.py`) is designed to test and evaluate the responses of a Language Model (LLM) using the OpenAI API and custom semantic similarity comparisons.

## Features
**Similarity Threshold**

The semantic_similarity function compares the model’s output to expected responses using cosine similarity. The threshold, from 0.0 (no match) to 1.0 (perfect match), controls how strictly the outputs are compared. A higher threshold (e.g., 0.8) demands close alignment, while a lower threshold (e.g., 0.7) allows more variation. Adjusting the threshold lets you balance accuracy and flexibility, defining what passes or fails.

**Context**

To assess the model's adaptability, you can set the context to different domains like "Scientific," "Historical," "Technical," or "Humorous," which tests how well the model handles various subject matters. For specialized fields, use contexts like "Customer Support," "Education," or "Medical" to evaluate performance. Additionally, by changing the context to reflect "Western," "Eastern," or "Global" viewpoints, you can examine the model’s ability to adapt to different cultural nuances. This approach helps you gain insights into the model's versatility and effectiveness across diverse scenarios.

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

* Context: Provides the scenario or subject area guiding the model’s response, offering insight into the perspective or background used.
* Dynamic Responses Enabled: Indicates whether the expected outputs were dynamically generated during the test (True) or predefined (False).
* Similarity Threshold: Defines the level of precision required for the model’s output to match the expected responses. A higher threshold demands closer alignment in meaning, while a lower one allows for more flexibility.
* Input: The exact prompt provided to the model, serving as the starting point of the test.
* Expected Responses: The set of responses against which the model’s actual output is compared, either dynamically generated or fixed.
* Actual Output: The model’s response to the input prompt, compared to the expected responses in terms of meaning and relevance.
* Result: Indicates whether the model’s output met the required criteria, with ✔ Pass meaning it did and ✘ Fail meaning it didn’t.

![Report Output](https://github.com/jadm11/deepeval_test/blob/development/report.png)
Report 1
![Report Output](https://github.com/jadm11/deepeval_test/blob/development/report-2.png)
Report 2
