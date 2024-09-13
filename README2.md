# LLM Semantic Similarity Testing Using Gherkin
## Other relevant tests would be needed

This repository demonstrates using Gherkin as a universal template to generate scripts in multiple programming languages. This setup enables testing semantic similarity across different Large Language Models (LLMs) within your development pipeline.

## Table of Contents

1. [Overview](#overview)
2. [Setup](#setup)
3. [Usage](#usage)
4. [Results Aggregation](#results-aggregation)
5. [CI/CD Integration](#cicd-integration)
6. [Benefits](#benefits)
7. [Considerations](#considerations)
8. [Conclusion](#conclusion)

---

## Overview

- **Gherkin as Template**: Defines script behavior in a language-agnostic manner.
- **SCoT Interpretation**: Generates consistent code across different programming languages by treating Gherkin as a Structured Chain of Thought.

## Setup

1. **Define Gherkin Scenarios**: Include language placeholders.

    ```gherkin
    Feature: Cross-Platform LLM Evaluation
    
      Scenario Outline: Evaluate LLM outputs in <Language>
        Given the environment is set up
        When the evaluation is run
        Then results are aggregated
    
      Examples:
        | Language   |
        | Python     |
        | Ruby       |
        | JavaScript |
    ```

2. **Prime the LLM**: Use the following prompt.

    ```
    You will interpret Gherkin as a Structured Chain of Thought for code generation. Generate code in <Language> based on the provided Gherkin scenario.
    ```

## Usage

1. **Automate Code Generation**:
    - Read Gherkin file.
    - Iterate over specified languages.
    - Generate and save code using the LLM.

2. **Set Up Execution Environments**:
    - Use Docker containers for Python, Ruby, and Node.js.
    - Install dependencies via `requirements.txt`, `Gemfile`, or `package.json`.

3. **Run Semantic Similarity Tests**:
    - Implement tests using language-specific libraries.

    **Python Example**:

    ```python
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def semantic_similarity(actual, expected, threshold=0.5):
        actual_emb = model.encode(actual, convert_to_tensor=True)
        expected_emb = model.encode(expected, convert_to_tensor=True)
        scores = util.cos_sim(actual_emb, expected_emb)
        return all(score >= threshold for score in scores[0])
    ```

## Results Aggregation

- **Standard Output**: Scripts output JSON files with results.
- **Aggregator Script**: Collects JSON results, compiles reports, and generates visualizations.

## CI/CD Integration

- **Pipeline Steps**:
    1. Generate scripts.
    2. Set up environments.
    3. Execute scripts.
    4. Collect and aggregate results.
    5. Generate reports.
    6. Cleanup.

- **Tools**: Jenkins, GitHub Actions, GitLab CI/CD.

---

## Benefits

- **Scalability**: Easily add new languages or platforms.
- **Consistency**: Uniform testing logic across environments.
- **Automation**: Streamlined code generation and testing processes.
- **Insights**: Comprehensive data on LLM performance.

## Considerations

- **LLM Variability**: Use deterministic settings (e.g., temperature=0).
- **Error Handling**: Ensure robust error checks in generated code.
- **Security**: Execute code in isolated environments to mitigate risks.

---

## Conclusion

Leveraging Gherkin as a central template facilitates the generation and deployment of scripts across multiple languages and platforms. This method enhances the testing of semantic similarity in various LLMs, enabling detailed result aggregation and continuous improvement within your development pipeline.

---
