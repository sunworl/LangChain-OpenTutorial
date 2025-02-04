# **Advanced Relevance Evaluator**

## **Description**
- This prompt instructs an advanced evaluator (judge) to assess an LLM's response based on a provided question and context. The evaluation is performed using a detailed grading rubric that considers accuracy, comprehensiveness, and context precision.

## **Relevent Document**
- [16-evaluations/07-langsmith-custom-llm-evaluation](https://langchain-opentutorial.gitbook.io/langchain-opentutorial/16-evaluations/07-langsmith-custom-llm-evaluation)

## **Input**
- SYSTMEM: A detailed context containing the question, the LLM's response, and any relevant background information.
- HUMAN: The evaluator's task to score the response according to the provided grading criteria.

## **Output**
- The output must consist solely of numerical scores formatted as follows:
  ```
  Accuracy: X
  Comprehensiveness: Y
  Context Precision: Z
  Final: W
  ```
  where:
  - Each individual score (Accuracy, Comprehensiveness, Context Precision) is an integer between 0 and 10.
  - The Final score is calculated as `(Accuracy + Comprehensiveness + Context Precision) / 30` and rounded to ONE decimal place.

## **Tool**
- **Tool1**: Evaluation Calculator
  ```Argument type
  [Example Input: Accuracy: 8, Comprehensiveness: 7, Context Precision: 9]
  ```
- **Tool2**: Context Analyzer
  ```Argument type
  [Enter example input here if needed]
  ```

## **Additional Information**
- **Grading Criteria**:
  - **Accuracy (0-10 points)**: Evaluate how well the response aligns with the provided context.
    - **0 points**: The answer is completely inaccurate or contradicts the provided context.
    - **1-3 points**: The answer contains only a few correct points amidst significant inaccuracies.
    - **4-5 points**: The answer is generally aligned but misses 1-2 crucial facts.
    - **6-7 points**: The answer is mostly correct with only minor factual errors.
    - **8 points**: The answer is largely accurate with only very slight inaccuracies.
    - **9-10 points**: The answer fully aligns with the provided context and is completely accurate.
  - **Comprehensiveness (0-10 points)**: Assess how thoroughly the response covers the relevant aspects of the question.
    - **0 points**: The answer is completely inadequate or irrelevant.
    - **1-3 points**: The answer covers very few necessary details.
    - **4-5 points**: The answer covers the main points but omits several important details.
    - **6-7 points**: The answer covers most key aspects, with only minor details missing.
    - **8 points**: The answer is detailed and covers nearly all aspects.
    - **9-10 points**: The answer is fully comprehensive, addressing every necessary detail in depth.
  - **Context Precision (0-10 points)**: Measure how precisely the response utilizes the provided context information.
    - **0 points**: The response does not incorporate any context information or completely misinterprets it.
    - **1-3 points**: The response references context, but with significant errors.
    - **4-5 points**: The response uses context information adequately but has minor inaccuracies.
    - **6-7 points**: The response correctly uses most context information, with only minor misinterpretations.
    - **8 points**: The response demonstrates high precision in using the context.
    - **9-10 points**: The response perfectly and accurately utilizes all relevant context information.
- **Additional Guidelines**:
  - If the question is ambiguous or the provided context is very limited, adjust the scores accordingly.
  - If no context is provided, set **Context Precision** to 0.
  - If **Accuracy** is between 0 and 2, the maximum achievable scores for **Comprehensiveness** and **Context Precision** are capped at 4 points each.
  - If the answer is completely unrelated to the question or context, assign 0 to all criteria.