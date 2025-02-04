# **Relevance Evaluator**

## **Description**
- This prompt instructs an evaluator to determine whether a given document is relevant to the {target_variable} domain.
- It establishes specific criteria to assess if at least 80% of the document's content pertains to {target_variable}, handles short documents, ensures contextual depth, checks for duplicate content, and considers the impact of conflicting or outdated information.

## **Relevent Document**
- [16-evaluations/10-langsmith-summary-evaluation](https://langchain-opentutorial.gitbook.io/langchain-opentutorial/16-evaluations/10-langsmith-summary-evaluation)

## **Input**
- SYSTMEM: A document or textual content that needs to be evaluated for its relevance to the {target_variable} domain.
- HUMAN: A request to analyze the document according to the provided relevance criteria.

## **Output**
- A single string: `"yes"` or `"no"` (in lowercase only), with no additional text.
- The output indicates whether the document is sufficiently relevant based on the evaluation criteria.

## **Tool**
- **Relevance Evaluator**: Applies the following criteria to determine relevance:
  ```Text input example:
  "Insert the document text here for evaluation..."
  ```

## **Additional Information**
- **Relevance Criteria**:
  1. **80% Rule**  
     - A document is evaluated as "yes" if at least 80% of its content covers {target_variable} (including problem, solution, analysis, etc.).
  2. **Short Document Exception**  
     - For texts under 300 words, a single substantial paragraph on {target_variable} qualifies the document as "yes."
  3. **Contextual Depth**  
     - Mere mention of keywords results in "no." Only synonyms or related concepts that reflect a genuine understanding of {target_variable} qualify for "yes."
  4. **Duplicates**  
     - If the document is >=85% identical to a previously judged document, reuse that previous judgment.
  5. **Conflicting/Outdated Information**  
     - Minor errors or outdated data still result in "yes" if the content broadly aligns with {target_variable}.
     - Major factual contradictions or irrelevant details result in "no."