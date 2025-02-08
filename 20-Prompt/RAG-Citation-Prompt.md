# RAG Citation Prompt

## Description

This prompt instructs a Retrieval-Augmented Generation (RAG) application to generate detailed responses with in-line citations based on content retrieved from external documents. Every response must include in-line citation markers formatted as footnotes—with each citation providing the complete source URL and relevant metadata. The final output is a Markdown document featuring a consolidated **References** section that lists all citations sequentially.[^1][^2]

## Relevant Document

- [How to get a RAG application to add citations](https://python.langchain.com/docs/how_to/qa_citations/)
- [Build RAG with in-line citations](https://docs.llamaindex.ai/en/stable/examples/workflow/citation_query_engine/)
- **Context Document:** [Attention Is All You Need (arXiv:1706.03762v7)](https://arxiv.org/html/1706.03762v7)

## Input

- **SYSTEM:**  
  "You are a professional research assistant. Generate a detailed answer to the user's query based solely on the provided document. Your response must include in-line citations in footnote format. Use the content loaded from the provided web source and ensure every citation contains the full URL."

- **HUMAN:**  
  "{input}" (User's query)

## Output

The final output will be a Markdown document that includes:

- A detailed answer with in-line citations (e.g., [^1], [^2], etc.) that reference the source documents.
- A consolidated **References** section at the end where each citation is defined using the format:
  - [^n]: [Short source description](full_source_url)

## Tools

- **WebBaseLoader:**
  - **Description:** Retrieves document content from a specified URL. For this prompt, it loads the article from "https://arxiv.org/html/1706.03762v7" and extracts both the textual content and associated metadata.
- **Additional Processing Tools:**  
  Tools for text splitting, embeddings-based filtering, and structured output (e.g., using data models for annotations) can be employed to refine or post-process the response and citation annotations.

## Additional Information

- **Footnotes:**
  - [^1]: "Citation" must include the complete URL along with any additional metadata, such as the document title or page number.
  - [^2]: All URLs in citations must be fully qualified (e.g., starting with https://).
- **Guidelines:**
  - The final **References** section must be a single, merged list at the end of the document.
  - The prompt and output must strictly adhere to the defined formatting rules to maintain clarity and consistency.

## Examples

Below are illustrative scenarios that demonstrate how the RAG Citation Prompt can be applied.

### Example 1: Basic RAG Chain

In this scenario, the application:

- Uses a web-based document loader to fetch the content of the "Attention Is All You Need" paper.
- Concatenates the loaded content to create the context for the language model.
- Instructs the model to answer the query, "How does the Transformer model handle sequential data differently from traditional RNNs?"
- Produces a detailed answer in Markdown that includes bullet-pointed explanations. Each bullet point features in-line citation markers (e.g., [^1]) that reference the source document.
- Concludes with a single **References** section where each citation is listed with its full URL.

### Example 2: Structured Output with Citation Identifiers

This example illustrates how the prompt can be extended to generate structured output:

- The document content is pre-formatted by assigning source IDs to different segments.
- The prompt instructs the language model to produce an answer that incorporates these source IDs.
- The final response is a structured answer that includes both the main content and a separate list of citation identifiers.
- Despite the structured format, the answer maintains the required in-line footnote citations and a unified **References** section.

### Example 3: XML-Based Output for Citations

In this case, the prompt directs the model to return its response in an XML format:

- The XML output contains an `<answer>` element that holds the generated text.
- A `<citations>` element is included, which comprises multiple `<citation>` sub-elements. Each sub-element provides details such as a source identifier and an excerpt or quote from the source.
- The XML response is subsequently parsed and reformatted into Markdown, ensuring that the in-line citations and consolidated **References** section conform to the prescribed format.

### Example 4: Post-Processing with Retrieval and Filtering

Here, additional pre-processing steps are applied before invoking the language model:

- The document is split into manageable chunks using a text splitting mechanism.
- An embeddings-based filtering tool selects the most relevant chunks based on the user’s query.
- The filtered content forms the context for the prompt, and the language model generates a post-processed summary.
- The resulting answer clearly highlights how the Transformer model differs from RNNs, with in-line citations (e.g., [^1]) included throughout the text.

### Example 5: Annotation and Citation Enhancement

This example demonstrates a two-step process for enhancing the generated answer:

- **Step 1:** The language model produces an initial detailed answer based on the full document content.
- **Step 2:** A subsequent annotation process enriches the answer by adding citation details, such as exact quotes and source IDs.
- The final output merges the annotated information with the in-line citation markers and concludes with a consolidated **References** section that lists each citation in the required format.

---

## References

- [^1]: How to get a RAG application to add citations (https://python.langchain.com/docs/how_to/qa_citations/)
- [^2]: Build RAG with in-line citations (https://docs.llamaindex.ai/en/stable/examples/workflow/citation_query_engine/)
- [^3]: Attention Is All You Need (arXiv:1706.03762v7) (https://arxiv.org/html/1706.03762v7)
