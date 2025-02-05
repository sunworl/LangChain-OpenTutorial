# **RAG Source Citation Prompt**

## **Description**

- This prompt instructs the AI to function as a RAG system that generates answers enriched with explicit source citations. It guides the model to incorporate references—such as document IDs, verbatim text snippets, or both—based on the retrieved documents used in crafting the response. The prompt covers strategies for both pre- and post-processing in the retrieval and generation stages to ensure that all core evidence is properly attributed.

## **Relevant Document**

- [How to get a RAG application to add citations](https://python.langchain.com/docs/how_to/qa_citations/)
- [Build RAG with in-line citations](https://docs.llamaindex.ai/en/stable/examples/workflow/citation_query_engine/)
- **Context Document:** [Attention Is All You Need (arXiv:1706.03762v7)](https://arxiv.org/html/1706.03762v7)

## **Input**

- **SYSTEM:**
  - Contains detailed instructions for including citations from the provided source in the final answer.
  - Specifies formatting rules and structured output requirements (using methods such as Pydantic schemas, XML formatting, or JSON structures) to ensure clarity and traceability.
- **HUMAN:**
  - Provides the following variables:
    - {question}: [User’s query, e.g., "What is the core idea behind the Transformer architecture described in this paper?"].
    - {context}: [Retrieved document snippets or formatted content from https://arxiv.org/html/1706.03762v7, including source IDs and excerpts].
    - {additional_instructions}: [Optional further guidance on citation style or output format].

## **Output**

- The output must include:
  - A synthesized answer that directly references the source document.
  - Clear citations in the specified format (e.g., document IDs and/or in-line quotes) that justify each part of the answer.
- The answer should adhere to one of the following citation strategies:
  1. **Direct Prompting:** Instructing the model to generate structured responses (such as XML or JSON) containing citation details.
  2. **Retrieval Post-Processing:** Compressing and filtering retrieved content to highlight only the most relevant portions, thereby implicitly defining citation boundaries.
  3. **Generation Post-Processing:** Generating an initial answer and then re-prompting for an annotated version that includes citations.

## **Additional Information**

- **Citation Methods Overview:**
  1. **Direct Prompting:**
     - Use explicit instructions within the system prompt (for example, by defining an XML or JSON format) to have the model generate output with citations.
  2. **Retrieval Post-Processing:**
     - Employ text splitting (via tools like `RecursiveCharacterTextSplitter`) and embedding filters to compress documents, making the source content minimal yet sufficient for citation.
  3. **Generation Post-Processing:**
     - First generate an answer and then reissue a prompt that asks the model to annotate its answer with precise citations.

## **Examples**

1. **QUESTION:** What is the core idea behind the Transformer architecture described in this paper?  
   **ANSWER:** The core idea behind the Transformer architecture is to utilize a model that relies entirely on attention mechanisms, specifically self-attention, to draw global dependencies between input and output sequences, without using recurrence or convolutions. This design allows for significant parallelization during training and leads to improved performance in tasks such as machine translation. The Transformer architecture consists of stacked self-attention and feed-forward layers in both the encoder and decoder, enabling it to achieve state-of-the-art results in translation quality while being more efficient in terms of training time compared to traditional recurrent or convolutional models [Source 4, Source 65].

2. **QUESTION:** How does the Transformer model handle sequential data differently from traditional RNNs?  
   **ANSWER:** The Transformer model handles sequential data differently from traditional RNNs by completely eliminating recurrence and instead relying solely on attention mechanisms. This allows the Transformer to process all input positions simultaneously, enabling significant parallelization during training. In contrast, RNNs process sequences in a stepwise manner, which inherently limits their ability to parallelize computations and can lead to longer training times, especially with longer sequences [Source 4, Source 8, Source 7].\n\nAdditionally, the self-attention mechanism in the Transformer allows it to model dependencies between all positions in the input sequence regardless of their distance, which is a challenge for RNNs that struggle with long-range dependencies due to their sequential nature [Source 8, Source 40]. The computational complexity of self-attention is also more favorable, as it connects all positions with a constant number of operations, while RNNs require a linear number of operations relative to the sequence length [Source 43].\n\n\nThus, the key differences are:\n1. **Parallelization**: The Transformer allows for parallel processing of input sequences, while RNNs are sequential.\n2. **Dependency Modeling**: The Transformer can model long-range dependencies more effectively through self-attention, whereas RNNs face challenges with this due to their sequential processing [Source 4, Source 8, Source 40].
