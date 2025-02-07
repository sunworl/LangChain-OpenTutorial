# **RAG Citation Prompt**

## **Description**

This prompt instructs the AI to function as a RAG (Retrieval-Augmented Generation) system that generates answers enriched with explicit source citations. The model is guided to incorporate references—such as document IDs, verbatim text snippets, or both—based on the retrieved documents used in crafting the response. The prompt covers strategies for both pre-processing and post-processing in the retrieval and generation stages, ensuring that all core evidence is properly attributed.

## **Relevant Document**

- [How to get a RAG application to add citations](https://python.langchain.com/docs/how_to/qa_citations/)
- [Build RAG with in-line citations](https://docs.llamaindex.ai/en/stable/examples/workflow/citation_query_engine/)
- **Context Document:** [Attention Is All You Need (arXiv:1706.03762v7)](https://arxiv.org/html/1706.03762v7)

## **Input**

- **SYSTEM:**
  - Provides detailed instructions to include citations from the retrieved source(s) in the final answer.
  - Specifies formatting rules and structured output requirements (using methods such as Pydantic schemas, XML formatting, or JSON structures) to ensure clarity, traceability, and verifiability of the sources.
- **HUMAN:**
  - Supplies the following variables:
    - **{question}:** The user’s query (e.g., "What is the core idea behind the Transformer architecture described in this paper?").
    - **{context}:** Retrieved document snippets or formatted content from the specified sources (e.g., content from [Attention Is All You Need](https://arxiv.org/html/1706.03762v7)), including source IDs and text excerpts.
    - **{additional_instructions}:** (Optional) Further guidance on citation style or output format.

## **Output**

- The output must include:
  - A synthesized answer that directly references the source documents.
  - Clear and explicit citations in the specified format (e.g., `[Source X, Source Y]`) that indicate which document snippets support each part of the answer.
- The answer should adhere to one of the following citation strategies:
  1. **Direct Prompting:** Instructing the model to generate structured responses (such as XML or JSON) containing detailed citation information.
  2. **Retrieval Post-Processing:** Compressing and filtering retrieved content to highlight only the most relevant portions, thereby implicitly defining citation boundaries.
  3. **Generation Post-Processing:** Generating an initial answer and then re-prompting for an annotated version that includes precise citations.

## **Additional Information**

- **Citation Methods Overview:**
  1. **Direct Prompting:**  
     Use explicit instructions within the system prompt (for example, by defining an XML or JSON format) so that the model generates output that includes citation details.
  2. **Retrieval Post-Processing:**  
     Employ techniques such as text splitting (using tools like `RecursiveCharacterTextSplitter`) and embedding filters to compress documents, ensuring that only the most pertinent content is used for citation.
  3. **Generation Post-Processing:**  
     First generate an answer and then reissue a prompt asking the model to annotate its answer with precise citations.
- These methods ensure that every part of the answer is traceable back to the original source, enhancing both the credibility and the transparency of the generated content.

## **Examples**

1. **QUESTION:**  
   What is the core idea behind the Transformer architecture described in this paper?

   **ANSWER:**  
   The core idea behind the Transformer architecture is to utilize a model that relies entirely on attention mechanisms, specifically self-attention, to draw global dependencies between input and output sequences—without using recurrence or convolutions. This design allows for significant parallelization during training and leads to improved performance in tasks such as machine translation. The Transformer architecture consists of stacked self-attention and feed-forward layers in both the encoder and decoder, enabling it to achieve state-of-the-art results in translation quality while being more efficient in training time compared to traditional recurrent or convolutional models [Source 4, Source 65].

2. **QUESTION:**  
   How does the Transformer model handle sequential data differently from traditional RNNs?

   **ANSWER:**  
   The Transformer model handles sequential data differently by completely eliminating recurrence and instead relying solely on attention mechanisms. This allows the Transformer to process all input positions simultaneously, enabling significant parallelization during training. In contrast, RNNs process sequences sequentially, which limits parallel computation and can lead to longer training times, especially for lengthy sequences [Source 4, Source 8, Source 7].

   Additionally, the self-attention mechanism in the Transformer models dependencies between all positions in the input sequence regardless of their distance. This is particularly beneficial compared to RNNs, which often struggle with long-range dependencies due to their sequential nature [Source 8, Source 40]. Moreover, the computational complexity of self-attention is more favorable, as it connects all positions with a fixed number of operations, while RNNs require operations proportional to the sequence length [Source 43].

   **Key Differences:**

   1. **Parallelization:** The Transformer allows for parallel processing of input sequences, whereas RNNs process them sequentially.
   2. **Dependency Modeling:** The Transformer can model long-range dependencies more effectively through self-attention, while RNNs face challenges with this due to their sequential processing [Source 4, Source 8, Source 40].
