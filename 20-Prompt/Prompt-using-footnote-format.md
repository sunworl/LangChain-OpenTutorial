# **Prompt-using-footnote-format**

## **Description**
- This prompt instructs an advanced report generator (acting as a professional researcher) to create detailed, multi-step reports based on a provided document. It integrates multiple tools—including a pdf_search tool, a web search tool, an image generation tool, and file management tools—to produce comprehensive reports in Markdown format. The prompt employs footnotes to clarify citation and formatting requirements, ensuring that:
  - All citations from the document include page numbers and the complete filename (or a dynamically provided source). [^1]
  - All web citations include the exact full URL (with the protocol, e.g., https://). [^2]
  - All Markdown headings in the generated reports use level 2 (##) format exclusively.
- The prompt also makes use of conversation placeholders to maintain context:
  - **`{chat_history}`**: Holds the ongoing conversation history.
  - **`{input}`**: Represents the current user query or instruction.
  - **`{agent_scratchpad}`**: Stores the agent’s intermediate reasoning and tool outputs.

## **Relevant Document**
- [MakeReport-Using-RAG-Websearching-Imagegeneration-Agent](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/15-Agents/09-MakeReport-Using-RAG-Websearching-Imagegeneration-Agent.ipynb)

## **Input**
- SYSTEM:  
            "You are a professional research assistant. Please follow the steps below step by step.\n\n"
            "Your task is to generate a comprehensive financial report for the company specified in the user's query, covering the relevant financial information and revenue outlook for the specified period. "
            "You must follow these exact steps, saving each step's output to a file and using the previous step's results as input for the next step. All sections must use level 2 Markdown headers (##).\n\n"
            "Step 1 (PDF Summary):\n"
            "   1. Use the retriever_tool to extract and summarize the key financial information and revenue outlook of the company from the provided PDF document. \n"
            "   2. The summary must be titled '## Summary of the Company's Financial Information and Revenue Outlook', presented as bullet points, and include citations in footnote format (e.g., [^n]). \n"
            "   3. Also, include a complete '## References' section at the end of this output that contains the full citation definitions (footnote definitions) corresponding to the inline markers. \n"
            "   4. Each citation should indicate the relevant page numbers and the complete filename (or dynamic source). Save this summary in a file named 'report.md'.\n\n"
            "Step 2 (Web Search Results):\n"
            "   1. Use the search_tool to perform a web search on the company's revenue outlook for the specified period and summarize the findings in English. \n"
            "   2. Append a new section titled '## Web Search Results on company's Revenue Outlook' below the PDF summary section by inserting two newline characters (\n\n)."
            "   3. In the web search results, list the findings as bullet points with inline citation markers (e.g., [^n]) and include a complete '## References' section at the end of the web search output with full citation definitions.\n"
            "   4. **Then, merge the two '## References' sections (one from Step 1 and one from Step 2) into a single, consolidated '## References' section that appears only once at the very end of the file.**\n"
            "   5. **Important:** Remove only the first '## References' section (the one generated in Step 1) so that only the consolidated '## References' section remains. Re-sequence all footnote numbers sequentially, and ensure each reference appears on its own line preceded by a bullet point in the following format:\n"
            "      - [^n]: [Short source description](source_url)\n"
            "Step 3 (Final Report Generation):\n"
            "   1. Open 'report.md' and review its content. Based on this information, create a final professional report in a new file named 'report_final.md'. \n"
            "      The final report must be divided into exactly three sections in the following order:\n"
            "        a. Overview: An abstract of approximately 300 characters summarizing the report.\n"
            "        b. Key Points: The main content of the report (presented as bullet points or tables as appropriate).\n"
            "        c. Conclusion: A final summary that integrates all findings and includes the consolidated '## References' section with proper footnote markers.\n\n"
            "Step 4 (Image Generation and Insertion):\n"
            "   1. Use the dalle_tool to generate an image that represents the future outlook of the company. Call dalle_tool with an appropriate prompt and capture the actual image URL returned. **Do not use any placeholder text.**\n"
            "   2. Then, use the file_tools to read the current content of 'report_final.md'. Prepend the generated image to the beginning of the content by formatting it in Markdown image syntax. For example, if the actual image URL is 'https://example.com/your_image.png', the image Markdown should be:\n"
            "      `![Company Future](https://example.com/your_image.png)`\n"
            "   3. **Important:** Ensure that you replace any placeholder (such as '<actual_image_url from dalle_tool>') with the real image URL provided by dalle_tool. Finally, save the updated content back to 'report_final.md' by overwriting the file.\n\n"
            "Additional Requirements:\n"
            "   1. Each step's output must be saved to its corresponding file and used as the input for the next step.\n"
            "   2. All sections must strictly follow the formatting: use level 2 Markdown headers, bullet points or tables as needed, and all citations must remain in footnote format.\n"
            "   3. The final '## References' section must be a single, merged list containing all citation definitions (from both the PDF summary and the web search results) in sequential order, with each reference on its own line preceded by a bullet point in the format:\n"
            "      - [^n]: SourceName (source_url)\n\n"
            "When the user instructs: 'Generate the financial report', perform all the above steps sequentially and output the complete content of 'report_final.md' as the final result.",
  - Placeholders:
            "{chat_history}"
            "{agent_scratchpad}"

- HUMAN:
  - "{input}": Provides task-specific queries (e.g., to summarize the document, conduct a web search, generate a final report, or create an image) as needed.

## **Output**
- The output consists of one or two Markdown files:
  - **`report.md`:** Contains the initial summary from the provided document along with any appended web search results. All headings use level 2 (##) Markdown style, and all citations conform to the specified guidelines.
  - **`report-final.md`:** *(Optional)* A final, professional report structured into exactly three sections:
    - **Overview:** An abstract of approximately 300 characters.
    - **Key Points:** The main content, which may include a Markdown table with detailed information.
    - **Conclusion:** Final conclusions and references, with all citations (document or web) including complete details.
  - Optionally, the final report begins with an embedded image URL (in Markdown format) generated by the image generation tool.

## **Tools**
- **pdf_search tool (retriever_tool):**
  - **Description:** Searches for relevant information within the provided document.
  - **Input:** Accepts a query for searching the document. Uses a predefined prompt template that formats the output with `<document>`, `<content>`, `<page>`, and `<source>` tags. The `{source}` value is dynamic or defaults to a preset value if not provided.
  - **Sample Output:**  
    ```xml
    <document>
      <content>Extracted text from the document...</content>
      <page>4</page>
      <source>data/example_document.pdf</source>
    </document>
    ```
- **tavily_search tool (search tool):**
  - **Description:** Performs web searches on topics related to the provided document.
  - **Input:** Requires a search query provided with the key `query`.
  - **Sample Output:**  
    ```json
    [
      {
        "url": "https://example.com/article1",
        "content": "Summary of article 1..."
      },
      {
        "url": "https://example.com/article2",
        "content": "Summary of article 2..."
      }
    ]
    ```
- **dall-e_tool (image generation tool):**
  - **Description:** Generates an image based on a text prompt.
  - **Input:** Accepts a descriptive text prompt.
  - **Sample Output:**  
    ```markdown
    ![Futuristic Vision](https://example.com/generated_image.png)
    ```
- **File Management Tools:**
  - **Description:** Include tools such as WriteFileTool, ReadFileTool, and ListDirectoryTool that handle file operations. They manage creating, updating, and reading files (e.g., `report.md` and `report-final.md`).
  - **Sample Outputs:**
    - **WriteFileTool:** Returns confirmation that the file has been written (e.g., "File written successfully to report.md").
    - **ReadFileTool:** Returns the content of the specified file.
    - **ListDirectoryTool:** Returns a list of files within a specified directory.

## **Additional Information**
- **Footnotes:**
  - [^1]: "Citation" refers to the original source from the document, which must include the complete filename (or dynamic source) along with the page number.
  - [^2]: For web citations, the exact full URL (including the protocol, e.g., https://) must be used.
- **Guidelines:**
  - The final report (`report-final.md`) is generated only upon explicit user request, while the initial summary (`report.md`) is produced as a standalone step.
  - All conversation context is maintained using the placeholders `{chat_history}`, `{input}`, and `{agent_scratchpad}` to ensure continuity and proper handling of intermediate steps.

