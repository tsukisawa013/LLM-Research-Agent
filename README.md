---
title: LLM Agent Project Report

---

# Project overview
Our project has successfully completed an LLM Agent that can interact with users through the natural language.
We completed this project based on LangChain and Google Gemini. We designed 4 tools as required: PDFUploadTool, arXivSearchTool, InternalSearchTool, and ComparePapersTool.
# System Architecture
### 1. Agent core
The core of our Agent is Google `gemini-2.5-flash-preview-04-17` with ReAct architecture provided by LangChain.
### 2. Tools
##### UploadPDFTool
In our design, UploadPDFTool is used to upload a PDF to the internal database. It then returns the extracted title and abstract to the agent.
First, agent has to identify the file path that the user provides as the tool's input.
Since there is a lot of variety in PDF structures, traditional algorithms often can't extract titles and abstracts perfectly. Therefore, we first load part of the PDF content (the first 8000 characters) using PyMuPDF. This extracted text is then sent to an LLM in a separate process with a prompt asking it to identify and extract the title and abstract.
After obtaining the title and abstract, we store this metadata in our SQL database (MySQL) and create and store an embedding of the title and abstract in ChromaDB for semantic search.
##### arXivSearchTool
The inputs for this tool are the search query, the desier number of the search results, and a boolean indicating whether to store the results in the internal database. The agent must extract these inputs from the user's statement.
The arXive API return information in XML format; we use python Python's built-in library  to extract the information. Finally, the tool formats and returns a summary of these findings to the agent.
We were not sure whether all search results should be automatically stored. We decided to let user decide to store the search result or not. We have found that the agent tended to set `save_to_DB` as True if user didn't mention. Therefore, we add a caution at tool describe to tell agent set `save_to_DB` False as default unless the user mention.
##### InternalSearchTool
The input for this tool is a natural language search query extracted by the agent from the user's input.
The seach query first embedded and then sent to ChromaDB. ChromaDB will returns the top 5 most similar results based on semantic search. In our design, when adding data to ChromaDB. ChromaDB returns the matching SQL IDs, the tool fetches the complete paper information from the SQL database using these IDs.
The agent tends to present all information returned by the tool. However, vector search can sometimes return less relevent results, especially if there are fewer than 5 highly relevant documents in the database.
To address this, we solved this by adding examples in promp template to teach agent how to filter out less relevent content.
##### ComparePapersTool
The inputs for this tool are the titles and the abstracts of the papers in a JSON format. The agent must extract these inputs from the search tools or its own memory.
This collected content is then sent to an LLM in a separate process with a prompt, asking it to generate comparison report in structured format within a limited length.
### 3. Memory
The memory in our agent is mostly used for recalling information for paper comparison. This mechanism helps to avoid redundant searches and reducing overall token usage.
### 4. DataBase
We utilize an SQL database(MySQL) and ChromaDB to store paper information. For the SQL database, we store `id`, `title`, `abstract`, `source` and `created_at` as required. Additionally, we store `authors`, `file_name` and `url`.
For ChromaDB we utilize Google's Generative AI Embedding Function. We store the combined text of the title and abstract as the document (which is then embedded by the function for storage and later retrieval). The metadatas for each embedding stored the corresponding SQL ID and the paper's title.
These two databases interact to enable searching for internally stored papers as described in the InternalSearchTool section.
# Environment Setup
This project was developed using Python 3.11.11. All Python dependencies are listed in the `requirements.txt` file and can be installed using the command: `pip install -r requirements.txt`.
This project needs a Google API Key. For simplicity the API key is set as a constant `GOOGLE_API_KEY` in the source code. (This is not safe for online environments.)
This project requires a running MySQL server instance.
Within that database, a table named papers is required with the following schema: 
```sql
CREATE TABLE papers (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT,
    source TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    authors TEXT,
    pdf_filename TEXT,
    paper_url TEXT
);
```
ChromaDB is used as the vector store with a PersistentClient, storing its data in the `./chroma_db_store` directory.
To run the project, execute the main script from your terminal:
`python agent.py`
# Limitations and Future work
### JSON Issue
The agent communicate with ComparePapersTool via JSON text string. However, paper abstracts sometimes contain several LaTeX symbols for mathematical or scientific notation, which can lead to invalid escape sequences or formatting issues within the JSON string. The agent can realize and fix it with less amount of \escpae over serveral try. But it is also posible that agent will give up and do the comparison by itself after serveral try.
Thus, finding a more robust format or method for transmitting paper information between the agent and its tools is a necessary area for future improvement.
### Prompt Robustness
As we mentioned at System Architecture section, we've perform extensive prompt engineering on our prompt template to guide the agent work as closely to our expectations as possible (such as using tools corectly, don't forget the `Final Ansewr:` tag). Our project was initially developed on `gemini-2.0-flash`, which is not very smart but obeyed the prompt. In the final stages of development we switched to `gemini-2.5-flash-preview-04-17`, hoping for better performance in prompt understanding. The 2.5-flash did indeed show better performance on prompt understanding but also more creative.
For example, with the \escape issue we mention earlier, 2.0-flash would get stuck in a loop, trying the tools repeatedly. While 2.5-flash can gave up and complete the work by itself in serveral try.
We also try the latest `gemini-2.5-flash-preview-05-20`, but its behavior was problematic for our ReAct setup; it forgot the `Final Answer:` tag much more often than the 04-17 version, which only occasionally missed it.
Thus, our prompt engineering is not robust on all LLM models, it is necssary to finetune the prompt and observe the agent behavior.
