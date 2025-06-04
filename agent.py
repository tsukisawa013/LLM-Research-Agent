import os
import fitz
import json
import urllib, urllib.request
import xml.etree.ElementTree as ET
import mysql.connector
import chromadb
from langchain import hub
from langchain.agents import initialize_agent, tool, AgentExecutor, create_react_agent
from langchain.agents import AgentType
from langchain.tools import BaseTool
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits.load_tools import load_tools

GOOGLE_API_KEY = "Your Google API key"
GOOGLE_MODEL_NAME = "gemini-2.5-flash-preview-04-17"

def create_embDB(): #chroma DB
    chroma_client = chromadb.PersistentClient(path="./chroma_db_store") #set vector store path
    emb_function = chromadb.utils.embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=GOOGLE_API_KEY) #use google AI embeddings
    name = "collection"
    collection = chroma_client.get_or_create_collection(
        name=name,
        embedding_function=emb_function,
    )
    return collection

def load_DB(): #SQL server
    mydb = mysql.connector.connect(
        host="127.0.0.1", #or your server address
        user="Your user name",
        passwd="Your password",
        database="Your db instance"
    )
    # The db instance should contain the following schema:
    # CREATE TABLE papers (
    #     id SERIAL PRIMARY KEY,
    #     title TEXT NOT NULL,
    #     abstract TEXT,
    #     source TEXT,
    #     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    #     authors TEXT,
    #     pdf_filename TEXT,
    #     paper_url TEXT
    # );

    return mydb

def get_llm_model():
    #return ChatOpenAI(model_name=OPENAI_MODEL_NAME, temperature=0, api_key=OPENAI_API_KEY)
    return ChatGoogleGenerativeAI(model=GOOGLE_MODEL_NAME, google_api_key=GOOGLE_API_KEY)


def get_agent_buildin_tool(llm, db, embDB):
    #load tools
    tools = load_tools(["llm-math", "wikipedia"], llm=llm)
    pdf_tool = PDFUploadTool(extract_model=llm, db=db, embDB=embDB)
    tools.append(pdf_tool)
    arXiv_tool = arXivSearchTool(db=db, embDB=embDB)
    tools.append(arXiv_tool)
    search_tool = InternalSearchTool(db=db, embDB=embDB)
    tools.append(search_tool)
    compare_tool = ComparePapersTool(compare_model=llm)
    tools.append(compare_tool)
    
    #enable memories with 10 conversation.
    memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=10)

    #pull ReAct prompt template
    pull_prompt = hub.pull("hwchase17/react")

    original_template = pull_prompt.template
    original_input_variables = pull_prompt.input_variables
    
    #add additional instruction to perform in-context learing
    new_instructions = """
                        IMPORTANT INSTRUCTIONS FOR HANDLING NO-TOOL SCENARIOS:
                        If you determine that no tool is necessary to respond to the user's question or instruction (for example, if it's a greeting, a direct question you can answer from general knowledge, or a negative command like "don't do X"), your thought process should clearly state why no tool is needed. You MUST then conclude your response with 'Final Answer:' followed by your direct, natural language reply to the user. Do NOT attempt to use a tool or output an empty or malformed action in such cases.

                        Example of handling a negative command without a tool:
                        Question: Don't upload this file 'report.pdf'.
                        Thought: The user is explicitly asking me NOT to upload a file. This is a direct instruction that doesn't require any tool. I should acknowledge this and confirm.
                        Final Answer: Ok, I won't upload this file.

                        Example of handling a general question without a tool:
                        Question: How are you?
                        Thought: The user is asking a general greeting. I don't need a tool for this. I should respond politely.
                        Final Answer: I'm doing well, thank you for asking! Is there anything I can help you with?

                        EXAMPLE OF USING A TOOL AND THEN GIVING A FINAL ANSWER:
                        Question: Search 3 papers related to "Mamba for MOT".
                        Thought: I should use the arXivSearchTool and request 3 results.
                        Action: arXivSearchTool
                        Action Input: query="Mamba for MOT", max_results=3
                        Observation: Based on your query 'Mamba for MOT', 3 related papers were found on arXiv:
                        1. Title: MambaTrack... Abstract: ... URL: ...
                        2. Title: MM-Tracker... Abstract: ... URL: ...
                        3. Title: RGBT Tracking... Abstract: ... URL: ...
                        Thought: The arXivSearchTool successfully returned information for 3 papers. This information is sufficient to answer the user's question. I should organize these results and present them to the user.
                        Final Answer: I found the following papers related to "Mamba for MOT" for you:
                        1. Title: MambaTrack... Abstract: ... URL: ...
                        2. Title: MM-Tracker... Abstract: ... URL: ...
                        3. Title: RGBT Tracking... Abstract: ... URL: ...
                        I hope this information is helpful!

                        If the user has a clear request, you should process the tool's output to provide a more precise answer.
                        Question: Find the paper related to "Mamba" from the internal database.
                        Thought: The user is asking me to search the internal database for information about "Mamba". I should use the InternalSearchTool for this.
                        Action: InternalSearchTool
                        Action Input: Mamba
                        Observation: Based on your query 'Mamba', 5 related papers were found in the internal database:
                        1. Title: MambaTrack... Abstract: ... URL: ...
                        2. Title: MM-Tracker... Abstract: Mamba... URL: ...
                        3. Title: YOLO... Abstract: ... URL: ...
                        4. Title: Transformer... Abstract: ... URL: ...
                        5. Title: Object Detection... Abstract: ... URL: ...
                        Thought: The user asked for papers related to "Mamba" from the local database. I should present the relevant results to the user.
                        Final Answer: I have found the following papers related to "Mamba" in the internal database:：
                        1. Title: MambaTrack... Abstract: ... URL: ... 
                        2. Title: MM-Tracker... Abstract: ... URL: ...
                        Hope this information is helpful!

                        If the user has a clear request, you should process the tool's output to provide a more complete or concise answer.
                        Question: Please search the internal database for papers related to "[user's query/keywords]".
                        Thought: The user is asking me to search the internal database for information about "[user's query/keywords]". I should use the InternalSearchTool for this.
                        Action: InternalSearchTool
                        Action Input: "[user's query/keywords]"
                        Observation: Based on your query "[user's query/keywords]", 5 related papers were found in the internal database:
                        1. Title: [Relevant Paper A Title] Abstract: ... URL: ...
                        2. Title: [Relevant Paper B Title] Abstract: ... [user's query/keywords] ... URL: ...
                        3. Title: [Less Relevant Paper C Title] Abstract: ... URL: ...
                        4. Title: [Relevant Paper D Title] Abstract: ... URL: ...
                        5. Title: [Less Relevant Paper E Title] Abstract: ... URL: ...
                        Thought: The InternalSearchTool returned multiple results. Some of them (e.g., Paper C and E) seem less directly relevant to "[user's query/keywords]" based on their titles/abstracts. I should filter these out and present the most relevant ones to the user.
                        Final Answer: I found the following papers related to "[user's query/keywords]" in the internal database:
                        1. Title: [Relevant Paper A Title] (Abstract and URL omitted for brevity in this example)
                        2. Title: [Relevant Paper B Title]
                        4. Title: [Relevant Paper D Title]
                        Hope this information is helpful!
                        
                        You have access to memory (conversation history). When the user refers to past interactions or information, please utilize this history.
                        Question: Please compare the paper about "MOTR" that I just uploaded with the first paper about "Mamba" that you found earlier.
                        Thought: The user wants to compare two papers. One is the "MOTR" paper that was just uploaded, and the other is the first "Mamba" paper from a previous search. I need to retrieve the titles and abstracts for both papers from my conversation history (memory).
                        (Simulated thought process of recalling from memory)
                        From memory (UploadPDFTool output for MOTR):
                        - MOTR Title: "MOTR: End-to-End Multiple Object Tracking with TRansformer"
                        - MOTR Abstract: "Multiple object tracking (MOT) is a challenging task..."
                        From memory (arXivSearchTool's first result for Mamba):
                        - Mamba Paper Title: "MambaTrack: A Simple and Effective Baseline for MOT"
                        - Mamba Paper Abstract: "This paper introduces MambaTrack..."
                        I have now found the necessary information for both papers. I will call the ComparePapersTool.
                        Action: ComparePapersTool
                        Action Input: 
                            [JSON content]
                        Observation: [Full comparison report text returned by ComparePapersTool, e.g., "## Comparison Report...\nResearch Goals: MOTR aims to... AED aims to...\nMethods: ..."]
                        Thought: I have received the comparison report from the tool. I can now provide the answer to the user.
                        Final Answer: Here is the comparison report for the "MOTR" paper and the "MambaTrack" paper:
                        [Full comparison report text from Observation]
                        I hope this information is helpful to you!

                        Regarding search results from tools:
                        Always present the retrieved data in its full original text, without any prior summarization or translation, unless the user explicitly asks for a modification.

                        And a crucial reminder for all interactions:
                        Your final response to the user MUST always start with the 'Final Answer:' tag.
                        """
    
    #spilt original template
    split_marker = "Final Answer: the final answer to the original input question"
    parts = original_template.split(split_marker)

    #merge original template and additional instruction
    if len(parts) == 2:
        modified_template_str = parts[0] + split_marker + new_instructions + parts[1]
    else:
        print("Warning: Could not split template string by the 'Final Answer:' format line as expected. Adding instructions to the beginning of the template.")
        modified_template_str = new_instructions + "\n\n" + original_template

    #create template instance
    custom_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template(modified_template_str)
    ])

    #create agent instance
    agent_runnable = create_react_agent(llm, tools, custom_prompt)

    agent_executor = AgentExecutor(
        agent=agent_runnable,
        tools=tools,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True
    )

    return agent_executor

def store_paper(db:mysql.connector=None,
                embDB:chromadb.Collection=None,
                title:str="", 
                abstract:str="", 
                source:str="", 
                authors:str="", 
                pdf_filename:str="", 
                link:str="") -> tuple[bool, str]:
    
    if not db or not embDB:
        return False, "Fail, cannot connect to SQL server."
    
    #store information to SQL server
    cursor = db.cursor()
    sql = "INSERT INTO papers (title, abstract, source, authors, pdf_filename, paper_url) VALUES (%s, %s, %s, %s, %s, %s)"
    val = (title, abstract, source, authors, pdf_filename, link)
    try:
        cursor.execute(sql, val)
    except Exception as e:
        return False, f"Fail with error: {e} while hadleing text."
    db.commit()
    
    #get last stored id for chroma db
    paper_id = cursor.lastrowid
    text_to_embed = f"Title: {title}\n\nAbstract: {abstract}"
    try:
        #store vector
        embDB.add(
            documents = [text_to_embed],
            metadatas=[{
                "sql_id": paper_id, #SQL id use for retriev full information from SQL server
                "title": title,
                "source": source
            }],
            ids=[str(paper_id)]
        )
    except Exception as e:
        delete_sql = "DELETE FROM papers WHERE id=%s;"
        try:
            cursor.execute(delete_sql, (paper_id,))
        except Exception as e:
            print(f"[DEBUG] 資料庫回滾時遭遇狀況：{e}，請確認資料庫狀態。")
        db.commit()
        return False, f"Fail with error: {e} while storing embedding."
    return True, "Success, the file is uploaded."


class PDFUploadTool(BaseTool):
    name : str = "PDFUploadTool"
    description : str = (
        """Use this tool to upload a research paper PDF from a local file path. The input must be the full file path to the PDF.
        The tool will attempt to extract the title and abstract, store them in the internal database. 
        The tool will return a message indicating the result (success or error)."""
        )
    extract_model: ChatGoogleGenerativeAI
    db: mysql.connector
    embDB: chromadb.Collection
    def __init__(self, extract_model: ChatGoogleGenerativeAI, db:mysql.connector, embDB:chromadb.Collection, **kwargs):
        super().__init__(extract_model=extract_model, db=db, embDB=embDB, **kwargs)

    def _run(self, file_route: str = None,):
        if file_route:
            try:
                if not os.path.exists(file_route):
                    return f"Fail, file not found: '{file_route}'."
                if not os.path.isfile(file_route):
                    return f"Fail, '{file_route}' is not a file."
                if not file_route.lower().endswith('.pdf'):
                    return f"Fail, '{file_route}' is not a PDF file."
                
                title = "Title not found"
                abstract = "Abstract not found"
                full_text = ""

                #load PDF text
                doc = fitz.open(file_route)
                pdf_filename = os.path.basename(file_route)

                #load whole PDF text
                for page in doc:
                    full_text += page.get_text()

                doc.close()

                #use LLM to extract Title and Abstract
                extraction_prompt = f"""
                    Please extract the title, abstract, and authors from the provided research paper text.
                    Strictly adhere to the following JSON format for your response. 
                    Do not include any explanatory text outside of the JSON object itself.

                    {{
                    "title": "The title of the paper",
                    "abstract": "The content of the paper's abstract",
                    "authors": "author1, author2, ..."
                    }}

                    If a clear title cannot be found, set the value of "title" to "Title not found".
                    If a clear abstract cannot be found, set the value of "abstract" to "Abstract not found".
                    If authors cannot be clearly identified, set the value of "authors" to "Authors not found" or an empty string.

                    The paper text is as follows:
                    ---
                    {full_text[:8000]}
                    ---
                    """
                #to save the token usage, only 8000 words send to LLM

                #get LLM response
                response = self.extract_model.invoke(
                            extraction_prompt,
                            #response_format={"type": "json_object"}
                        )
                content_str = response.content
                #some time LLM return in markdown format, need to fix it back to correct JSON format
                if content_str.strip().startswith("```json"):
                    content_str = content_str.strip()[7:]
                if content_str.strip().endswith("```"):
                    content_str = content_str.strip()[:-3]

                content_str = content_str.strip()

                try:
                    extracted_data = json.loads(content_str)
                    title = extracted_data["title"]
                    abstract = extracted_data["abstract"]
                    authors = extracted_data["authors"]
                    print(f"[DEBUG] 成功處理文件 '{pdf_filename}'。")
                except json.JSONDecodeError as e:
                    print(f"[DEBUG] LLM 返回的不是有效的 JSON: {content_str}")
                    return f"Fail with error: {e} while extracting Title and Abstract."
                
                #store paper
                state, description = store_paper(db=db, 
                                                 embDB=embDB, 
                                                 title=title, 
                                                 abstract=abstract, 
                                                 source="internal_upload",
                                                 authors=authors,
                                                 pdf_filename=pdf_filename,
                                                 link="None")
                
                if state:
                    return f"Success, the file is uploaded. Title: {title}. Abstract: {abstract}."
                else:
                    return description

            except Exception as e:
                # 捕獲所有潛在錯誤（文件讀取、解析、數據庫操作等）
                return f"Fail with error {e} while processing: '{file_route}'."
                
        else:
            return "Fail, no file provided."
        
class arXivSearchTool(BaseTool):
    name : str = "arXivSearchTool"
    description : str = (
        """This is the arXiv Search Tool, used to search for research papers on arXiv.org. Use this tool when the user wants to find recent papers on a specific topic, keywords, or by author.
        The input for this tool must be a JSON string containing the following keys: "query", "max_results", and "save_to_DB".

        "query" (string): The search topic or keywords.
        "max_results" (integer, optional): The maximum number of papers to return (e.g., defaults to 3 or 5 if not specified).
        "save_to_DB" (boolean, optional): Indicates whether the search results should be saved to the internal database. 

        Example of Action Input: {"query": "Mamba for MOT", "max_results": 2, "save_to_DB": true} 

        This tool will attempt to search for papers on arXiv. The tool will return relevant information and a processing result (success or failure). 
        Important Note: The "save_to_DB" field should default to false unless the user explicitly asks to save the papers."""
    )
    db: mysql.connector
    embDB: chromadb.Collection
    def __init__(self, db: mysql.connector, embDB: chromadb.Collection, **kwargs):
        super().__init__(db=db, embDB=embDB, **kwargs)

    def _run(self, input: str = None):
        print(f"[DEBUG] {input}")
        if input.strip().startswith("```json"):
            input = input.strip()[7:]
        if input.strip().endswith("```"):
            input = input.strip()[:-3]

        input = input.strip()

        try:
            data = json.loads(input)
            query = data["query"]
            max_results = data["max_results"]
            save_to_DB = data["save_to_DB"]
            print(f"[DEBUG] {query}")
            print(f"[DEBUG] {max_results}")
            print(f"[DEBUG] {save_to_DB}")
        except Exception as e:
            return f"Fail with error: {e} Please check the JSON format."
        
        #format the query into URL format
        query_formatted = urllib.parse.quote_plus(query)
        url = f'http://export.arxiv.org/api/query?search_query=all:{query_formatted}&start=0&max_results={max_results}'

        #send request
        with urllib.request.urlopen(url) as response:
            try:
                xml_data = response.read().decode('utf-8')
            except Exception as e:
                return f"Fail with error: {e} while searcing on arXive."

        #arXiv return information in XML format in atom format
        root = ET.fromstring(xml_data)
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        
        #get information from XML format
        found_papers = []
        for entry in root.findall('atom:entry', namespace):
            title = entry.find('atom:title', namespace).text.strip().replace('\n', ' ').replace('\r', '')
            abstract = entry.find('atom:summary', namespace).text.strip().replace('\n', ' ').replace('\r', '')
            paper_id_url = entry.find('atom:id', namespace).text.strip()
            authors = [author.find('atom:name', namespace).text for author in entry.findall('atom:author', namespace)]
            published_date_element = entry.find('atom:published', namespace)
            published_date = published_date_element.text.strip() if published_date_element is not None else "N/A"

            found_papers.append({
                "title": title,
                "abstract": abstract,
                "authors": ", ".join(authors),
                "link": paper_id_url,
                "published_date": published_date.split('T')[0] # 只取日期部分
            })
        
        if not found_papers:
            return f"No paper found from arXiv based on your query: '{query}'."

        try:
            #format the output for agent
            output_parts = [f"Based on your query '{query}', {len(found_papers)} related papers were found on arXiv:\n"]
            for i, paper in enumerate(found_papers):
                output_parts.append(
                    f"\n{i+1}. Title: {paper['title']}\n"
                    f"   Abstract: {paper['abstract']}\n"
                    f"   Authors: {paper['authors']}\n"
                    f"   URL: {paper['link']}\n"
                    f"   Published Date: {paper['published_date']}"
                )
                if save_to_DB:
                    state, description = store_paper(db=db, 
                                                 embDB=embDB, 
                                                 title=paper['title'], 
                                                 abstract=paper['abstract'], 
                                                 source="web_search",
                                                 authors=paper['authors'],
                                                 pdf_filename="None",
                                                 link=paper['link'])
                    if not state:
                        return description
            
            return "".join(output_parts)
        except Exception as e:
            return f"Fail with error: {e} while processing search result."
        
class InternalSearchTool(BaseTool):
    name : str = "InternalSearchTool"
    description : str = (
        """
        This is the Internal Paper Search Tool. The input should be a string (a natural language query).
        This tool will return the top 5 candidate results.
        """
        )
    db: mysql.connector
    embDB: chromadb.Collection

    def __init__(self, db: mysql.connector,  embDB: chromadb.Collection, **kwargs):
        super().__init__(db=db, embDB=embDB, **kwargs)

    def _run(self, query: str = None,):
        #search in chroma db
        results = embDB.query(
            query_texts=[query],
            n_results=5,
            include=["metadatas"],
        )

        #search in SQL
        cursor = db.cursor()
        found_papers = []
        for id in results["ids"][0]: #use id stored in chroma db to find specific paper
            search_sql = "SELECT * FROM papers WHERE id=%s;"
            try:
                cursor.execute(search_sql, (int(id),))
                result = cursor.fetchone()
            except Exception as e:
                return f"Fail with error: {e} while searching in database."
            #format search result into dict
            if result:
                column_names = [desc[0] for desc in cursor.description]
                paper_dict = dict(zip(column_names, result))
                found_papers.append(paper_dict)

        #format the output for agent
        output_parts = [f"Based on your query '{query}', {len(found_papers)} papers were found in the internal database:\n"]
        for i, paper in enumerate(found_papers):
            output_parts.append(
                f"\n{i+1}. Title: {paper['title']}\n"
                f"   Abstract: {paper['abstract']}\n"
                f"   Authors: {paper['authors']}\n"
                f"   URL: {paper['paper_url']}\n"
            )
        
        return "".join(output_parts)
            

        
class ComparePapersTool(BaseTool):
    name : str = "ComparePapersTool"
    description : str = (
        """ Use this tool to compare two or more research papers. Provide the titles and abstracts for each paper to be compared. 
        The information should be clearly structured so that each paper's title and abstract are distinguishable.
        Example of expected Action Input format (clearly structured information):
        {
            title1: [Title of Paper 1],
            abstract1: [Abstract of Paper 1],
            title2: [Title of Paper 2],
            abstract2: [Abstract of Paper 2],
            ...
        }
        This tool will return the outcome (either the report or an error message)."""
        )
    compare_model: ChatGoogleGenerativeAI

    def __init__(self, compare_model: ChatGoogleGenerativeAI, **kwargs):
        super().__init__(compare_model=compare_model, **kwargs)

    def _run(self, input: str = None,):
        if input.strip().startswith("```json"):
            input = input.strip()[7:]
        if input.strip().endswith("```"):
            input = input.strip()[:-3]

        input = input.strip()
        #directly send the str to avoid JSON format issue
        '''try:
            data = json.loads(input)
            title1 = data["title1"]
            title2 = data["title2"]
            print(f"[DEBUG]{title1}")
            print(f"[DEBUG]{title2}")
        except Exception as e:
            return f"發生錯誤{e}，請確認輸入是正確的JSON格式。"'''
        print(f"[DEBUG]\n{input}")
        
        #use LLM to generate report
        compare_prompt = f"""
                    Please generate a comparison report for the research papers based on their titles and abstracts provided below.
                    The report should be approximately 200-300 words long.

                    The report must follow this structure:
                    research goals: ...
                    methods: ...
                    contributions: ...
                    strengths: ...
                    weaknesses: ...
                    similarities: ...

                    The details of the papers to be compared are: 
                    ---
                    {input} 
                    ---
                    """
        #get LLM response
        response = self.compare_model.invoke(
                    compare_prompt,
                    #response_format={"type": "json_object"}
                )
        content_str = response.content

        return f"Success, here is the report: {content_str}"


if __name__ == "__main__":
    llm = get_llm_model()
    db = load_DB()
    embDB = create_embDB()
    agent = get_agent_buildin_tool(llm=llm, db=db, embDB=embDB)

    while True:
        try:
            user_input = input("You: ")

            if user_input.lower() in ["退出", "掰掰", "再見", "quit", "exit", "bye"]:
                print("Agent: Happy to help! Goodbye.")
                break

            response = agent.invoke({"input": user_input})

            print(f"Agent: {response.get('output', '抱歉，我沒有得到明確的輸出。')}")

        except Exception as e:
            print(f"An error occurred: {e}")
            #use try except block to avoid unexpected crash, but could cause debug difficulty
            #remove try except block to get specific error message