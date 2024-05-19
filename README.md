# AGENTIC RAG LANGCHAIN 👮

<p align="center">
<img alt="Screen Shot 2024-01-05 at 9 05 56 AM" src="https://github.com/benitomartin/Agentic-RAG-Langchain-Pinecone/assets/116911431/5341fd83-8f18-4698-9382-e2734828f308", width=500>
</p>

This repository contains a full Q&A pipeline using LangChain framework, Pinecone as vector database and Tavily as Agent. The data used are the transcriptions of TEDx Talks. A short description of how Tokenizers and Embeddings work is included. Use this **[Link](https://nbviewer.org/github/benitomartin/agentic-rag-langchain-pinecone/blob/main/RAG_Langchain_Agents.ipynb)** if the notebook cannot be opened.

The main steps taken to build the RAG pipeline can be summarize as follows:

* **Data Ingestion**: load data from csv file

* **Tokenization**: how a tokenizer works

* **Embeddgings**: how a embeddgings works with cosine similarity concept

* **Indexing**: RecursiveCharacterTextSplitter for indexing in chunks

* **Vector Store**: Pinecone with several namespace (multi-tenancy)

* **QA Chain Retrieval**: RetrievalQA with memory and agents

Feel free to ⭐ and clone this repo 😉

## 👨‍💻 **Tech Stack**


![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenAI](https://img.shields.io/badge/OpenAI-74aa9c?style=for-the-badge&logo=openai&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)


## 📐 Set Up

In the initial project phase, the documents are loaded using **CSVLoader** and indexed. Indexing is a fundamental process for storing and organizing data from diverse sources into a vector store, a structure essential for efficient storage and retrieval. This process involves the following steps:

- Select a splitting method and its hyperparameters: we will use the **RecursiveCharacterTextSplitter**.

- Select the embeddings model: in our case the **OpenAI**

- Select a Vector Store: **Pinecone**.

Storing text chunks along with their corresponding embedding representations, capturing the semantic meaning of the text. These embeddings facilitate easy retrieval of chunks based on their semantic similarity. 

After indexing, a QA Chain Retrieval Pipeline is set up in order to check the Q&A functioning and performance. Memory and Agents any are included in the process.


## 🌊 QA Chain Retrieval Pipeline

The pipeline created contains the main llm model, memory, the QA chain and the agents. The prompt template is used to complete the QA chain with an slight modification to point out tot he mode to look up first in the Vectorstore.

```
# Set prompt template

template= '''
          Answer the following questions as best you can. You have access to the following tools:

          {tools}

          Use the following format:

          Question: the input question you must answer
          Thought: you should always think about what to do
          Action: the action to take, should be one of [{tool_names}]. Always look first in Pinecone Document Store
          Action Input: the input to the action
          Observation: the result of the action
          ... (this Thought/Action/Action Input/Observation can repeat 2 times)
          Thought: I now know the final answer
          Final Answer: the final answer to the original input question

          Begin!

          Question: {input}
          Thought:{agent_scratchpad}
          '''

prompt = PromptTemplate.from_template(template)
```
```
llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo", max_tokens=512)


# Conversational memory
conversational_memory = ConversationBufferWindowMemory(
                        memory_key='chat_history',
                        k=5,
                        return_messages=True)

# Retrieval qa chain
qa_db = RetrievalQA.from_chain_type(
                                    llm=llm,
                                    chain_type="stuff",
                                    retriever=vectorstore.as_retriever())

tavily = TavilySearchResults(max_results=10, tavily_api_key=TAVILY_API_KEY)

tools = [
    Tool(
        name = "Pinecone Document Store",
        func = qa_db.run,
        description = "Use it to lookup information from the Pinecone Document Store"
    ),

    Tool(
        name="Tavily",
        func=tavily.run,
        description="Use this to lookup information from Tavily",
    )
]

agent = create_react_agent(llm,
                           tools,
                           prompt)

agent_executor = AgentExecutor(tools=tools,
                         agent=agent,
                         handle_parsing_errors=True,
                         verbose=True,
                         memory=conversational_memory)

```

## 📈 Further Steps

* Different database: `Deep Lake`, `Qdrant`, ...
* Adding reranker model: Cohere
* Agentic hierarchies with LangGraph 

