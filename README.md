# SECure RAG

SECure RAG is a framework for financial retrieval-augmented generation that combines agentic workflows, hybrid retrieval, and tool use to answer queries over structured and unstructured financial data.

This repository accompanies the paper **[Enhancing Financial RAG with Agentic AI and Multi-HyDE: A Novel Approach to Knowledge Retrieval and Hallucination Reduction](https://aclanthology.org/2025.finnlp-2.3/)**. The ACL Anthology page links the official metadata, PDF, and citation details.  [oai_citation:0‡ACL Anthology](https://aclanthology.org/2025.finnlp-2.3/)

Evaluation results are included in `./evaluation results`.

---

## Overview

SECure RAG is designed for financial question answering over large and diverse data sources. It combines:

- hybrid retrieval using **MultiHyDE** and **BM25**
- **cross-encoder reranking**
- **agentic tool use** for live and structured data access
- **state-based routing** for different query types
- **custom PDF parsing** for complex financial documents

The system supports both direct retrieval-heavy pipelines and more flexible tool-driven workflows for harder queries.

---

## Repository Structure

```text
.
├── FinAgent/                  # Core agent code, tools, retrievers, config
├── PDF Parser/                # Custom parser integrated with Pathway
│   ├── Dense Retriever/
│   └── Sparse Retriever/
├── evaluation results/        # Evaluation outputs
├── images/                    # Architecture diagrams and README assets
├── app.py                     # Streamlit app entry point
├── main.py                    # FastAPI backend entry point
├── Dockerfile
├── Streamlit-Dockerfile
└── README.md
⸻

Installation

Running the UI with Streamlit
	1.	Enter this directory and install the FinAgent package in a virtual environment with Python 3.12:

pip install -e .

	2.	Install Streamlit:

pip install streamlit

	3.	Set your environment variables in a .env file in the parent folder, or set them by another method. Instructions for obtaining API keys are in API Key Instructions￼.
	4.	Run the app:

streamlit run app.py

The UI will be hosted at:

http://localhost:8501/

Note: Between queries, the agent should be reset using the reset button in the UI, since follow-up queries are treated differently from new queries.

⸻

Hosting locally with FastAPI and Docker

Note: This setup currently supports one client at a time, since the agent state must be reset between conversations to preserve conversation history and explainability.
	1.	Build the Docker image:

docker build -t secure_rag .

	2.	Run the Docker container:

docker run -p 1524:1524 --env-file .env secure_rag

	3.	The FastAPI server will be hosted at:

http://localhost:1524/

Available endpoints:
	•	PUT /reset
Resets the agent.
	•	POST /query
Query the agent with JSON:

{
  "query": "Your query here"
}

Response format:

{
  "messages": [
    {
      "role": "One of (User, Tool, Guardrail, Assistant)",
      "content": "Message content"
    }
  ],
  "iterations": 0
}

	•	POST /explain
Explain a selected assistant response with JSON:

{
  "query": "Copied text from the assistant to explain"
}

Response format:

{
  "explanation": "Explanation text"
}


⸻

Hosting the UI with Docker
	1.	Build the Docker image:

docker build -t secure_rag_ui -f Streamlit-Dockerfile .

	2.	Run the Docker container:

docker run -p 8501:8501 --env-file .env secure_rag_ui

	3.	The UI will be available at:

http://localhost:8501/

Notes:
	•	Between queries, the agent should be reset, since follow-ups are treated differently from new queries.
	•	Guardrails are not active on the agent’s thoughts, plans, and queries shown in the UI sidebar, since these would not be exposed in a production setting.

⸻

Running the Vector Store

This repository contains a PDF Parser folder with the code for the custom document parser integrated with Pathway. Two retrievers are used: a dense retriever and a sparse retriever.

Dense Retriever

The dense retriever uses the OpenAI model text-embedding-3-large to generate embeddings. The vector store reads from ./to-process/, parses .pdf files, and creates a vector store.

To enable reading from a Google Drive file, uncomment lines 17–22 in vectorstore.py.

Setup:

python -m venv .venv
pip install -r requirements.txt
python vectorstore.py

The folder PDF Parser/Dense Retriever contains the code hosted on our GCP instance.

If an older version of libgl causes errors, run:

apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

Enabling read from Google Drive
	•	Add the Google Drive object ID
	•	Add the credentials.json file containing the details of your Drive folder

Once running, the program exposes port 8666.

To confirm functionality, run:

python rag-tool.py

For our experiments, we hosted a VM on Google Cloud and exposed the port using ngrok.

⸻

Sparse Retriever

Setup:

python -m venv .venv
pip install -r requirements.txt
python app.py

Additional step:
	•	Copy docParser.py from the folder
	•	Paste it into:

.venv/lib/python3.11/site-packages/pathway/xpacks/llm

The folder PDF Parser/Sparse Retriever contains the code hosted on our GCP instance.

⸻

Architecture

StatelessAgent

StatelessAgent is the optimized retrieval-first pipeline.
	•	Uses HyDE with Pathway’s Vector Store and Document Store
	•	Parses, chunks, and embeds data dynamically
	•	Allows new data to be added through a Google Drive folder
	•	Uses retrieval and tools to answer user queries
	•	Works well for most direct financial queries, but may struggle on more complex queries or when required information is missing from the dataset

This pipeline combines:
	•	dense retrieval with MultiHyDE
	•	sparse retrieval with BM25
	•	cross-encoder reranking

The LLM generates multiple rephrased versions of the query, creates HyDE embeddings for each, retrieves top chunks using MultiHyDE and BM25, merges the results, and reranks them.

The final retrieved chunks and confidence scores are then passed to the LLM.

If more information is needed, the LLM can use tools such as:
	•	Yahoo Finance
	•	Python Calculator
	•	Edgar Tool
	•	Bing Web Search

This makes the pipeline suitable for well-defined data flows where strong retrieval is the main requirement.

⸻

MultiState

MultiState allows the agent to switch between different states depending on the query.
	•	The agent selects the state best suited to the user query
	•	Each state contains its own tools and instructions
	•	The agent self-checks the final response before presenting it

This approach uses three states in addition to a base state with no tools:
	1.	Fact-based Finance
	2.	Statistical Analysis Comparison
	3.	Trend Analysis and Event-Based

This setup helps the agent use the right tools and reasoning style for different query types.

⸻

MetaState

MetaState is designed for more complex queries that may require multiple tools and more deliberate reasoning.
	•	The master state can call any tool in the system
	•	The LLM is given planning and query decomposition capabilities in addition to standard reasoning

The available tools include:
	•	HyDE-based RAG tools
	•	edgar_tool
	•	Alpha Vantage Exchange Rate
	•	web_search
	•	Python calculator

This setup supports more dynamic workflows where the agent must combine retrieval, planning, and live information access.

⸻

Final Pipeline

Our experiments showed that the strongest results came from the StatelessAgent and MetaState pipelines.

The final pipeline therefore combines both:
	•	starts with useful outputs from HyDE
	•	follows up with broader tool use from MetaState when needed

This provides:
	•	strong initial retrieval grounding
	•	better handling of missing information
	•	more flexible support for difficult queries

⸻

Features
	•	Robust and accurate retrieval from both websites and privately hosted databases
	•	Agentic system that handles and corrects tool calls for adaptive retrieval
	•	Support for multiple data sources and large databases, given sufficient resources for hosting the vector store
	•	A thoughts, tools, and audio approach, allowing separation of internal thoughts, tool calls, and user-facing responses
	•	Easily modifiable configuration for new use cases through states and prompts in FinAgent/config/
	•	Guardrails for both user input and agent output
	•	Streamlit frontend for chatting, with a transparent thought process and an explanation page for user-selected outputs

Wide and extensible tool support
	•	Retrievers and API calls are both treated as tools
	•	Tools can be added or removed easily
	•	Multiple tool calls can be executed asynchronously when they do not depend on each other

Robust PDF Parser

We use a PDF parsing system implemented in Python with Docling and integrated with Pathway to extract and organize data from complex documents.

The parser supports:
	•	text
	•	tables
	•	images
	•	JSON export
	•	HTML export for tables

Table embeddings are handled by row-and-column aggregation over parsed HTML tables.

We also provide a base class for integrating custom parsers with Pathway’s ETL library.

This is useful for enterprise and financial documents where important information is often spread across text, tables, and semi-structured layouts.

Easily configurable
	•	States, tools, prompts, and agents can be changed easily
	•	Models can be customized for every state
	•	The system supports Gemini, OpenAI, Groq, Ollama, and LiteLLM-compatible models
	•	Additional models can be added in FinAgent/models/models.py
	•	States and prompts can be added in config files
	•	Tools can be added in FinAgent/tools/ or FinAgent/retrievers/
	•	The system can be extended to support agents with dynamic prompts or default tool-use patterns

⸻

Configuration

The system is configurable at the level of states, tools, prompts, and agents.

Core files:
	•	FinAgent/agents.py — agent definitions
	•	FinAgent/config/prompts.py — prompts
	•	FinAgent/config/states.py — states
	•	FinAgent/tools/ — tools
	•	FinAgent/retrievers/ — retrievers and retrieval tools

Environment variables

These can be defined in a .env file in the parent folder or passed through Docker using --env-file.

USE_GUARDRAIL=
PATHWAY_VECTOR_STORE_URL=
HYDE_BM25_URL=
GEMINI_API_KEY=
GROQ_API_KEY=
OPENAI_API_KEY=
BING_SEARCH_API_KEY=
REPLICATE_API_KEY=
WOLFRAM_ALPHA_APPID=
ALPHAVANTAGE_API_KEY=
ASKNEWS_CLIENT_ID=
ASKNEWS_CLIENT_SECRET=
FINPREP_API_KEY=

Information on getting API keys is in API Key Instructions￼.

Changing prompts
	•	Prompts are defined in FinAgent/config/prompts.py
	•	Built-in agents can be modified directly there
	•	If prompts need to change with state, str.format is used in Agent.set_system_prompt()
	•	A helper function in prompts.py is included to escape {} and use alternate characters such as <> where needed

Changing states and tools
	•	New state sets can be added in FinAgent/config/states.py
	•	Tools are initialized in states.py and assigned to states there
	•	Tools can be added or removed as needed
	•	States contain their own instructions, which are inserted into the prompt through {state_details} when used by Agent.set_system_prompt()
	•	Tool and state format details are documented in Adding new tools.md￼

Switching agents
	•	Agents are defined in FinAgent/agents.py
	•	The selected agent can be changed in agent_page.py and main.py
	•	This is done by changing the chosen class in the config dictionary used in those files

⸻

Evaluation

Evaluation results are included in:

./evaluation results


⸻

Citation

If you use this repository, please cite the associated paper.

@inproceedings{george-etal-2025-enhancing,
  title = "Enhancing Financial {RAG} with Agentic {AI} and Multi-{H}y{DE}: A Novel Approach to Knowledge Retrieval and Hallucination Reduction",
  author = "George, Ryan and
            Srinivasan, Akshay Govind and
            Joe, Jayden Koshy and
            R, Harshith M and
            J, Vijayavallabh and
            Kant, Hrushikesh and
            Vimalkanth, Rahul and
            S, Sachin and
            Suresh, Sudharshan",
  booktitle = "Proceedings of The 10th Workshop on Financial Technology and Natural Language Processing",
  month = nov,
  year = "2025",
  address = "Suzhou, China",
  publisher = "Association for Computational Linguistics",
  pages = "19--32",
  doi = "10.18653/v1/2025.finnlp-2.3",
  url = "https://aclanthology.org/2025.finnlp-2.3/"
}

