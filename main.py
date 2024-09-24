import os
from typing import Literal, List

from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_fireworks import ChatFireworks
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph, START
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["FIREWORKS_API_KEY"] = os.getenv("FIREWORKS_API_KEY", "")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")
os.environ["USER_AGENT"] = os.getenv("USER_AGENT", "rag-routing-app/0.1")



# Initialize Streamlit
st.set_page_config(page_title="Question Answering System")

# ==================== Data and Models ====================

# Load documents (URLs should be updated for your use case)
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Text Splitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Vectorstore
temp_dir = tempfile.mkdtemp()

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    embedding=embeddings,
    persist_directory=temp_dir,
    collection_name="rag-chroma" 
)
retriever = vectorstore.as_retriever()

# LLMs
llm_router = ChatFireworks(model="accounts/fireworks/models/mixtral-8x22b-instruct")
simple_llm = ChatFireworks(model="accounts/fireworks/models/mixtral-8x7b-instruct")
complex_llm = ChatFireworks(
    model="accounts/fireworks/models/llama-v3p1-70b-instruct"
)

# RAG Prompt
prompt = hub.pull("rlm/rag-prompt")

# Web Search Tool
web_search_tool = TavilySearchResults(k=3)


# ==================== Pydantic Models ====================


class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web_search or a vectorstore.",
    )


class Question_type(BaseModel):
    """Route a user query to the most relevant Type"""
    Question_type: Literal["Simple", "Complex"] = Field(
        ...,
        description="Given a user question choose to route it to Simple or Complex.",
    )


class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    complexity_grade: str


# ==================== Langchain Components ====================


structured_llm_router = llm_router.with_structured_output(RouteQuery)
system_router = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. Otherwise, use web-search."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_router),
        ("human", "{question}"),
    ]
)
question_router = route_prompt | structured_llm_router

structured_llm_type = llm_router.with_structured_output(Question_type)
system_type = """You are a chatbot assessing the complexity of a user's question.
If the question is straightforward and easy to understand, rate it as 'Simple'.
If the question is complex and requires detailed understanding, rate it as 'Complex'.
Output a binary score of 'Simple' or 'Complex' only."""
type_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_type),
        ("human", "{question}"),
    ]
)
question_type_router = type_prompt | structured_llm_type

rag_chain_simple = prompt | simple_llm | StrOutputParser()
rag_chain_complex = prompt | complex_llm | StrOutputParser()


# ==================== Workflow Functions ====================


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def retrieve(state):
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def generate_simple(state):
    print("---GENERATE SIMPLE---")
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain_simple.invoke(
        {"context": format_docs(documents), "question": question}
    )
    return {"documents": documents, "question": question, "generation": generation}


def generate_complex(state):
    print("---GENERATE COMPLEX---")
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain_complex.invoke(
        {"context": format_docs(documents), "question": question}
    )
    return {"documents": documents, "question": question, "generation": generation}


def web_search(state):
    print("---WEB SEARCH---")
    question = state["question"]
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    return {"documents": [web_results], "question": question}


def route_question(state):
    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"


def question_type(state):
    print("---ASSESS COMPLEXITY---")
    question = state["question"]
    score = question_type_router.invoke({"question": question})
    grade = score.Question_type
    state["complexity_grade"] = grade
    return state


# ==================== LangGraph Workflow ====================

workflow = StateGraph(GraphState)

workflow.add_node("web_search", web_search)
workflow.add_node("question_type", question_type)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate_simple", generate_simple)
workflow.add_node("generate_complex", generate_complex)

workflow.add_conditional_edges(
    START,
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
    },
)

workflow.add_edge("web_search", "generate_simple")  # Assuming web search is simpler
workflow.add_edge("retrieve", "question_type")

workflow.add_conditional_edges(
    "question_type",
    lambda state: state["complexity_grade"],  
    {
        "Simple": "generate_simple",  
        "Complex": "generate_complex",
    },
)

workflow.add_edge("generate_simple", END)
workflow.add_edge("generate_complex", END)

app = workflow.compile()

# ==================== Streamlit App ====================


def main():
    st.title("Question Answering System")

    question = st.text_input("Enter your question:")
    if st.button("Submit"):
        if question:
            with st.spinner("Thinking..."):
                for output in app.stream({"question": question}):
                    for key, value in output.items():
                        st.write(f"**Node '{key}':**")
                    st.write("---")

                st.success(value["generation"])  # Display the final generation
        else:
            st.warning("Please enter a question.")


if __name__ == "__main__":
    main()