from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain.tools import BaseTool
from typing import TypedDict, Annotated, Dict, Any, Optional
from langgraph.graph.message import add_messages
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import asyncio

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient

import psycopg
import requests
import os
import threading
import sys
import tempfile
import streamlit as st

load_dotenv()

if sys.platform.startswith("win"):
    # Psycopg async does NOT support ProactorEventLoop on Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Dedicated async loop for backend tasks
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()

# PDF retriever store(per thread)
""" _THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {} """

# Persist across Streamlit reruns / Added on 11-04-2026
if "retrievers" not in st.session_state:
    st.session_state["retrievers"] = {}

if "metadata" not in st.session_state:
    st.session_state["metadata"] = {}

_THREAD_RETRIEVERS = st.session_state["retrievers"]
_THREAD_METADATA = st.session_state["metadata"]

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and str(thread_id) in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[str(thread_id)]
    return None

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass

def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)

def run_async(coro):
    return _submit_async(coro).result()

def submit_async_task(coro):
    # Schedule a coroutine on the backend event loop.
    return _submit_async(coro)

search_tool = DuckDuckGoSearchRun(response_format='content')

@tool
def calculator(num1: int, num2: int, operation: str) -> dict:
    """
    Do the mentioned arithmatic operation between two numbers.
    For example: add 19 with 6, the num1 contains 19, num2 contains 6 and operation is add.
    supported operations: add, sub, mult, div
    """
    try:
        if operation == 'add':
            result = num1 + num2
        elif operation == 'sub':
            result = num1 - num2
        elif operation == 'mult':
            result = num1 * num2
        elif operation == 'div':
            if num2 == 0:
                return {'error': 'division by zero is not allowed'}
            result = num1 / num2
        else:
            return {'errror': f'unsupported operation {operation}'}
        return {
            'num1': num1,
            'num2': num2,
            'operation': operation,
            'result': result
        }
    except Exception as e:
        return {'error': str(e)}
    
@tool
def stock_price(symbol: str):
    """
    Fetch stock price by the given symbol(e.g. AAPL, TSLA)
    Use given API key for fetch the stock price details.
    """
    url =  f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={os.environ['STOCK_API_KEY']}"
    response = requests.get(url=url)
    return response.json()

@tool
def rag_tool(query: str, config: RunnableConfig) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling the tool.
    """
    thread_id = config.get("configurable", {}).get("thread_id")
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first",
            "query": query
        }
    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    """ return {
        "query": query,
        "context": context,
        "metadata": metadata, 
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename")
    } """

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename")
    }

# build MCP client
client = MultiServerMCPClient(
    {
        "expense": {
            "transport": "streamable_http",
            "url": "https://sheer-copper-tarantula.fastmcp.app/mcp"
        }
    }
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(client.get_tools())
    except Exception:
        return []
    
mcp_tools = load_mcp_tools()
""" async def build_graph(): """
print(mcp_tools)

tools = [calculator, stock_price, search_tool, rag_tool, *mcp_tools]

ll_with_tools = llm.bind_tools(tools)
async def chat_node(state: ChatState, **kwargs):
    """
    LLM Node may answer the question or requests for a call.
    """
    config: RunnableConfig = kwargs.get("config", {})
    messages = [
        SystemMessage(content="""
            You are a helpful assistant.

            If a document is uploaded, ALWAYS use rag_tool to answer questions about it.
            Do NOT answer from general knowledge if the answer exists in the document.
            """),
        *state['messages']
    ]
    #messages = state['messages']
    #thread_id = config.get("configurable", {}).get("thread_id")
    response = ll_with_tools.invoke(
        messages,
        config=config
    )
    return {'messages': [response]}

tool_node = ToolNode(tools)

# Estalish the database connection
#conn = sqlite3.connect('chatarena.db', check_same_thread=False)

DB_URI = os.environ["DB_URL"]
#checkpoint = SqliteSaver(conn=conn)

async def _init_checkpointer():
    conn = await psycopg.AsyncConnection.connect(DB_URI)
    await conn.set_autocommit(True)
    checkpoint = AsyncPostgresSaver(conn)
    await checkpoint.setup()
    return checkpoint

""" async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpoint:
    checkpoint.setup() """

graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')

if tool_node:
    graph.add_node('tools', tool_node)
    graph.add_conditional_edges('chat_node', tools_condition)
    graph.add_edge('tools', 'chat_node')
else:
    graph.add_edge('chat_node', END)

#chatbot = graph.compile(checkpoint)
checkpointer = run_async(_init_checkpointer())
chatbot = graph.compile(checkpointer)

async def _alist_threads():
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)

def retrieve_all_threads():
    return run_async(_alist_threads())

def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS

def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})

# test purpose
""" config = {'configurable': {'thread_id': 'thread-1'}}
for message, metadata in chatbot.stream(
    {'messages': HumanMessage(content='Generate a 200 words essay about Royal Bengal Tiger')},
    config=config,
    stream_mode='messages'
) :
    if message.content:
        print(message.content, end='|', flush=True) """

""" conn.row_factory = sqlite3.Row
cursor = conn.cursor()
cursor.execute("drop table writes") """

""" def fetch_all_threads():
    all_threads = set()
    for item in checkpoint.list(None):
        all_threads.add(item.config['configurable']['thread_id'])

    print(list(all_threads)) """

""" config = {
    'configurable': {'thread_id': 1},
    'metadata': {'thread_id': 1},
    'run_name': 'chat-turn'
}


async def main():
    chatbot = await build_graph()
    response = await chatbot.ainvoke({'messages': [HumanMessage("List all expenses for the december 2025")]}, config=config)
    print(response['messages'][-1].content)

if __name__ == "__main__":
    asyncio.run(main()) """