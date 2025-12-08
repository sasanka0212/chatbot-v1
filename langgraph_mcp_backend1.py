from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain.tools import BaseTool
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
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

load_dotenv()

if sys.platform.startswith("win"):
    # Psycopg async does NOT support ProactorEventLoop on Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Dedicated async loop for backend tasks
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


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

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

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
    
tools = load_mcp_tools()
""" async def build_graph(): """
print(tools)

ll_with_tools = llm.bind_tools([calculator, stock_price, search_tool, *tools])
async def chat_node(state: ChatState):
    """
    LLM Node may answer the question or requests for a call.
    """
    messages = state['messages']
    response = ll_with_tools.invoke(messages)
    return {'messages': [response]}

tool_node = ToolNode([calculator, search_tool, stock_price, *tools])

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