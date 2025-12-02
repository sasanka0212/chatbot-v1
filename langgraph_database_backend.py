from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import ToolNode, tools_condition

import sqlite3
import requests
import os

load_dotenv()

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
ll_with_tools = llm.bind_tools([calculator, search_tool, stock_price])

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    """
    LLM Node may answer the question or requests for a call.
    """
    messages = state['messages']
    response = ll_with_tools.invoke(messages)
    return {'messages': [response]}

tool_node = ToolNode([calculator, search_tool, stock_price])

# Estalish the database connection
conn = sqlite3.connect('chatarena.db', check_same_thread=False)

checkpoint = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpoint)

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

def fetch_all_threads():
    all_threads = set()
    for item in checkpoint.list(None):
        all_threads.add(item.config['configurable']['thread_id'])

    print(list(all_threads))