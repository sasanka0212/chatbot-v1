from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
import sqlite3

load_dotenv()

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}

# Estalish the database connection
conn = sqlite3.connect('chatarena.db', check_same_thread=False)

checkpoint = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

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