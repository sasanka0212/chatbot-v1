import streamlit as st
from langgraph_database_backend import chatbot
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid
import sqlite3

st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50; font-size: 32px'>
        🚀 ChatArena
    </h1>
    """,
    unsafe_allow_html=True
)

# ----------- Functionalities ----------- #
def get_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    st.session_state['message_history'] = []
    thread_id = get_thread_id()
    st.session_state['thread_id'] = thread_id
    for message in st.session_state['message_history']:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    add_thread(thread_id=thread_id)

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def get_messages(thread_id):
    config = {'configurable': {'thread_id': thread_id}}
    if 'messages' not in chatbot.get_state(config=config).values:
        return []
    return chatbot.get_state(config=config).values['messages']

def truncate_label(text, max_len=25):
    return text if len(text) <= max_len else text[:max_len - 3] + "..."

def get_first_message(thread_id):
    config = {'configurable': {'thread_id': thread_id}}
    if 'messages' not in chatbot.get_state(config=config).values:
        return truncate_label(str(thread_id))
    return truncate_label(chatbot.get_state(config=config).values['messages'][0].content)

def db_connect():
    conn = sqlite3.connect('chatarena.db', check_same_thread=False)
    return conn

def fetch_all_threads(conn):
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        # Try to fetch all values from the given column
        cursor.execute("select distinct(thread_id) from checkpoints")
        results = cursor.fetchall()
        values = [res['thread_id'] for res in results]
    except sqlite3.OperationalError as e:
        # If table does not exist, return empty list
        if "no such table" in str(e):
            values = []
        else:
            raise  # re-raise other unexpected errors
    finally:
        conn.close()

    return values

# ----------- Session History ----------- #

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = get_thread_id()

conn = db_connect()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = fetch_all_threads(conn=conn)

add_thread(st.session_state['thread_id'])

# ----------- Sidebar ------------ #
st.sidebar.title('ChatArena')
if st.sidebar.button('New chat'):
    reset_chat()

st.sidebar.header('Coversation History')

for i in range(len(st.session_state['chat_threads'])-1, -1, -1):
    if st.sidebar.button(get_first_message(st.session_state['chat_threads'][i])):
        thread_id = st.session_state['chat_threads'][i]
        st.session_state['thread_id'] = thread_id
        messages = get_messages(thread_id)
        temp_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                role = 'user'
            else:
                role = 'assistant'
            temp_messages.append({'role': role, 'content': message.content})
        st.session_state['message_history'] = temp_messages

# ----------- User input ---------- #

user_input = st.chat_input('Type here')

config = {
    'configurable': {'thread_id': st.session_state['thread_id']},
    'metadata': {'thread_id': st.session_state['thread_id']},
    'run_name': 'chat-turn'          
}

# ------------ Chat UI ------------- #
if user_input:
    # developer designed dictionary 
    with st.chat_message('user'):
        st.text(user_input)
        st.session_state['message_history'].append({'role': 'user', 'content': user_input})

    with st.chat_message('assistant'):
        # use a mutable holder so that the generator can set/modify
        status_holder = {"box": None}
        def ai_response():
            for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=config,
                stream_mode='messages',
            ) :
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder['box'] is None:
                        status_holder['box'] = st.status(
                            f"🔨 Using {tool_name} ...", expanded=True
                        )
                    else:
                        status_holder['box'].update(
                            label=f"🔨 Using {tool_name} ...",
                            state="running",
                            expanded=True
                        )

                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content
        ai_message = st.write_stream(ai_response)
        if status_holder['box'] is not None:
            status_holder['box'].update(
                label=f"🔨 Tool finished ...",
                state="complete",
                expanded=False
            )
        st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})