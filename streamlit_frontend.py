import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage
import uuid

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

# ----------- Session History ----------- #

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = get_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

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

config = {'configurable': {'thread_id': st.session_state['thread_id']}}

# ------------ Chat UI ------------- #
if user_input:
    # developer designed dictionary 
    with st.chat_message('user'):
        st.text(user_input)
        st.session_state['message_history'].append({'role': 'user', 'content': user_input})

    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=config,
                stream_mode='messages',
            )
        )
        st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})