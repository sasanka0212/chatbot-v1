import importlib
import json
import contextlib
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from dotenv import load_dotenv
import os

load_dotenv()

# ------------------------------------------------------------------
# IMPORTANT: patch LangGraph Postgres checkpointer to use our pool
# ------------------------------------------------------------------

DB_URI = os.environ["DB_URL"]

@st.cache_resource
def get_pool():
    # Pool persists across Streamlit reruns and provides fresh connections for each operation
    return ConnectionPool(conninfo=DB_URI, min_size=1, max_size=10)

pool = get_pool()

# quick pool smoke test (safe in Streamlit because it's inside cache_resource)
with pool.connection() as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT 1")
        _ = cur.fetchone()

def patch_langgraph_postgres_checkpointer(pool: ConnectionPool):
    """
    Monkey-patch langgraph.checkpoint.postgres.PostgresCheckpointer._cursor
    so it uses the provided ConnectionPool for short-lived connections.

    Accepts arbitrary kwargs such as pipeline, binary, row_factory for compatibility.
    """
    try:
        import langgraph.checkpoint.postgres as lg_pg
    except Exception as e:
        st.warning(f"Could not import langgraph.checkpoint.postgres to patch checkpointer: {e}")
        return False

    @contextlib.contextmanager
    def _cursor(self, *args, **kwargs):
        """
        Replacement _cursor that accepts arbitrary kwargs and uses the pooled connection.

        Known kwargs:
        - row_factory: a row factory (e.g. dict_row)
        - binary: bool
        - pipeline: bool (ignored here but accepted for compatibility)
        """
        # extract known kwargs, keep defaults compatible with original behavior
        row_factory = kwargs.pop('row_factory', dict_row)
        binary = kwargs.pop('binary', False)
        # pipeline flag may be passed by langgraph — accept and ignore (we don't need special behavior)
        kwargs.pop('pipeline', None)

        # Any remaining kwargs we don't explicitly handle are ignored.
        with pool.connection() as conn:
            # create cursor with requested options
            # conn.cursor accepts binary and row_factory in psycopg_pool context
            with conn.cursor(binary=binary, row_factory=row_factory) as cur:
                yield cur

    # Set the _cursor method on the class if it exists
    if hasattr(lg_pg, "PostgresCheckpointer"):
        lg_pg.PostgresCheckpointer._cursor = _cursor
        #st.info("Patched langgraph.checkpoint.postgres.PostgresCheckpointer to use ConnectionPool.")
        return True
    else:
        # fallback: try to patch any class that looks like a checkpointer
        patched_any = False
        for attr_name in dir(lg_pg):
            try:
                attr = getattr(lg_pg, attr_name)
                if isinstance(attr, type):
                    # heuristics: look for classes with 'Checkpointer' in the name or with an existing _cursor
                    if "Checkpointer" in attr.__name__ or hasattr(attr, "_cursor"):
                        attr._cursor = _cursor
                        st.info(f"Patched {attr.__name__} in langgraph.checkpoint.postgres to use ConnectionPool.")
                        patched_any = True
            except Exception:
                continue
        if not patched_any:
            st.warning("Could not find a Checkpointer class to patch in langgraph.checkpoint.postgres.")
        return patched_any

# apply patch early, before we import the backend that will create checkpointer instances
patched = patch_langgraph_postgres_checkpointer(pool)

# ------------------------------------------------------------------
# import the module (we will also import the chatbot symbol for normal use)
# reload it to ensure it picks up our patched checkpointer class
# ------------------------------------------------------------------
import langgraph_database_backend1 as lgdb
lgdb = importlib.reload(lgdb)
from langgraph_database_backend1 import chatbot  # keep the original import for direct use

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
    # return a string for easy storage in st.session_state
    return str(uuid.uuid4())

def reset_chat():
    st.session_state['message_history'] = []
    thread_id = get_thread_id()
    st.session_state['thread_id'] = thread_id
    # initialize a label for the new thread
    add_thread(thread_id=thread_id, label="New chat")
    for message in st.session_state['message_history']:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    # optionally persist a new empty thread to DB via chatbot (not done here)

def add_thread(thread_id, label: str | None = None):
    """
    Ensure thread_id exists in session_state['chat_threads'] and has a label.
    Thread IDs are stored as strings.
    """
    if 'chat_threads' not in st.session_state:
        st.session_state['chat_threads'] = []
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
    if 'thread_labels' not in st.session_state:
        st.session_state['thread_labels'] = {}
    if thread_id not in st.session_state['thread_labels']:
        st.session_state['thread_labels'][thread_id] = truncate_label(label or str(thread_id))

# Use the pool to fetch thread ids from the checkpoints table (initial load only)
def fetch_all_threads():
    values = []
    try:
        with pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("SELECT DISTINCT(thread_id) FROM checkpoints")
                results = cur.fetchall()
                # results may be a list of dicts because of row_factory
                values = [str(res['thread_id']) for res in results if res.get('thread_id') is not None]
    except Exception as e:
        msg = str(e).lower()
        # If table not present, return empty list silently
        if "does not exist" in msg or "relation" in msg or "undefined table" in msg:
            values = []
        else:
            # re-raise unexpected errors so you see them in logs
            raise
    return values

# Helper to try calling chatbot.get_state and recover from a closed DB connection
def safe_get_state(config):
    global chatbot, lgdb
    try:
        return chatbot.get_state(config=config)
    except psycopg.OperationalError as e:
        # connection was closed inside langgraph checkpointer — try to reload module and retry once
        st.warning("DB connection error when reading state — reloading backend and retrying.")
        try:
            # reload the module and rebind chatbot; patched checkpointer will still be used
            lgdb = importlib.reload(lgdb)
            chatbot = getattr(lgdb, "chatbot", chatbot)
        except Exception as reload_err:
            st.error(f"Failed to reload langgraph backend: {reload_err}")
            raise
        # retry once
        return chatbot.get_state(config=config)

def safe_stream_call(request_payload, config, stream_mode='messages'):
    """
    Calls chatbot.stream and if OperationalError occurs, reloads the backend and retries once.
    Returns a generator (or raises).
    """
    global chatbot, lgdb
    try:
        return chatbot.stream(request_payload, config=config, stream_mode=stream_mode)
    except psycopg.OperationalError:
        st.warning("DB connection error during streaming — reloading backend and retrying.")
        lgdb = importlib.reload(lgdb)
        chatbot = getattr(lgdb, "chatbot", chatbot)
        return chatbot.stream(request_payload, config=config, stream_mode=stream_mode)

def get_messages(thread_id: str):
    """
    Load messages for a thread. This intentionally calls chatbot.get_state()
    only when the user explicitly asks to open the thread (e.g. by clicking).
    """
    config = {'configurable': {'thread_id': thread_id}}
    state = safe_get_state(config=config)
    if not hasattr(state, "values") or 'messages' not in state.values:
        return []
    return state.values['messages']

def truncate_label(text, max_len=25):
    text = str(text)
    return text if len(text) <= max_len else text[:max_len - 3] + "..."

# get_first_message is kept but not used in rendering; use only on demand
def get_first_message(thread_id: str):
    config = {'configurable': {'thread_id': thread_id}}
    state = safe_get_state(config=config)
    if not hasattr(state, "values") or 'messages' not in state.values or not state.values['messages']:
        return truncate_label(str(thread_id))
    return truncate_label(state.values['messages'][0].content)

# ----------- Session History ----------- #
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = get_thread_id()

# On first load populate chat_threads and thread_labels from DB (lightweight)
if 'chat_threads' not in st.session_state:
    initial_threads = fetch_all_threads()
    st.session_state['chat_threads'] = []
    st.session_state['thread_labels'] = {}
    for tid in initial_threads:
        add_thread(tid, label=None)

# Ensure the current thread is present
add_thread(st.session_state['thread_id'])

# ----------- Sidebar ------------ #
st.sidebar.title('ChatArena')
if st.sidebar.button('New chat'):
    reset_chat()

st.sidebar.header('Conversation History')

# Render sidebar with labels from session_state only (no DB calls during render)
for i in range(len(st.session_state['chat_threads']) - 1, -1, -1):
    tid = st.session_state['chat_threads'][i]
    label = st.session_state.get('thread_labels', {}).get(tid, truncate_label(tid))
    # st.sidebar.button returns True on click; do NOT call DB while rendering other widgets
    if st.sidebar.button(label, key=f"thread-btn-{tid}"):
        # user clicked the thread: load messages (safe to hit DB now)
        thread_id = tid
        st.session_state['thread_id'] = thread_id

        # Attempt to load messages via chatbot.get_state (keeps your existing reconnection logic)
        try:
            messages = get_messages(thread_id)
        except Exception as e:
            st.error(f"Failed to load thread messages: {e}")
            messages = []

        temp_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                role = 'user'
            else:
                role = 'assistant'
            # message.content may not exist for all objects, guard it
            content = getattr(message, "content", str(message))
            temp_messages.append({'role': role, 'content': content})
        st.session_state['message_history'] = temp_messages

        # update the label for this thread now that we loaded messages
        if temp_messages:
            first = temp_messages[0]['content']
            st.session_state['thread_labels'][thread_id] = truncate_label(first)
        else:
            # leave existing label or set tid
            st.session_state['thread_labels'].setdefault(thread_id, truncate_label(thread_id))

# ----------- User input ---------- #
user_input = st.chat_input('Type here')

config = {
    'configurable': {'thread_id': st.session_state['thread_id']},
    'metadata': {'thread_id': st.session_state['thread_id']},
    'run_name': 'chat-turn'
}

# ------------ Chat UI ------------- #
if user_input:
    with st.chat_message('user'):
        st.text(user_input)
        st.session_state['message_history'].append({'role': 'user', 'content': user_input})

    with st.chat_message('assistant'):
        status_holder = {"box": None}
        def ai_response():
            # use safe_stream_call that will reload backend on OperationalError and retry
            stream_gen = safe_stream_call({'messages': [HumanMessage(content=user_input)]}, config=config, stream_mode='messages')
            for message_chunk, metadata in stream_gen:
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder['box'] is None:
                        status_holder['box'] = st.status(f"🔨 Using {tool_name} ...", expanded=True)
                    else:
                        status_holder['box'].update(label=f"🔨 Using {tool_name} ...", state="running", expanded=True)
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_response)
        if status_holder['box'] is not None:
            status_holder['box'].update(label=f"🔨 Tool finished ...", state="complete", expanded=False)
        st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})

        # Update label for the active thread with assistant's first chunk (if any)
        try:
            if ai_message:
                st.session_state['thread_labels'][st.session_state['thread_id']] = truncate_label(str(ai_message))
        except Exception:
            pass