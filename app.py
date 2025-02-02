from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings, load_index_from_storage, set_global_service_context
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.indices.vector_store.base import GPTVectorStoreIndex
from langchain_openai import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv
import os
import shelve

load_dotenv()

st.title("Streamlit Chatbot Interface")

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

# Load chat history from shelve file
def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages", [])

# Save chat history to shelve file
def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages

# Initialize or load chat history
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# Sidebar with a button to delete chat history
with st.sidebar:
    if st.button("Delete Chat History"):
        st.session_state.messages = []
        save_chat_history([])

# Display chat messages
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Prediction function
def answerMe(question):
    storage_context = StorageContext.from_defaults(persist_dir='store')
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()
    response = query_engine.query(question)
    return response

# Main chat interface
if prompt := st.chat_input("How can I help?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        full_response = answerMe(prompt)
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Save chat history after each interaction
save_chat_history(st.session_state.messages)
