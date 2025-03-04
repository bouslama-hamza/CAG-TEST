from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import PromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv
import os
import shelve

# Load environment variables
load_dotenv()

# Set up Streamlit interface with a professional title
st.title("Brand Impact Chatbot")

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Functions to load and save chat history
def load_chat_history():
    """Load the conversation history from a shelve file."""
    with shelve.open("chat_history") as db:
        return db.get("conversations", [])

def save_chat_history(conversations):
    """Save the conversation history to a shelve file."""
    with shelve.open("chat_history") as db:
        db["conversations"] = conversations

# Initialize session state for conversations
if "conversations" not in st.session_state:
    st.session_state.conversations = load_chat_history()
    if not st.session_state.conversations:
        st.session_state.conversations = [{"title": "New Conversation", "messages": []}]
    st.session_state.current_conversation = 0

# Sidebar for conversation management
with st.sidebar:
    # Button to start a new conversation
    if st.button("New Conversation"):
        new_conv = {"title": "New Conversation", "messages": []}
        st.session_state.conversations.append(new_conv)
        st.session_state.current_conversation = len(st.session_state.conversations) - 1

    # Selectbox to choose an existing conversation
    selected_conv = st.selectbox(
        "Conversations",
        options=range(len(st.session_state.conversations)),
        format_func=lambda i: st.session_state.conversations[i]["title"],
        index=st.session_state.current_conversation,
        key="conversation_selectbox"
    )
    st.session_state.current_conversation = selected_conv

    # Button to clear all conversations
    if st.button("Clear All Conversations"):
        st.session_state.conversations = [{"title": "New Conversation", "messages": []}]
        st.session_state.current_conversation = 0
        save_chat_history(st.session_state.conversations)

# Display messages from the current conversation
current_conv = st.session_state.conversations[st.session_state.current_conversation]
for message in current_conv["messages"]:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Custom prompt template for the assistant
custom_prompt = PromptTemplate(
    "You are a helpful assistant. Respond in the same language as the question. "
    "For other questions, use only the following context to answer and do not use any external knowledge:\n"
    "{context_str}\n"
    "If the context does not provide the information needed to answer the question, say that you do not have enough information to provide an answer. "
    "Question: {query_str}\n"
    "Answer:"
)

# Prediction function using LlamaIndex query engine
def answerMe(question):
    """Generate a response to the user's question using the stored index."""
    storage_context = StorageContext.from_defaults(persist_dir='store')
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine(
        text_qa_template=custom_prompt,
        llm=ChatOpenAI(temperature=0.0)  # Low temperature for deterministic responses
    )
    response = query_engine.query(question)
    return str(response)

# Chat input interface
if prompt := st.chat_input("How can I help?"):
    # Update the conversation title if it's the first message
    if len(current_conv["messages"]) == 0:
        current_conv["title"] = prompt  # Set title to the first user message
    
    # Append the user's message to the current conversation
    current_conv["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    # Generate and display the assistant's response
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        full_response = answerMe(prompt)
        message_placeholder.markdown(full_response)
    current_conv["messages"].append({"role": "assistant", "content": full_response})
    
    # Save the updated conversation history
    save_chat_history(st.session_state.conversations)