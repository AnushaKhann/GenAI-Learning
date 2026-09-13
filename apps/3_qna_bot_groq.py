from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

llm = ChatGroq(model="openai/gpt-oss-20b", streaming=True) # Streaming is enabled for real-time response generation
search = GoogleSerperAPIWrapper()
tool = [search.run]

# Initialize session state for memory and history if not already present
if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history = []

agent = create_agent(
    model=llm,
    tools=tool,
    checkpointer=st.session_state.memory,
    system_prompt="You are a helpful assistant that can answer questions using Google Search results."
)

st.subheader("QuickAnswer - Answer at the speed of thought")

for message in st.session_state.history:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

query = st.chat_input("Ask me Anything...")

if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role": "user", "content": query}) # Stores Memory of the user query
    
    response = agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        {"configurable": {"thread_id": "1"}},
        stream_mode="messages"
    )
    
    ai_container = st.chat_message("ai")
    with ai_container:
        space = st.empty()
        message = ""
        
        for chunk in response:
            message += chunk[0].text
            space.write(message)
        
        st.session_state.history.append({"role": "ai", "content": message}) # Stores Memory of the AI response
    