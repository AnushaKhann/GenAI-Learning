from dotenv import load_dotenv
load_dotenv()  

from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

st.title("🤖 AskBuddy - AI QnA Bot")
st.markdown("My QnA bot is powered by Google Gemini 3.5 model. You can ask any question and it will answer you.")

query = st.chat_input("Ask me anything...")

if "messages" not in st.session_state:
    st.session_state.messages = []
    
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)
    
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").markdown(query)
    res = llm.invoke(query)
    
    if isinstance(res.content, list):
        answer_text = "".join([item.get("text", "") for item in res.content if isinstance(item, dict)])
    else:
        answer_text = str(res.content)
        
    st.chat_message("ai").markdown(answer_text)
    st.session_state.messages.append({"role": "ai", "content": answer_text})