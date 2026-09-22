import os

from dotenv import load_dotenv
load_dotenv() 

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.tools import tool
from langchain.agents import create_agent 
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st



if "document_uploaded" not in st.session_state:
    st.session_state["document_uploaded"] = False

if "agent" not in st.session_state:
    st.session_state["agent"] = None
    
if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None

if "messages" not in st.session_state:
    st.session_state["messages"] = []



def process_document(path):
    # Load the Document
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()


    # Split the Document into Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents=docs)


    # Embeddings and VectorDB
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    vector_db = InMemoryVectorStore.from_documents(docs, embeddings)


    # Create Agent - tool , llm , prompt
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")


    @tool
    def retrieve_context(query:str):
        """
        Retrieve documents relevant to relevant to the query from the knowledge base.
        """
        context = ""
        docs = vector_db.similarity_search(query, k=3)
        
        for doc in docs:
            context += doc.page_content + "\n\n"
            
        return context


    system_prompt = """
    You are a helpful assistant that answers questions based on the context provided.
    My knowledge base consists of the details from the uploaded document.
    Always use `retrieve_context` tool for questions requiring external knowledge.
    """


    memory = InMemorySaver()

    agent = create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt=system_prompt, 
        checkpointer=memory
    )
    
    st.session_state.agent = agent
    st.session_state.document_uploaded = True


# Upload UI
if not st.session_state.document_uploaded:
    uploaded = st.file_uploader(label="Select PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        with st.spinner("Processing..."):
            path = "./doc_files/"
            os.makedirs(path, exist_ok=True)
            for file in uploaded:
                with open(path + file.name, "wb") as f:
                    f.write(file.getvalue())
                    
            process_document(path)
            st.rerun()


# Chat UI
if st.session_state.document_uploaded and st.session_state.agent:
    for message in st.session_state.messages:
        role = message.get("role")
        content = message.get("content")
        st.chat_message(role).markdown(content)
    
    query = st.chat_input("Ask me anything about the uploaded document...")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)
        
        # FIX: Added st.spinner wrapper around the model execution task
        with st.chat_message("ai"):
            with st.spinner("Thinking..."):
                response = st.session_state.agent.invoke(
                    {"messages": [{"role": "user", "content": query}]},
                    {"configurable": {"thread_id": 1}}
                )
                result = response["messages"][-1].text
                st.markdown(result)
                
        st.session_state.messages.append({"role": "ai", "content": result}) 
