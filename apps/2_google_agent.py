import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
search = GoogleSerperAPIWrapper()
memory  = MemorySaver()

agent = create_agent(
    model=llm,
    tools=[search.run],
    checkpointer=memory,
    system_prompt="You are a helpful assistant that can answer questions using Google Search results.",
)

while True:
    query = input("User: ")
    if query.lower() in ["exit", "quit"]:
        print("Exiting the agent. Goodbye!")
        break
    
    if not query.strip():
        continue
        
    response = agent.invoke(
                {"messages": [{"role": "user", "content": query}]},
                {"configurable": {"thread_id": "1"}}
            )
    print("AI:", response["messages"][-1].text )
