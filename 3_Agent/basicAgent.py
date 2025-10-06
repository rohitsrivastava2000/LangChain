from dotenv import load_dotenv
import os
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import tool, DuckDuckGoSearchRun

from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError(" GOOGLE_API_KEY not found. Please check your .env file.")

# > Use an available model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   
    temperature=0.7,
    api_key=api_key
)

search_tool = DuckDuckGoSearchRun()

@tool
def get_wether_data(city:str)->str:
    """This function fetch the current data for a given city"""
    
    #TODO: Simple put you weater api url here
    url=""
    response=requests.get(url)

    return response.json()

# Pulling the ReAct prompt from langchain hub
prompt = hub.pull('hwchase17/react')


# Create the ReAct agent manually with the pulled prompt
agent = create_react_agent(
    llm=llm,
    tools=[search_tool,get_wether_data],
    prompt=prompt
)

# Wrap with it AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool,get_wether_data],
    verbose=True
)

# Invoke
response = agent_executor.invoke({"input":"Find the capital of Uttar Pradesh, then find it's current weather condition"})
print(response)