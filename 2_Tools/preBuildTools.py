from langchain_community.tools import DuckDuckGoSearchRun

search_tool=DuckDuckGoSearchRun()

result=search_tool.invoke('asia cup news, IND VS PAK')
print(result)