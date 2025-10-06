from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


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


@tool
def multiply(a:int, b:int) -> int:
    """Multiply Two Number and return the Result"""
    return a * b


# Binding the tools with llm
llm_with_tools = llm.bind_tools([multiply])

query=HumanMessage("hi how are you, can you multiply this two number 5 and 10")

messages=[query]

result=llm_with_tools.invoke(messages)
# print(result)

# print(result.tool_calls[0])

messages.append(result)

## Making a ToolMessage
getToolMessage= multiply.invoke(result.tool_calls[0])
# print(getToolMessage)

messages.append(getToolMessage)

# Now the again to call the llm with all messages, to only show the result

mainAns=llm_with_tools.invoke(messages).content
print(mainAns)

