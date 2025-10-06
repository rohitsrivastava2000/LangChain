from dotenv import load_dotenv
import os
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolArg
from typing import Annotated

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


# create tool

# Conversion rate tool
@tool
def get_conversion_factor(base_currency:str,target_currency:str) -> float:
    """
    This function fetches the currency factor between a given base currency and a target currency
    """
    url=f'https://v6.exchangerate-api.com/v6/3a8647a704b4f3dd0fe54b50/pair/{base_currency}/{target_currency}'
    response = requests.get(url)
    return response.json()['conversion_rate']

# Convert into target currency
@tool
def convert(base_currency_value:int, conversion_rate:float) -> float:
    """
    Given a currency conversion rate this function calculates currency value from a given base currency value
    """
    return base_currency_value * conversion_rate


# Calling the to find the conversion rate
result = get_conversion_factor.invoke({'base_currency':'USD','target_currency':'INR'})
print("Get Conversion Rate -->> ",result)

# Convert the currency
mainResult = convert.invoke({"base_currency_value":10, "conversion_rate":85.16})
# print(mainResult)


# Binding this two tool with the llm 
llm_with_tool = llm.bind_tools([get_conversion_factor,convert])

query="What is teh conversion factor b/w USD and INR, based on that can you convert 10 usd to inr "
messages=[HumanMessage(query)]


ai_message = llm_with_tool.invoke(messages)
messages.append(ai_message) # Add AI message to history
print("Step 1 AI Decision:", ai_message.tool_calls)


tool_call = ai_message.tool_calls[0]

if tool_call['name']=='get_conversion_factor':
    tool_result=get_conversion_factor.invoke(tool_call['args'])
    print("Step 1 Tool Result (Conversion Rate):", tool_result)

    # Step 3: Feed the clean tool result back into messages
    messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"]))

    # Step 4: Ask LLM again → now it has the rate and can call the second tool
    print("\n--- Invoking LLM for Step 2 ---")
    ai_message2 = llm_with_tool.invoke(messages)
    messages.append(ai_message2) # Add AI message to history
    print("Step 2 AI Decision:", ai_message2.tool_calls)

    # Step 5: Run the second tool call
    tool_call2 = ai_message2.tool_calls[0]
    if tool_call2["name"] == "convert":
        # The LLM now correctly identifies both arguments from the conversation history
        tool_result2 = convert.invoke(tool_call2["args"])
        print("Step 2 Tool Result (Converted Value):", tool_result2)

        # Step 6: Feed the final result back into the conversation
        messages.append(ToolMessage(content=str(tool_result2), tool_call_id=tool_call2["id"]))
        
        print("\n--- Generating Final Answer ---")
        final_answer = llm_with_tool.invoke(messages)
        print("\nFinal Answer:", final_answer.content)














