from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()

def basic_chat():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(" GOOGLE_API_KEY not found. Please check your .env file.")

    # > Use an available model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",   
        temperature=0.7,
        api_key=api_key
    )

    chat_history=[]
    system_message=SystemMessage(content="You are a Software Engineer ,show you only answer the technical question which are only relate to the software field")
    chat_history.append(system_message)

    while True:
        query=input("You:")
        if query.lower()=='exit':
            break
        chat_history.append(HumanMessage(content=query))

        response=llm.invoke(chat_history)
        result=response.content
        chat_history.append(AIMessage(content=result))
        print(f"AI Response: {result}")

    

if __name__ == "__main__":
    basic_chat()
