from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

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

    response = llm.invoke("Tell me a poem about the monsoon in India.")
    print("Response:", response.content)

if __name__ == "__main__":
    basic_chat()
