from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.chat import ChatPromptTemplate

load_dotenv()

def basic_chat():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found. Please check your .env file.")

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        api_key=api_key
    )

    # Create chat prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a comedian who tells jokes about {topic} in Hinglish."),
        ("human", "Tell me {joke_count} jokes.")
    ])

    # Fill in the variables
    prompt_value = prompt_template.format_prompt(
        topic="south films",
        joke_count=4
    )

    # Use the LLM to generate messages
    messages = prompt_value.to_messages()
    response = llm.invoke(messages)

    print("AI Response:\n", response.content)

if __name__ == "__main__":
    basic_chat()
