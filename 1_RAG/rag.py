from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


#Load the pdf
loader = PyPDFLoader("python_Core_book.pdf")
documents =loader.load()

# Filter only pages 1–300
selected_docs = [doc for doc in documents if 1 <= doc.metadata["page"] <= 300]

print("Total pages loaded:",len(selected_docs))

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,   # length of each chunk
    chunk_overlap=200  # overlap between chunks for context
)

docs = text_splitter.split_documents(selected_docs)
print("Total chunks:", len(docs))

# Embedding the Chunks
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Store chunks in Chroma
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")

# Save to disk
vectorstore.persist()

print("✅ Stored embeddings in ChromaDB")


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


   

if __name__ == "__main__":
    basic_chat()
