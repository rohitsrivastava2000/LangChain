from rag import vectorstore,llm

def ask_question(question:str,k:int=3):
    """Ask a question using the RAG setup"""

    # Retrieve relevant documents
    relevant_docs = vectorstore.similarity_search(question, k=k)

    # Combine retrieved chunks into context
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    print(f"This is the context: {context}")
    # Construct prompt for LLM
    prompt = f"""You are a helpful assistant. Use the following context to answer the question.Context:{context} Question: {question} Answer in a clear and concise way."""
    response = llm.invoke(prompt)
    return response.content


if __name__=="__main__":
    while True:
        query=input("Ask a Question -> ")
        if query.lower()=='exit':
            break

        answer=ask_question(query)
        print(f"AI answer -> {answer}")

