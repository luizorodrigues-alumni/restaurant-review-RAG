from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vector import retrieve_embedded_reviews

load_dotenv()

# Create model
base_url = os.getenv("BASE_URL")
model = ChatOllama(
        base_url=base_url,
        model="llama3:8b"
    )

# Create Prompt Template
system_prompt = """
        You are an expert in aswering questions about a pizza restaurant based on customer reviews. Use the following reviews to answer the questions.
        If you don't know the answer, just say you don't know. Do not make up an answer.
        Reviews:{reviews}
        Human question: {question}
    """
prompt_template = ChatPromptTemplate(
    [
        ("system", system_prompt),
    ]
)

# Generate Chain
chain = prompt_template | model | StrOutputParser()

# User question
question = input("Ask your question: ")

# Retrieve embedded reviews
retriever = retrieve_embedded_reviews(file_path="./files/restaurant_reviews.csv", search_k=5)
reviews = retriever.invoke(question)
chain_result = chain.invoke({"reviews": reviews, "question": question})
print(chain_result)