from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd
import os

def retrieve_embedded_reviews(file_path: str, search_k: int = 3) -> any:
    """
    This function retrieves content from a csv file, embeds it using OllamaEmbeddings,
    and stores it in a Chroma vector database. Finally, it returns a retriever object for querying the vector store.
    Args:
        file_path (str): Path to the CSV file.
        search_k (int): Number of top similar documents to retrieve. Default is 3.
    Returns:
        retriever: A retriever object for querying the Chroma vector store.
    """

    # Load dataset
    df = pd.read_csv(file_path)

    # Embedding
    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large"
    )
    # Chroma DB location
    db_location = "./chroma_db"

    # Check if DB exists
    add_document = not os.path.exists(db_location)

    # Create documents and IDs if DB does not exist
    if add_document:
        documents = []
        ids = []
        for index, row in df.iterrows():
            document = Document(
                page_content = row["Title"] + " " + row["Review"],
                metadata = {"rating": row["Rating"], "date": row["Date"]},
                id = str(index)
            )
            documents.append(document)
            ids.append(str(index))

    vector_store = Chroma(
        collection_name="restaurant_reviews",
        embedding_function=embeddings,
        persist_directory=db_location
    )
    if add_document:
        vector_store.add_documents(documents=documents, ids=ids)

    # Retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": search_k})
    return retriever


