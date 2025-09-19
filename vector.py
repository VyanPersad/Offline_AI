from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("realistic_restaurant_reviews.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_db"
add_doc = not os.path.exists(db_location)

if add_doc:
    docs = []
    ids =[]

    for i, row in df.iterrows():
        doc = Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata={"rating": row["Rating"], "date":row["Date"]},
            id = str(i)
        )
        ids.append(str(i))
        docs.append(doc)

vector_store = Chroma(
    collection_name="reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_doc:
    vector_store.add_documents(docs, ids=ids)

retriever = vector_store.as_retriever(search_kwargs={"k":5})