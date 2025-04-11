from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

from src.constant import embeddings
from src.pdf_loaders import load_pdf_and_chunk
from src.llm_setup import *



import os
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("OPEN_API_KEY")



def create_vector_store(documents):
    return FAISS.from_documents(documents=documents, embedding=embeddings)




