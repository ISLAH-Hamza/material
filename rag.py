from langchain_chroma import Chroma
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os
import faiss

import os

load_dotenv()

os.environ["OPENAI_API_KEY"]="""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"""


llm = OpenAI(temperature=0.5)

# Load the PDF document
loader = PyPDFLoader(file_path='./code_commerce_maroc.pdf')
docs = loader.load()

# Split the documents
splitter = RecursiveCharacterTextSplitter()
chunks = splitter.split_documents(docs)

# Create embeddings
embeddings = OpenAIEmbeddings()

# Create FAISS index
faiss_index = FAISS.from_documents(chunks, embeddings)

# Set up the retriever
retriever = faiss_index.as_retriever(k=4)

# Invoke the retriever with a query
result = retriever.invoke('SARL')

print(result)



# pip install langchain faiss-cpu openai python-dotenv langchain-community
