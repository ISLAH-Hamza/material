from langchain_chroma import Chroma
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["OPENAI_API_KEY"]="""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"""

llm=OpenAI(temperature=0.5)


loader=PyPDFLoader(file_path='./code_commerce_maroc.pdf')
docs=loader.load()

spliter=RecursiveCharacterTextSplitter()

chunks=spliter.split_documents(docs)

vectors_db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory="./db")


retriver=vectors_db.as_retriever(k=4)

print(retriver.invoke('SARL'))
