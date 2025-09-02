from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import main

embeddings = HuggingFaceEmbeddings()

file_path = main.file_path 

loader = UnstructuredPDFLoader(file_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2000,
    chunk_overlap = 500,
    separators = ["\n", " ", ".", ",", "!", "?"]
)

text_chunks = text_splitter.split_documents(documents)

vector_db = Chroma.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    persist_directory='vector_db_dir'
)