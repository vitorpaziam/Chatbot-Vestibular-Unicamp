import pickle
import os

from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def pdf_data_process(file_name: str) -> FAISS:

    load_dotenv()
    
    if os.path.exists(f"embeddings/{file_name}.pkl"):
        with open(f"embeddings/{file_name}.pkl", "rb") as f:
            vector_store = pickle.load(f)
    else:
        # Extracting the content of dataset pdf files
        file_reader = PdfReader(open(f"dataset/{file_name}", "rb")) #./dataset/vestibular-data.pdf

        text = ""
        for page in file_reader.pages:
            text += page.extract_text()

        # Splitting text into chunks with intersections determined by overlap parameter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200, length_function = len, )

        # Creating embeddings - vectorization of chunks
        chunks = text_splitter.split_text(text=text)
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        
        with open(f"embeddings/{file_name.split('.')[0]}.pkl", "wb") as f:
            pickle.dump(vector_store, f)

    return vector_store