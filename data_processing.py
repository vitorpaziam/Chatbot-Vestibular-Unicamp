import os

from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

VECTOR_STORE_DIRECTORY = "vectordb"
DATASET_DIRECTORY = "dataset"


def get_pdf_text(file_name: str) -> str:
    """
    Extracts text from the PDF file.

    Args:
        file_name (str): The filename of the PDF.

    Returns:
        str: The extracted text from the PDF.
    """
    
    file_reader = PdfReader(open(f"{DATASET_DIRECTORY}/{file_name}", "rb"))
    text = ""
    for page in file_reader.pages:
        text += page.extract_text()
    return text


def get_text_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list:
    """
    Splits the text into chunks based on the provided parameters.

    Args:
        text (str): The text to be split into chunks.
        chunk_size (int): The size of each chunk.
        chunk_overlap (int): The overlap between chunks.

    Returns:
        list: A list of text chunks (str).
    """

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap, length_function = len)
    chunks = text_splitter.split_text(text = text)
    return chunks


def pdf_data_process(file_name: str) -> FAISS:
    """
    Processes the data from the PDF file, creating FAISS embeddings.

    Args:
        file_name (str): The filename of the PDF.

    Returns:
        FAISS: The FAISS embeddings.
    """

    load_dotenv()
    embeddings = OpenAIEmbeddings()
    
    # Verify whether the faiss_index has already been created locally
    if os.path.exists(f"{VECTOR_STORE_DIRECTORY}/faiss_index"): 
        vector_store = FAISS.load_local(f"{VECTOR_STORE_DIRECTORY}/faiss_index", embeddings)
    else:
        # Extracting the content of dataset pdf files
        raw_text = get_pdf_text(file_name)

        # Splitting text into chunks with intersections determined by overlap parameter
        chunks = get_text_chunks(raw_text, 1000, 200)

        # Creating embeddings - vectorization of chunks
        vector_store = FAISS.from_texts(chunks, embedding = embeddings)
        vector_store.save_local("{VECTOR_STORE_DIRECTORY}/faiss_index")

    return vector_store