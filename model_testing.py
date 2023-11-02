import pandas as pd

from data_processing import pdf_data_process
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOC_DATA_FILE = "vestibular-data.pdf"


def generate_validation_data(conversation: ConversationalRetrievalChain, file_name: str):
    """
    Generate a CSV file that contains the validation data for the model by evaluating 
    responses and calculating similarity scores.

    Args:
        conversation (ConversationalRetrievalChain): The conversation retrieval chain.
        file_name (str): The name of the file.
    """

    # Load the test dataset with previous answers if it exists, otherwise load the test dataset without validation responses
    try:
        test_data = pd.read_csv(f"dataset/{file_name}_with_model_answers.csv")
    except FileNotFoundError:
        test_data = pd.read_csv(f"dataset/{file_name}.csv")

    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Iterate through each question and answer in the test dataset
    for index, row in test_data.iterrows():
        user_question = row['generated_question']
        expected_answer = row['generated_answer']

        # Get the response from the chatbot
        response = conversation({'question': user_question})
        model_answer = response['chat_history'][-1].content

        # Vectorize the answers
        expected_answer_vectorized = vectorizer.fit_transform([expected_answer])
        model_answer_vectorized = vectorizer.transform([model_answer])

        # Calculate the cosine similarity between the generated and expected answers
        similarity_score = cosine_similarity(model_answer_vectorized, expected_answer_vectorized)

        # Update the test dataset with the generated response and the similarity_score value
        test_data.at[index, 'model_answer'] = str(model_answer)
        test_data.at[index, 'model_answer_score'] = similarity_score.max()

    # Save the updated dataset to a new CSV file
    test_data.to_csv(f"dataset/{file_name}_with_model_answers.csv", index = False, float_format = '%.3f')


def get_conversation_chain(vector_store: FAISS) -> ConversationalRetrievalChain:
    """
    Retrieves a conversation chain for the vector store.

    Args:
        vector_store (FAISS): The vector store of a data file.

    Returns:
        conversation_chain (ConversationalRetrievalChain): The conversation retrieval chain.
    """

    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages = True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vector_store.as_retriever(),
        memory = memory
    )
    return conversation_chain


def setup_model():
    """
    Set up the model by initializing the vector store and conversation chain, and create data to validate the model.
    """

    vector_store = pdf_data_process(DOC_DATA_FILE)
    conversation = get_conversation_chain(vector_store)

    generate_validation_data(conversation, "testset_direct")

setup_model()   