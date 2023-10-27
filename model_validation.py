import pandas as pd

from data_processing import pdf_data_process
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_FILE = "vestibular-data.pdf"


def generate_validation_data(conversation: ConversationalRetrievalChain):
    # Carregar o conjunto de dados de teste com as respostas anteriores, se existirem
    try:
        test_data = pd.read_csv("dataset/testset_with_model_answers.csv")
    except FileNotFoundError:
        test_data = pd.read_csv("dataset/testset.csv")

    # Inicializar o vetorizador TF-IDF
    vectorizer = TfidfVectorizer()

    # Percorrer cada pergunta e resposta no conjunto de dados de teste
    for index, row in test_data[78::].iterrows():
        print(index)
        user_question = row['generated_question']
        expected_answer = row['generated_answer']

        # Obter a resposta do chatbot
        response = conversation({'question': user_question})
        model_answer = response['chat_history'][-1].content

        # Vetorizar as respostas
        expected_answer_vectorized = vectorizer.fit_transform([expected_answer])
        model_answer_vectorized = vectorizer.transform([model_answer])

        # Calcular a similaridade de cosseno entre as respostas geradas e esperadas
        similarity_score = cosine_similarity(model_answer_vectorized, expected_answer_vectorized)

        # Atualizar o conjunto de dados de teste com a resposta gerada e o valor do similarity_score
        test_data.at[index, 'model_answer'] = str(model_answer)
        test_data.at[index, 'model_answer_score'] = similarity_score.max()

    # Salvar o conjunto de dados atualizado em um novo arquivo CSV
    test_data.to_csv("dataset/testset_with_model_answers.csv", index = False, float_format = '%.3f')


def get_conversation_chain(vector_store: FAISS) -> ConversationalRetrievalChain:
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages = True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vector_store.as_retriever(),
        memory = memory
    )
    return conversation_chain


def setup_model():
    # Initialize vector store and conversation chain
    vector_store = pdf_data_process(DATA_FILE)
    conversation = get_conversation_chain(vector_store)

    # Create the data to validate the model
    generate_validation_data(conversation)

setup_model()