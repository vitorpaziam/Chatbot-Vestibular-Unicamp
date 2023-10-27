import streamlit as st

from data_processing import pdf_data_process
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

DATA_FILE = "vestibular-data.pdf"


def get_conversation_chain(vector_store: FAISS) -> ConversationalRetrievalChain:
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages = True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vector_store.as_retriever(),
        memory = memory
    )
    return conversation_chain


def handle_userinput(user_question: str):
    response = conversation({'question': user_question})
    st.session_state.chat_history.append(response['chat_history'])

    for i in range(len(st.session_state.chat_history)):
        for j, message in enumerate(st.session_state.chat_history[i]):
            if j % 2 == 0:
                with st.chat_message("user"):
                    st.write(message.content)
            else:
                with st.chat_message("assistant"):
                    st.write(message.content)


def main():

    st.set_page_config(
        page_title = "Chatbot - Admissão Unicamp 2024",
        page_icon = "🎓",
        layout = "wide",
        initial_sidebar_state = "expanded",
        menu_items = {
            'Get Help': 'https://github.com/vitorpaziam',
            'About': '''O chatbot foi concebido utilizando a tecnologia do modelo de LLM da OpenAI, através do processamento do arquivo 
                da *Resolução do Vestibular Unicamp 2024*. Oferece uma _experiência interativa_, permitindo que os usuários façam perguntas 
                sobre os procedimentos de inscrição, documentos necessários, datas de exames, políticas de cotas e outras dúvidas comuns. 
                Empregando técnicas derivadas de *inteligência artificial*, o chatbot busca proporcionar orientações claras e precisas. 
                Com uma interface amigável e acessível, o objetivo é *simplificar o processo de admissão* e facilitar a navegação dos 
                candidatos pelo complexo processo de seleção da *Unicamp*.'''
        }
    )

    st.header("🎓 Chatbot - Admissão Unicamp 2024")

    st.markdown('''Bem-vindo ao __*Chatbot de Admissão Unicamp 2024*!__ Desenvolvido com base na [*Resolução GR-031/2023*](https://www.pg.unicamp.br/norma/31594/0), 
    de 13/07/2023, que "Dispõe sobre o _Vestibular Unicamp 2024_ para vagas no ensino de Graduação", tem como objetivo 
    oferecer suporte aos candidatos interessados no processo seletivo da __Universidade Estadual de Campinas (Unicamp)__ 
    para o ano de 2024. Projetado para atender às perguntas frequentes sobre o processo de inscrição, requisitos de admissão, 
    prazos cruciais e critérios de seleção, este chatbot também fornece __informações detalhadas__ sobre os cursos oferecidos, 
    as diferentes etapas do processo de admissão e valiosas dicas para que os candidatos possam se preparar de forma abrangente 
    e eficaz.''')

    # Initialize state session variable chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if query := st.chat_input("Escreva sua dúvida..."):
       handle_userinput(query)


if __name__ == "__main__":

    # Initialize vector store and conversation chain
    vector_store = pdf_data_process(DATA_FILE)
    conversation = get_conversation_chain(vector_store)
    
    main()