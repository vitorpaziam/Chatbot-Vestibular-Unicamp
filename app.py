from data_processing import pdf_data_process
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

def main():
    vector_store = pdf_data_process("vestibular-data.pdf")
    
    query = input("Faça uma pergunta: ")
    
    if query:
        docs = vector_store.similarity_search(query=query, k=3)

        llm = OpenAI()
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=query)
            print(cb)
        
        print(response)

main()