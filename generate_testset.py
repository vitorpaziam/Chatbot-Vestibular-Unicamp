import openai
import pandas as pd
import os
import random

from data_processing import get_pdf_text, get_text_chunks
from constants import DATA_FILE, DATASET_DIRECTORY

# Query to generate a question
QUESTION_GENERATOR_QUERY = "Escreva uma pergunta com o conteúdo contido no trecho: {}"
# Query to generate an answer based on the question and a block of text
ANSWER_GENERATOR_QUERY = "Escreva uma resposta sucinta para a pergunta '{}' utilizando os dados do trecho '{}'" 


def generate_question_and_answer(query: str) -> str:
    """
    Generate a response from the query using OpenAI's ChatCompletion API.

    Args:
        query (str): The query for generating the question and answer.

    Returns:
        str: Content of the generated response from query.
    """

    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [
            {"role": "system", "content": "Você é um assistente que deve formular perguntas e também responder as dúvidas referentes ao processo de Admissão na Unicamp 2024."},
            {"role": "user", "content": query}],
        max_tokens = 250,
        temperature = 0.2, 
        stop = None
    )
    return response.choices[0].message['content'].strip()


def create_testset_csv(text_blocks: list):
    """
    Create a CSV file containing the generated questions and answers.

    Args:
        text_blocks (list): The blocks of text used to generate questions and answers.
    """

    data = {'chunk': [], 'generated_question': [], 'generated_answer': []}
    
    for _ in range(5): # Append 5 lines of data in testset.csv (limit of timout)
        i = random.randint(0, len(text_blocks) - 1) # Choose a pseudo-random block of text
        block = text_blocks[i]
        if len(block) > 0:
            question = generate_question_and_answer(QUESTION_GENERATOR_QUERY.format(block))
            data["chunk"].append(block)
            data["generated_question"].append(question)
            answer = generate_question_and_answer(ANSWER_GENERATOR_QUERY.format(question, block))
            data["generated_answer"].append(answer)
    
    df_to_append = pd.DataFrame(data)

    # If the file already exists, concatenate the new values
    if os.path.exists(f"{DATASET_DIRECTORY}/testset.csv"):
        df = pd.read_csv(f"{DATASET_DIRECTORY}/testset.csv")
        df = df.concat([df, df_to_append], ignore_index = True)
    else:
        df = df_to_append

    df.to_csv(f"{DATASET_DIRECTORY}/testset.csv", index=False)


def main():
    openai.api_key = "<YOUR_OPENAI_API_KEY>"
    raw_text = get_pdf_text(DATA_FILE)
    text_blocks = get_text_chunks(raw_text, chunk_size = 2000, chunk_overlap = 200)
    create_testset_csv(text_blocks)

main()