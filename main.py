# main.py

from src.pdf_loaders import load_pdf_and_chunk
from src.llm_setup import setup_llm, get_ques_gen_chain
from src.vector_store import create_vector_store, get_answer_chain
from src.qna import generate_questions, answer_questions

def main():
    file_path = 'Data/Srinivas-JDSA.pdf'
    
    # Step 1: Load and chunk the PDF
    chunks = load_pdf_and_chunk(file_path)
    
    # Step 2: Set up LLM and question generation chain
    llm = setup_llm()
    ques_chain = get_ques_gen_chain(llm)
    
    # Step 3: Generate questions
    questions = generate_questions(ques_chain, chunks)
    
    # Step 4: Set up vector DB and answer chain
    vector_store = create_vector_store(chunks)
    answer_chain = get_answer_chain(llm, vector_store)
    
    # Step 5: Get answers and save
    answer_questions(questions, answer_chain)

if __name__ == "__main__":
    main()
