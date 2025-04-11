
# from langchain.chains import RetrievalQA
# from langchain.chains import load_summarize_chain

# from src.constant import embeddings
# from src.pdf_loaders import load_pdf_and_chunk
# from src.llm_setup import *
# from src.vector_store import create_vector_store

# llm=setup_llm()





# def generate_question_list_from_pdf(pdf_path: str, llm):
#     """
#     Loads a PDF, generates questions using a refine chain, and returns a list of questions.
    
#     Parameters:
#     - pdf_path (str): Path to the PDF file.
#     - llm: The language model instance.

#     Returns:
#     - List[str]: A list of generated questions.
#     """
#     documents = load_pdf_and_chunk(pdf_path)

#     ques_gen_chain = load_summarize_chain(
#         llm=llm,
#         chain_type="refine",
#         verbose=True,
#         question_prompt=prompt_ques_gen,
#         refine_prompt=REFINE_PROMPT_QUESTIONS
#     )

#     questions_text = ques_gen_chain.run(documents)
#     ques_list = [q.strip() for q in questions_text.split('\n') if q.strip()]
    
#     return ques_list

# def generate_answers_from_pdf(pdf_path: str, llm, return_results: bool = False):
#     """
#     Generate and optionally return answers for questions extracted from a PDF.

#     Args:
#         pdf_path (str): Path to the PDF file.
#         llm: Language model instance to use for question generation and answering.
#         return_results (bool): If True, returns a list of (question, answer) tuples. Otherwise, prints them.

#     Returns:
#         Optional[List[Tuple[str, str]]]: List of (question, answer) pairs if return_results is True.
#     """
    
#     # Create the RetrievalQA answer generation chain
#     answer_generation_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=create_vector_store.as_retriever()
#     )

#     # List to store results if needed
#     results = []

#     # Generate and process questions
#     for question in generate_question_list_from_pdf(pdf_path, llm):
#         question = question.strip()
#         if question:
#             answer = answer_generation_chain.run(question)
#             if return_results:
#                 results.append((question, answer))
#             else:
#                 print(f"Question: {question}")
#                 print(f"Answer: {answer}\n")

#     if return_results:
#         return results

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain  # use this instead of load_summarize_chain

from langchain_huggingface import HuggingFaceEmbeddings  # updated import

from src.constant import embeddings
from src.pdf_loaders import load_pdf_and_chunk
from src.llm_setup import *
from src.vector_store import create_vector_store

llm = setup_llm()


def generate_question_list_from_pdf(pdf_path: str, llm):
    """
    Loads a PDF, generates questions using a QA chain, and returns a list of questions.
    (Simulating summarization-based question gen for now.)

    Parameters:
        pdf_path (str): Path to the PDF file.
        llm: The language model instance.

    Returns:
        List[str]: A list of generated questions.
    """

    documents = load_pdf_and_chunk(pdf_path)

    # You may replace this prompt with a more detailed one
    prompt = PromptTemplate(
        input_variables=["context"],
        template="""
        Based on the following context, generate 5 thoughtful and diverse questions:
        {context}
        """
    )

    chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)
    output = chain.run(input_documents=documents)

    # Split into lines assuming each question is in a new line
    ques_list = [q.strip() for q in output.split("\n") if q.strip()]
    return ques_list


def generate_answers_from_pdf(pdf_path: str, llm, return_results: bool = False):
    """
    Generate and optionally return answers for questions extracted from a PDF.

    Args:
        pdf_path (str): Path to the PDF file.
        llm: Language model instance to use for question generation and answering.
        return_results (bool): If True, returns a list of (question, answer) tuples. Otherwise, prints them.

    Returns:
        Optional[List[Tuple[str, str]]]: List of (question, answer) pairs if return_results is True.
    """
    documents = load_pdf_and_chunk(pdf_path)
    vector_store = create_vector_store(documents)
    retriever = vector_store.as_retriever()

    answer_generation_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    results = []

    for question in generate_question_list_from_pdf(pdf_path, llm):
        question = question.strip()
        if question:
            answer = answer_generation_chain.run(question)
            if return_results:
                results.append((question, answer))
            else:
                print(f"Question: {question}")
                print(f"Answer: {answer}\n")

    if return_results:
        return results



