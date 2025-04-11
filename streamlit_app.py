# streamlit_app.py
import streamlit as st
import os
from dotenv import load_dotenv



import streamlit as st
import tempfile

from src.pdf_loaders import load_pdf_and_chunk
from src.llm_setup import setup_llm, get_ques_gen_chain
from src.constant import prompt_template, REFINE_PROMPT_QUESTIONS
from src.vector_store import create_vector_store
from src.qna import  generate_question_list_from_pdf ,generate_answers_from_pdf



st.title("PDF Q&A Generator 🤖")

llm = setup_llm()

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    st.write("Generating questions and answers...")
    results = generate_answers_from_pdf(tmp_file_path, llm, return_results=True)

    for q, a in results:
        st.markdown(f"**Question:** {q}")
        st.markdown(f"**Answer:** {a}")
        st.markdown("---")














