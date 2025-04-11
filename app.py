# from fastapi import FastAPI ,Form , Request , Response ,Depends ,HTTPException ,status,File, UploadFile
# from fastapi.responses import RedirectResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from fastapi.encoders import jsonable_encoder

#import uvicorn
import os
from dotenv import load_dotenv
from flask import Flask, render_template, request
#import aiofiles

from src.constant import prompt_template, REFINE_PROMPT_QUESTIONS
from src.llm_setup import setup_llm, get_ques_gen_chain
from src.pdf_loaders import load_pdf_and_chunk
from src.qna import generate_questions, answer_questions
from src.vector_store import create_vector_store , get_answer_chain



#groq authnetication
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")




#




app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    if request.method == "POST":
        uploaded_file = request.files["pdf"]
        if uploaded_file.filename != "":
            pdf_path = os.path.join("uploads", uploaded_file.filename)
            uploaded_file.save(pdf_path)

            documents = load_pdf_and_chunk(pdf_path)
            llm = setup_llm()
            ques_chain = get_ques_gen_chain(llm)

            questions = generate_questions(ques_chain, documents)
            vector_store = create_vector_store(documents)
            answer_chain = get_answer_chain(llm, vector_store)

            results = answer_questions(answer_chain, questions)

            with open("answers.txt", "a") as f:
                for q, a in results:
                    f.write(f"Question: {q}\nAnswer: {a}\n{'='*40}\n")

    return render_template("index.html", results=results)
