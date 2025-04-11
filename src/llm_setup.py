from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq


from src.constant import  prompt_ques_gen, REFINE_PROMPT_QUESTIONS

import os
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")    


def setup_llm():
    return ChatGroq(
        model="llama3-70b-8192",
        temperature=0.2,
        api_key=GROQ_API_KEY,
    )

def get_ques_gen_chain(llm):
   
    return load_summarize_chain(
        llm=llm,
        chain_type="refine",
        verbose=True,
        question_prompt=prompt_ques_gen,
        refine_prompt=REFINE_PROMPT_QUESTIONS
    )


