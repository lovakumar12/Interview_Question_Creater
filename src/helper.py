import os
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("OPEN_API_KEY")

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import load_summarize_chain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from src.constant import *






## parse the pdf file and split it into chunks
file_path='Data/Srinivas-JDSA.pdf'
loader=PyPDFLoader(file_path)
data=loader.load()


splitter_ques_gen=TokenTextSplitter(chunk_size=1000, chunk_overlap=200)
chunk_question_gen=splitter_ques_gen.split_documents(data)
len(chunk_question_gen)


docuemnt_ques_gen=[Document(page_content = t.page_content) for t in chunk_question_gen]




prompt_template=""" 
you are an expert at creating question based on coding materials and documents .
your goal is to prepare a coder or programmer for their exam and coding test.
You do this by asking questions about the text below:

-----------
{text}
-----------

Create 2 questions that will prepare the coder or programmers for their tests.
Make sure not to lose any important information .

QUESTIONS:
"""



refine_template = ("""

You are an expert at creating questions based on coding material and documentation.
Your goal is to help a coder or programmer prepare for a coding test.
We have received some practice questions to a certain extent: {existing_answer}.
We have teh option to refine the existing questions or create new ones.
(only if necessary) with some more contex below.

-----------
{text}
-----------

Given the new context, refine the original questions in English.
If the context is not helpful, please provide the 1 original questions.
QUESTIONS:

"""
)


REFINE_PROMPT_QUESTIONS = PromptTemplate(
    template=refine_template,
    input_variables=["existing_answer", "text"],
)





llm_ques_gen_pipeline =ChatGroq(
    model="llama3-70b-8192",
    temperature=0.2,
    api_key=GROQ_API_KEY,
)



prompt_ques_gen = PromptTemplate(
    template=prompt_template,
    input_variables=["text"],
)


# It looks like the code snippet you provided is a Python script that processes a PDF file, splits it
# into chunks, generates questions based on the content, and then attempts to answer those questions
# using a retrieval-based question-answering model.
ques_gen_chain=load_summarize_chain(
    llm=llm_ques_gen_pipeline,
    chain_type="refine",
    verbose=True,
    question_prompt=prompt_ques_gen,
    refine_prompt=REFINE_PROMPT_QUESTIONS)




vector_store=FAISS.from_documents(
    documents=docuemnt_ques_gen,
    embedding=embeddings
)


answer_genetation_chain=RetrievalQA.from_chain_type(
    llm=llm_ques_gen_pipeline,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    
)

ques=ques_gen_chain.run(docuemnt_ques_gen)
print(ques)
ques_list=ques.split('\n')


#Answer each question and save to a file
for question in ques_list:
    if question.strip():
        print(f"Question: {question}")
        answer = answer_genetation_chain.run(question)
        print(f"Answer: {answer}\n")
        print("=====================================\\n\\n")
        
        #save answer to file
        with open("answers.txt", "a") as f:
            f.write(f"Question: {question}\n")
            f.write(f"Answer: {answer}\n")
            f.write("=====================================\n\n")