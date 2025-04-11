
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings

prompt_template=""" 
you are an expert at creating question based on coding materials and documents .
your goal is to prepare a coder or programmer for their exam and coding test.
You do this by asking questions about the text below:

-----------
{text}
-----------

Create  questions that will prepare the coder or programmers for their tests.
Make sure not to lose any important information .

QUESTIONS:
"""

prompt_ques_gen = PromptTemplate(
    template=prompt_template,
    input_variables=["text"],
)


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
If the context is not helpful, please provide the  original questions.
QUESTIONS:

"""
)


REFINE_PROMPT_QUESTIONS = PromptTemplate(
    template=refine_template,
    input_variables=["existing_answer", "text"],
)


model="BAAI/bge-large-en"
embeddings = HuggingFaceEmbeddings(model_name=model)
