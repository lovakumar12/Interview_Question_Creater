# Interview-question-creater

# 🤖 Interview Question Creator

Welcome to the **Interview Question Creator**, a powerful Streamlit app that analyzes a PDF (like a resume or document) and generates relevant interview questions along with potential answers using advanced NLP and LLMs.

---

## 🚀 Live Demo

🌐 [Click here to try the app](https://interviewquestioncreater.streamlit.app/)

---

## 📌 Features

- 📄 Upload PDF files (e.g., resumes)
- 🧠 Extracts content using LLMs (like OpenAI/GPT or HuggingFace models)
- ❓ Generates relevant interview questions
- ✅ Answers the questions based on content
- 🔍 Uses FAISS for efficient semantic search and retrieval
- 📊 Clean and interactive Streamlit interface

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **LangChain**
- **FAISS**
- **HuggingFace / OpenAI LLMs**
- **PyPDF** for PDF parsing

---

## 📂 Project Structure


Interview-question-creater/ │ ├── streamlit_app.py # Streamlit UI ├── requirements.txt # Python dependencies ├── README.md # You're here │ └── src/ ├── qna.py # Core Q&A logic ├── loader.py # PDF loader and document processor ├── constant.py # Constants and model setup └── utils.py # Utilities














### How to run?
1. Create an environment

```bash
conda create -n interview python=3.10 -y

conda activate interview

```bash

2.install requirements.txt

```
pip install -r requirements.txt
```
