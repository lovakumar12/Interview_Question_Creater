from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_core.documents import Document

def load_pdf_and_chunk(file_path):
    loader = PyPDFLoader(file_path)
    data = loader.load()

    splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(data)

    return [Document(page_content=t.page_content) for t in chunks]
