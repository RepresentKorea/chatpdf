__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from dotenv import load_dotenv
load_dotenv()
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
#from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os

#제목
st.title("ChatPDF")
st.write("---")

#파일 업로드
uploaded_file = st.file_uploader("pdf 파일을 올려주세요!", type=['pdf'])
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

#업로드 되면 동작하는 코드
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)


    #loader
    #loader = PyPDFLoader("unsoo.pdf")
    #pages = loader.load_and_split()

    #split
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 500,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )
    texts = text_splitter.split_documents(pages)

    #embeddings
    embeddings_model = OpenAIEmbeddings()
    db = Chroma.from_documents(texts, embeddings_model)

    #question
    st.header("ChatPDF에게 질문하세요!")
    question = st.text_input('질문을 입력하세요')
    if st.button('질문하기'):
        with st.spinner("기다려봐"):
        #question = "주인공의 아내가 어떻게 됐어?"
            llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0)
        qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
        result = qa_chain({"query":question})
        st.write(result["result"])
