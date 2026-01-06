import streamlit as st
import os
import tempfile
import asyncio

# --- ASYNCIO DÖNGÜSÜ YAMASI ---
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
# ------------------------------

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Sayfa Ayarları
st.set_page_config(page_title="Orman Mevzuat Asistanı", layout="wide", page_icon="🌲")

st.title("🌲 Orman Mevzuat Asistanı (AI)")
st.markdown("Yönetmelik PDF'lerini yükleyin ve sorun.")

# Yan Menü
st.sidebar.header("📁 Belge Yükle")

# API Key Kontrolü
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("API Anahtarı bulunamadı.")
    api_key = None

uploaded_files = st.sidebar.file_uploader("PDF Yükle", accept_multiple_files=True, type="pdf")
process_button = st.sidebar.button("Belgeleri İşle")

if process_button and uploaded_files:
    if not api_key:
        st.error("API Anahtarı yok!")
    else:
        with st.spinner("Belgeler işleniyor... (İlk seferde model indirildiği için 1-2 dk sürebilir)"):
            documents = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name

                loader = PyPDFLoader(temp_file_path)
                docs = loader.load()
                documents.extend(docs)
                os.remove(temp_file_path)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)

            # HuggingFace Embedding (Yerel ve Ücretsiz)
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            vector_store = FAISS.from_documents(splits, embeddings)
            st.session_state.vector_store = vector_store
            st.success(f"Tamamlandı! {len(splits)} parçaya bölündü.")

soru = st.text_input("Sorunuzu yazın:")

if soru:
    if "vector_store" not in st.session_state:
        st.warning("Önce belge yükleyin.")
    else:
        if api_key:
            # GÜNCELLENEN KISIM: Model ismi değişti
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash-latest", 
                google_api_key=api_key, 
                temperature=0.3
            )
            
            prompt_template = """
            Sen uzman bir Orman Mühendisi asistanısın.
            Bağlam: {context}
            Soru: {question}
            Cevap:
            """
            PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=st.session_state.vector_store.as_retriever(),
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            with st.spinner("Cevap hazırlanıyor..."):
                try:
                    cevap = qa_chain.run(soru)
                    st.write(cevap)
                except Exception as e:
                    st.error(f"Hata: {e}")
