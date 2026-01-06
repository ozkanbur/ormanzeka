import streamlit as st
import os
import tempfile
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Sayfa Ayarları
st.set_page_config(page_title="Orman Mevzuat Asistanı", layout="wide", page_icon="🌲")

# Başlık ve Açıklama
st.title("🌲 Orman Mevzuat Asistanı (AI)")
st.markdown("""
Bu asistan, **Google Gemini** altyapısını kullanarak yüklediğiniz ormancılık mevzuatını analiz eder.
Yönetmelik, kanun veya tebliğ PDF'lerini yükleyin ve sorun.
""")

# Yan Menü (Sidebar) - Dosya Yükleme Alanı
st.sidebar.header("📁 Belge Yükle")
api_key = st.secrets["GOOGLE_API_KEY"] # API Anahtarını güvenli alandan çekeceğiz

uploaded_files = st.sidebar.file_uploader("Mevzuat PDF'lerini Buraya Sürükleyin", accept_multiple_files=True, type="pdf")

# Buton
process_button = st.sidebar.button("Belgeleri İşle ve Hazırla")

# Ana Fonksiyonlar
if process_button and uploaded_files:
    if not api_key:
        st.error("Lütfen API anahtarınızı tanımlayın!")
    else:
        with st.spinner("Belgeler taranıyor ve yapay zeka için hazırlanıyor..."):
            documents = []
            # PDF'leri geçici olarak kaydet ve oku
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name

                loader = PyPDFLoader(temp_file_path)
                docs = loader.load()
                documents.extend(docs)
                os.remove(temp_file_path) # Temizlik

            # Metinleri parçalara böl (Chunking)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)

            # Vektör Veritabanı Oluştur (Embeddings)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
            vector_store = FAISS.from_documents(splits, embeddings)
            
            # Veritabanını oturuma kaydet (Session State)
            st.session_state.vector_store = vector_store
            st.success(f"Tamamlandı! Toplam {len(splits)} parçaya bölündü. Artık soru sorabilirsiniz.")

# Soru Sorma Alanı
soru = st.text_input("Mevzuat ile ilgili sorunuz nedir?", placeholder="Örn: 6831 sayılı kanuna göre işgal ve faydalanma suçu nedir?")

if soru:
    if "vector_store" not in st.session_state:
        st.warning("Lütfen önce sol menüden PDF yükleyin ve 'İşle' butonuna basın.")
    else:
        # Model Ayarları (Gemini 1.5 Flash)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.3)
        
        # Özel Prompt (Yapay Zekaya Rol Verme)
        prompt_template = """
        Sen uzman bir Orman Mühendisi asistanısın. Aşağıdaki bağlamı (context) kullanarak kullanıcının sorusunu cevapla.
        Cevap verirken ilgili kanun maddesine veya yönetmelik bölümüne atıf yapmaya çalış.
        Eğer bilgi metinlerde yoksa "Bu bilgi yüklenen belgelerde bulunamadı" de.
        
        Bağlam: {context}
        Soru: {question}
        
        Cevap:
        """
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        # Zinciri Kurma
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.vector_store.as_retriever(),
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        with st.spinner("Mevzuat taranıyor..."):
            cevap = qa_chain.run(soru)
            st.write("### 🤖 Asistanın Cevabı:")
            st.write(cevap)