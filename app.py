# app.py — RAG'e bağlanmış sürüm
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()
HAS_KEY = bool(os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="YÖK Öğrenci Hakları RAG", page_icon="🎓", layout="centered")

st.title("🎓 YÖK Öğrenci Hakları Asistanı (RAG)")
st.caption(f"🔐 Gemini API key loaded: {'YES' if HAS_KEY else 'NO'}")

st.write("""
Bu uygulama, YÖK ve üniversite yönetmelikleri temelinde sorularınıza **RAG** mimarisi ile
yanıt üretir. Aşağıya sorunuzu yazın ve **Sor** butonuna basın.
""")

from rag_pipeline import ask_question

user_q = st.text_input("Bir soru yazın (örn. 'Mazeret sınavı şartları nelerdir?')")

if st.button("Sor"):
    if not user_q.strip():
        st.warning("Lütfen bir soru yazın.")
    elif not HAS_KEY:
        st.error("Gemini API anahtarı bulunamadı. Lütfen .env dosyasına GOOGLE_API_KEY ekleyin.")
    else:
        with st.spinner("Yanıt üretiliyor..."):
            answer, sources = ask_question(user_q, k=4)

        st.markdown("### 🧠 Yanıt")
        st.write(answer)

        if sources:
            st.markdown("### 📎 Kaynaklar")
            for i, s in enumerate(sources, start=1):
                src = s.get("source", "unknown")
                page = s.get("page")
                chunk = s.get("chunk")
                tail = []
                if page: tail.append(f"sayfa {page}")
                if chunk: tail.append(f"chunk {chunk}")
                tail_str = f" ({', '.join(tail)})" if tail else ""
                st.write(f"- Kaynak {i}: **{src}**{tail_str}")

with st.sidebar:
    st.subheader("Durum")
    st.write(f"API Key: {'OK' if HAS_KEY else 'Eksik'}")
    st.write("Vector DB: Chroma (persist)")
    st.write("LLM: Gemini 1.5 Flash")

