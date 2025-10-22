# app.py (en ÜSTE ekle, streamlit'ten ÖNCE)
import os

# Streamlit'in yazacağı yerleri kullanıcı home'a yönlendir
os.environ["HOME"] = "/home/user"
os.environ["XDG_CONFIG_HOME"] = "/home/user"
os.environ["STREAMLIT_BROWSER_GATHERUSAGESTATS"] = "false"

# Home içinde .streamlit klasörünü ve config dosyasını garanti et
os.makedirs("/home/user/.streamlit", exist_ok=True)
CONFIG_PATH = "/home/user/.streamlit/config.toml"
if not os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(
            "[server]\n"
            "headless = true\n"
            "enableCORS = false\n"
            "enableXsrfProtection = false\n"
            "port = 7860\n\n"
            "[browser]\n"
            "gatherUsageStats = false\n"
        )

from dotenv import load_dotenv
load_dotenv()

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

