# YÖK Öğrenci Hakları RAG Chatbot

## 🎯 Amaç
YÖK ve üniversite yönetmeliklerine dayanarak öğrencilerin haklarına dair sorulara **RAG** (Retrieval-Augmented Generation) ile doğru ve kaynaklı yanıtlar üretmek.

## 🧰 Kullanılan Yöntem ve Teknolojiler
- **LLM**: Google Gemini (`${LLM_MODEL}`, varsayılan: `models/gemini-2.5-flash`)
- **Embeddings**: `models/text-embedding-004`
- **Vektör Veritabanı**: Chroma (persist)
- **RAG Pipeline**: LangChain (retriever + prompt), custom glue code
- **UI**: Streamlit
- **Dil**: Python 3.10+

## 📚 Veri Seti
- Kaynak: Kamuya açık **YÖK / üniversite yönetmelikleri** (PDF/HTML/TXT).
- Bu repoda veri bulunmaz; `data/` klasörüne **lokalde** eklenir.
- Ingestion aşamasında belgeler parçalara (chunk) bölünür ve Chroma’ya yazılır.

## 🏗️ Mimari (RAG Akışı)
1. **Ingestion**: `data/` → metin çıkarımı (PDF/HTML/TXT) → chunking (1000/150) → embeddings → **Chroma**.
2. **Sorgu**: Kullanıcı sorusu → query embedding → **Top-k** benzer chunk geri çağırma.
3. **Cevap**: Chunk içerikleri + talimatlar → **Gemini** → yanıt + **kaynak listesi**.


## 🚀 Çalıştırma Kılavuzu
```bash
python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows
# venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env
# .env içine:
# GOOGLE_API_KEY=... (AI Studio)
# EMBEDDING_MODEL=models/text-embedding-004
# LLM_MODEL=models/gemini-2.5-flash
# CHROMA_DIR=./storage

# 1) Ingestion (ilk kurulum veya veri güncellendiğinde)
python ingest.py

# 2) Uygulama
python -m streamlit run app.py



