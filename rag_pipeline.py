# rag_pipeline.py
# Amaç: Chroma'dan bağlam çek, Gemini ile kaynaklı yanıt üret.

import os
from typing import List, Tuple
from dotenv import load_dotenv

load_dotenv()

# Chroma + LangChain
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.embeddings.base import Embeddings

# Gemini
import google.generativeai as genai

# --- Ortam değişkenleri ---
CHROMA_DIR = os.getenv("CHROMA_DIR", "./storage")
COLLECTION_NAME = "yok_student_rights"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "models/gemini-2.5-flash")



if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY eksik. .env dosyasında tanımlayın.")
genai.configure(api_key=GOOGLE_API_KEY)

# --- Embeddings (Gemini) ---
class GeminiEmbeddings(Embeddings):
    def __init__(self, model: str = EMBEDDING_MODEL):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out = []
        for t in texts:
            content = t if t and t.strip() else " "
            resp = genai.embed_content(model=self.model, content=content)
            out.append(resp["embedding"])
        return out

    def embed_query(self, text: str) -> List[float]:
        text = text if text and text.strip() else " "
        resp = genai.embed_content(model=self.model, content=text)
        return resp["embedding"]

# --- Vector store okuyucu ---
def _load_vectordb() -> Chroma:
    embeddings = GeminiEmbeddings(EMBEDDING_MODEL)
    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    return vectordb

def retrieve_context(query: str, k: int = 4) -> List[Document]:
    vectordb = _load_vectordb()
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    docs: List[Document] = retriever.get_relevant_documents(query)
    return docs

# --- Prompt şablonu ---
SYSTEM_INSTRUCTIONS = """
You are a Turkish Q&A assistant specialized in YÖK and university regulations.
- Answer in **Turkish**.
- Use ONLY the provided context. Do not fabricate.
- If the answer is not in the context, say "Buna dair net bir hüküm bulamadım." and suggest where to check.
- Provide a short, clear answer first. Then list the relevant articles as bullet points.
- At the end, add a brief "Kaynaklar" section with document names and page/chunk references.
"""

def build_prompt(query: str, context_docs: List[Document]) -> str:
    ctx_blocks = []
    for i, d in enumerate(context_docs, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        chunk = d.metadata.get("chunk")
        tag = f"{src} | " + (f"sayfa {page}, " if page else "") + (f"chunk {chunk}" if chunk else "")
        ctx_blocks.append(f"[Kaynak {i}: {tag}]\n{d.page_content.strip()}\n")

    context_text = "\n\n".join(ctx_blocks) if ctx_blocks else "Kontekst bulunamadı."
    user_block = f"Soru: {query}\n"
    prompt = f"{SYSTEM_INSTRUCTIONS}\n\n{user_block}\nBağlam:\n{context_text}\n\nYanıt:"
    return prompt

def ask_question(query: str, k: int = 4) -> Tuple[str, List[dict]]:
    docs = retrieve_context(query, k=k)

    # Kaynak meta bilgisini topla
    sources = []
    for d in docs:
        sources.append({
            "source": d.metadata.get("source", "unknown"),
            "page": d.metadata.get("page"),
            "chunk": d.metadata.get("chunk"),
        })

    # Bağlam yoksa kısa yanıt ver
    if not docs:
        return ("Buna dair net bir hüküm bulamadım. Lütfen ilgili yönetmeliğin güncel sürümünü kontrol edin.", sources)

    prompt = build_prompt(query, docs)
    model = genai.GenerativeModel(LLM_MODEL)
    resp = model.generate_content(prompt)
    answer = resp.text if hasattr(resp, "text") else str(resp)

    return (answer.strip(), sources)
