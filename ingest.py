# Bir sonraki adımda doldurulacak:
# - data/ içindeki PDF/HTML/TXT belgeleri okuma
# - Parçalara bölme (chunking)
# - Embedding üretme
# - ChromaDB'ye kaydetme
# ingest.py
# Amaç: data/ içindeki PDF/HTML/TXT dosyalarını okuyup
#       metinleri parçalayıp (chunk), embedding üretip
#       ChromaDB'ye kalıcı olarak yazmak.

import os
import glob
from dataclasses import dataclass
from typing import List
CHROMA_TELEMETRY_DISABLED=1
from dotenv import load_dotenv
load_dotenv()
import os
os.environ["CHROMA_TELEMETRY_DISABLED"] = os.getenv("CHROMA_TELEMETRY_DISABLED", "1")


# --- Basit dosya okuyucular ---
from pypdf import PdfReader
from markdownify import markdownify as md

# --- Parçalama (chunking) ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

# --- Chroma vector store ---
from langchain_community.vectorstores import Chroma

# --- Gemini embeddings için minimal wrapper ---
import google.generativeai as genai
from langchain.embeddings.base import Embeddings

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./storage")
COLLECTION_NAME = "yok_student_rights"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY yok. Lütfen .env dosyanıza GOOGLE_API_KEY ekleyin."
    )
genai.configure(api_key=GOOGLE_API_KEY)

class GeminiEmbeddings(Embeddings):
    """LangChain Embeddings arayüzünü Google Generative AI ile sağlayan basit sınıf."""
    def __init__(self, model: str = EMBEDDING_MODEL):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for t in texts:
            content = t if t and t.strip() else " "
            resp = genai.embed_content(model=self.model, content=content)
            vectors.append(resp["embedding"])
        return vectors

    def embed_query(self, text: str) -> List[float]:
        text = text if text and text.strip() else " "
        resp = genai.embed_content(model=self.model, content=text)
        return resp["embedding"]

# --- Yardımcı: dosyaları oku (pdf/html/txt) ---
@dataclass
class RawDoc:
    text: str
    meta: dict

def read_pdf(path: str) -> List[RawDoc]:
    reader = PdfReader(path)
    docs = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        docs.append(RawDoc(text=txt, meta={"source": os.path.basename(path), "page": i+1, "path": path}))
    return docs

def read_txt(path: str) -> List[RawDoc]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    return [RawDoc(text=txt, meta={"source": os.path.basename(path), "path": path})]

def read_html(path: str) -> List[RawDoc]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()
    text_md = md(html) or ""
    return [RawDoc(text=text_md, meta={"source": os.path.basename(path), "path": path})]

def load_raw_documents(data_dir: str = "data") -> List[RawDoc]:
    patterns = [
        os.path.join(data_dir, "**/*.pdf"),
        os.path.join(data_dir, "**/*.PDF"),
        os.path.join(data_dir, "**/*.txt"),
        os.path.join(data_dir, "**/*.md"),
        os.path.join(data_dir, "**/*.html"),
        os.path.join(data_dir, "**/*.htm"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))

    if not files:
        raise FileNotFoundError(
            f"'{data_dir}/' içinde işlenecek dosya bulunamadı. "
            "Lütfen PDF/HTML/TXT dosyalarını data/ klasörüne koyun."
        )

    raw_docs: List[RawDoc] = []
    for path in files:
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".pdf":
                raw_docs.extend(read_pdf(path))
            elif ext in [".txt", ".md"]:
                raw_docs.extend(read_txt(path))
            elif ext in [".html", ".htm"]:
                raw_docs.extend(read_html(path))
        except Exception as e:
            print(f"[WARN] {path} okunurken hata: {e}")

    raw_docs = [d for d in raw_docs if d.text and d.text.strip()]
    if not raw_docs:
        raise RuntimeError("Metin çıkarılamadı. Dosyalar boş olabilir veya okunamadı.")
    return raw_docs

def chunk_documents(raw_docs: List[RawDoc]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    docs: List[Document] = []
    for d in raw_docs:
        chunks = splitter.split_text(d.text)
        for idx, ch in enumerate(chunks):
            meta = dict(d.meta)
            meta["chunk"] = idx + 1
            docs.append(Document(page_content=ch, metadata=meta))
    return docs

def build_chroma(docs: List[Document]):
    os.makedirs(CHROMA_DIR, exist_ok=True)
    embeddings = GeminiEmbeddings(EMBEDDING_MODEL)
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    vectordb.persist()
    return vectordb

def main():
    print(">> Belgeler yükleniyor...")
    raw = load_raw_documents("data")
    print(f">> {len(raw)} ham parça okundu (sayfa/dosya bazlı).")

    print(">> Chunking yapılıyor...")
    docs = chunk_documents(raw)
    print(f">> {len(docs)} adet chunk üretildi.")

    print(">> Embedding ve Chroma'ya yazılıyor (bu biraz sürebilir)...")
    _ = build_chroma(docs)
    print(">> Tamam! Chroma index oluşturuldu.")
    print(f">> Koleksiyon: {COLLECTION_NAME}")
    print(f">> Klasör: {CHROMA_DIR}")

if __name__ == "__main__":
    main()
