# app_gradio.py — HF Spaces için sağlam arayüz (bootstrap + RAG)
import os
from dotenv import load_dotenv

load_dotenv()
CHROMA_DIR = os.getenv("CHROMA_DIR", "./storage")

def _needs_bootstrap():
    has_storage = os.path.exists(CHROMA_DIR) and any(os.scandir(CHROMA_DIR))
    has_data = os.path.exists("data") and any(os.scandir("data"))
    return (not has_storage) and has_data

if _needs_bootstrap():
    try:
        import ingest
        ingest.main()
    except Exception as e:
        print("[BOOTSTRAP WARN] Ingestion at startup failed:", e)

from rag_pipeline import ask_question
import gradio as gr

def chat_fn(message, history):
    answer, sources = ask_question(message, k=4)
    src_lines = []
    for s in sources or []:
        tail = []
        if s.get("page"): tail.append(f"sayfa {s['page']}")
        if s.get("chunk"): tail.append(f"chunk {s['chunk']}")
        tail_str = f" ({', '.join(tail)})" if tail else ""
        src_lines.append(f"- {s.get('source','unknown')}{tail_str}")
    src_text = "\n".join(src_lines) if src_lines else "- (kaynak bulunamadı)"
    return f"{answer}\n\n---\n📎 Kaynaklar:\n{src_text}"

demo = gr.ChatInterface(
    fn=chat_fn,
    title="🎓 YÖK Öğrenci Hakları RAG",
    description="YÖK ve üniversite yönetmeliklerine dayalı kaynaklı yanıtlar."
)

if __name__ == "__main__":
    demo.launch()
