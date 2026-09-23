import streamlit as st

from config import DOCS_DIR, EMBED_MODEL, LLM_MODEL
from rag.rag_chain import answer_question
from rag.vectorstore import get_store

st.set_page_config(page_title="Assistant Supply Chain", page_icon="📦", layout="wide")

with st.sidebar:
    st.header("📦 Assistant Supply Chain")
    st.caption("RAG local — Ollama + index vectoriel numpy")

    try:
        doc_count = get_store().count()
    except Exception:
        doc_count = None

    if doc_count:
        st.metric("Chunks indexés", doc_count)
    else:
        st.warning("Aucun index trouvé. Lance `python ingest.py` avant de poser des questions.")

    st.divider()
    st.markdown(f"**LLM :** `{LLM_MODEL}`")
    st.markdown(f"**Embeddings :** `{EMBED_MODEL}`")
    st.markdown(f"**Documents source :** `{DOCS_DIR.name}/`")

    st.divider()
    if st.button("Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()

st.title("Assistant supply chain (RAG)")
st.caption(
    "Pose une question sur les Incoterms, le transport, la gestion des stocks, "
    "les retours ou les KPI logistiques. Les réponses citent leurs sources."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources utilisées"):
                for src in message["sources"]:
                    st.markdown(f"**{src['source']}** — similarité {src['similarity']:.2f}")
                    st.caption(src["text"][:400] + ("…" if len(src["text"]) > 400 else ""))

question = st.chat_input("Ta question sur la supply chain…")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche dans les documents et génération de la réponse…"):
            result = answer_question(question)
        st.markdown(result.answer)
        if result.sources:
            with st.expander("Sources utilisées"):
                for src in result.sources:
                    st.markdown(f"**{src['source']}** — similarité {src['similarity']:.2f}")
                    st.caption(src["text"][:400] + ("…" if len(src["text"]) > 400 else ""))

    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer, "sources": result.sources}
    )
