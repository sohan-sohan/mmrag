import streamlit as st
from pathlib import Path
import hashlib
from backend import (
    initialise_db,
    populate_db,
    process_query,
    generate_answer,
    reset_storage,
)

st.set_page_config(page_title="Multimodal RAG App", layout="wide")
st.title("📄🖼️ Multimodal RAG App")

# ------------------ SESSION STATE ------------------
if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

if "indexed" not in st.session_state:
    st.session_state.indexed = False


# ------------------ UPLOAD ------------------
st.header("Upload PDF")
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    file_bytes = uploaded_file.getbuffer()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    if st.session_state.file_hash != file_hash:
        reset_storage()
        st.session_state.file_hash = file_hash
        st.session_state.indexed = False

    save_path = Path("uploads") / "base_pdf.pdf"
    save_path.parent.mkdir(exist_ok=True)

    with open(save_path, "wb") as f:
        f.write(file_bytes)

    st.success("PDF uploaded")

    # ------------------ INDEXING ------------------
    if not st.session_state.indexed:
        with st.spinner("Indexing document..."):
            client = initialise_db()
            populate_db(save_path.as_posix(), client)
            st.session_state.indexed = True
        st.success("Indexing complete")

    # ------------------ QUERY ------------------
    st.divider()
    query = st.text_input("Ask a question")

    if query:
        retrieved = process_query(query)

        # ------------------ LLM ANSWER ------------------
        st.subheader("🤖 LLM Answer")
        with st.spinner("Thinking..."):
            answer = generate_answer(query, retrieved)
            st.write(answer)

        # ------------------ TEXT ------------------
        st.divider()
        st.subheader("📄 Retrieved Text")

        for item in retrieved:
            if item.payload["type"] == "text":
                st.markdown(
                    f"""
                    **Page:** {item.payload['page_no']}  
                    {item.payload['text'][:800]}
                    """
                )

        # ------------------ IMAGES ------------------
        st.divider()
        st.subheader("🖼️ Retrieved Images")

        cols = st.columns(3)
        idx = 0

        for item in retrieved:
            if item.payload["type"] == "image":
                with cols[idx]:
                    st.caption(f"Page {item.payload['page_no']}")
                    st.image(
                        f"images/{item.payload['filename']}",
                        width=220,
                    )
                idx = (idx + 1) % 3
