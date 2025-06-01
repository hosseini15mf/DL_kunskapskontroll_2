import streamlit as st
from functions import read_pdf, chunk_text, generate_embeddings, semantic_search, get_response
import os

st.set_page_config(page_title="RAG PDF QA", layout="centered")
st.title("📄 RAG-powered PDF QA")

# Session state variables
if 'chunks' not in st.session_state:
    st.session_state.chunks = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'pdf_text' not in st.session_state:
    st.session_state.pdf_text = None
if 'response' not in st.session_state:
    st.session_state.response = ""

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file is not None:
    with st.spinner("Reading and processing PDF..."):
        # Save to temp file
        temp_file_path = os.path.join("temp_uploaded.pdf")
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Process PDF
        st.session_state.pdf_text = read_pdf(temp_file_path)
        st.session_state.chunks = chunk_text(st.session_state.pdf_text, 100, 20)
        st.session_state.embeddings = generate_embeddings(st.session_state.chunks)
        st.success("PDF processed and embeddings cached.")

# Question input
query = st.text_input("Enter your question about the PDF")

# Ask button
if st.button("Ask"):
    if st.session_state.embeddings is None or st.session_state.chunks is None:
        st.warning("Please upload and process a PDF first.")
    elif query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer (API preferred, fallback to local)..."):
            context = semantic_search(query, st.session_state.chunks, st.session_state.embeddings)
            prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
            st.session_state.response = get_response(prompt)  # handles API + fallback

# Show result
if st.session_state.response:
    st.text_area("Response", value=st.session_state.response, height=200)
