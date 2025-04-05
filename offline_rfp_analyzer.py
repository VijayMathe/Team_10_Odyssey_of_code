# offline_rfp_analyzer.py

import streamlit as st
import pdfplumber
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import tempfile
import os
import warnings
from groq import Groq

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="CropBox missing from /Page")

# Groq API setup
client = Groq(api_key="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")

# Setup Streamlit UI
st.set_page_config(page_title="Offline RFP Analyzer", layout="wide")
st.title("📄 Offline RFP Analyzer (Groq LLaMA 3 + FAISS)")

@st.cache_resource
def load_embedder():
    try:
        return SentenceTransformer('BAAI/bge-small-en')
    except Exception as e:
        st.error(f"Error loading embedder: {e}")
        return None

embedder = load_embedder()
if embedder is None:
    st.stop()

# Functions
def extract_text_from_pdf(file_path):
    try:
        full_text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    full_text += txt + "\n"
        return full_text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def chunk_text(text, max_length=500):
    try:
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) < max_length:
                current_chunk += para + "\n"
            else:
                chunks.append(current_chunk)
                current_chunk = para + "\n"
        if current_chunk:
            chunks.append(current_chunk)
        return chunks
    except Exception as e:
        st.error(f"Error chunking text: {e}")
        return []

def build_vector_store(chunks):
    try:
        embeddings = embedder.encode(chunks)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings))
        return index, chunks, embeddings
    except Exception as e:
        st.error(f"Error building vector store: {e}")
        return None, [], []

def get_top_k(query, index, chunks, embeddings, k=4):
    try:
        query_embedding = embedder.encode([query])
        D, I = index.search(np.array(query_embedding), k)
        return [chunks[i] for i in I[0]]
    except Exception as e:
        st.error(f"Error retrieving top-k chunks: {e}")
        return []

def generate_prompt(user_query, top_chunks):
    context = "\n\n".join(top_chunks)
    return f"""
You are a government procurement analyst.
Given the following context, answer the user's question accurately.

Context:
{context}

Question: {user_query}
Answer:
"""

def get_llm_answer(prompt):
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
            stream=False,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error getting LLM answer: {e}")
        return "Unable to generate response due to an error."

# UI
st.sidebar.markdown("Made with 🧠 Groq LLaMA 3 & FAISS")
uploaded_file = st.file_uploader("Upload a Government RFP PDF", type="pdf")
company_file = st.file_uploader("Upload Company Data PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    st.success("PDF uploaded and parsed!")

    full_text = extract_text_from_pdf(file_path)
    if not full_text:
        st.stop()

    chunks = chunk_text(full_text)
    if not chunks:
        st.stop()

    index, all_chunks, embeddings = build_vector_store(chunks)
    if index is None:
        st.stop()

    st.subheader("🔍 Ask about this RFP")
    user_query = st.text_input("Enter your question (e.g., Are we eligible? What is the submission deadline?)")

    if st.button("💡 Get Answer") and user_query:
        top_chunks = get_top_k(user_query, index, all_chunks, embeddings)
        if top_chunks:
            prompt = generate_prompt(user_query, top_chunks)
            with st.spinner("Thinking like a contract analyst..."):
                result = get_llm_answer(prompt)
            st.write(result)
        else:
            st.warning("No relevant chunks found to answer the query.")

    if st.button("🛡 Check Eligibility") and company_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(company_file.read())
            company_path = tmp.name

        company_text = extract_text_from_pdf(company_path)
        if not company_text:
            st.error("Failed to extract company data.")
        else:
            eligibility_prompt = f"""
Compare the following RFP and company data.
Determine if the company is eligible to apply for the contract based on registration, certifications, and past performance.
Explain clearly.

RFP:
{full_text[:3000]}

Company Info:
{company_text[:2000]}
"""
            with st.spinner("Analyzing eligibility..."):
                eligibility_result = get_llm_answer(eligibility_prompt)
            st.subheader("✅ Eligibility Result")
            st.write(eligibility_result)
        os.remove(company_path)

    if st.button("📝 Generate Submission Checklist"):
        checklist_prompt = f"""
From the following RFP content, extract all submission requirements into a checklist.
Include formatting rules, page limits, attachments, deadlines, and forms.

RFP:
{full_text[:3000]}
"""
        with st.spinner("Extracting checklist..."):
            checklist = get_llm_answer(checklist_prompt)
        st.subheader("📋 Submission Checklist")
        st.write(checklist)

    os.remove(file_path)