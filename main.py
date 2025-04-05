from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

os.environ["OPENAI_API_KEY"] = "your-openai-key"

def analyze_rfp(pdf_path):
    # 1. Load the PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()

    # 2. Create a vector database
    db = FAISS.from_documents(pages, OpenAIEmbeddings())

    # 3. Build a Q&A system
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(),
        retriever=db.as_retriever(),
        return_source_documents=False
    )

    # 4. Ask your questions
    eligibility = qa.run("Are we eligible for this contract?")
    checklist = qa.run("Extract all the formatting and submission requirements.")
    risks = qa.run("Are there any risky or biased legal clauses?")

    return {
        "eligibility": eligibility,
        "checklist": checklist,
        "risks": risks
    }
