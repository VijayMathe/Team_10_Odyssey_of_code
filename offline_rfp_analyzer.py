from pymongo import MongoClient
import streamlit as st
import pdfplumber
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import tempfile
import os
import warnings
from groq import Groq
import time
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="CropBox missing from /Page")

# MongoDB setup
MONGO_URI = "mongodb+srv://madrohtech:hvafUOPuownEHhcj@cluster0.q8qm3qd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["rfp_analyzer_db"]
collection = db["history"]

# Groq API setup
client = Groq(api_key="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")

# Setup Streamlit UI
st.set_page_config(page_title="RFP Analyzer Pro", page_icon="📊", layout="wide")

# CSS Styles
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .subheader {
        font-size: 1.4rem;
        font-weight: 600;
        color: #3B82F6;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #DBEAFE;
        padding-bottom: 0.3rem;
    }
    .section {
        background-color: #F8FAFC;
        border-radius: 0.5rem;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .status-card {
        background-color: #EFF6FF;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #BFDBFE;
    }
    .sidebar-header {
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #1E3A8A;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #6B7280;
        font-size: 0.8rem;
    }
    .summary-box {
        background-color: #F0FDF4;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-top: 1rem;
        border: 1px solid #DCFCE7;
    }
    .warning-box {
        background-color: #FFFBEB;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-top: 1rem;
        border: 1px solid #FEF3C7;
    }
    hr {
        margin: 1.5rem 0;
        border: 0;
        height: 1px;
        background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(59, 130, 246, 0.5), rgba(0, 0, 0, 0));
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">📊 RFP Analyzer Pro</div>', unsafe_allow_html=True)
st.markdown('Generate comprehensive analysis reports from RFP documents', unsafe_allow_html=True)

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

# Define risk recommendations for risk analysis
risk_recommendations = {
    "termination": "Review termination clauses carefully to avoid sudden contract ending.",
    "liability": "Ensure liability clauses are mutually fair and limited.",
    "insurance": "Verify insurance requirements are clearly defined and feasible.",
    "risk": "Assess overall risk allocation and responsibilities.",
    "penalty": "Understand the conditions and amount of penalties.",
    "damages": "Clarify what constitutes damages and acceptable limits.",
    "unilateral": "Watch for unilateral decision-making powers that could be risky."
}

# ----------------------- Functions -----------------------

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

def get_key_info(full_text):
    prompt = f"""
Extract the following key information from this RFP document. 
Provide concise, specific answers for each category:

1. RFP Number/ID
2. Issuing Agency/Department
3. Project Title
4. Submission Deadline (with timezone if specified)
5. Contract Duration/Period of Performance
6. Estimated Contract Value (if available)

RFP text:
{full_text[:4000]}
"""
    return get_llm_answer(prompt)

def get_summary(full_text):
    prompt = f"""
Create a concise executive summary (200-300 words) of this RFP that covers:
1. The overall purpose of the procurement
2. Key deliverables requested
3. Primary evaluation criteria
4. Any unique aspects or special requirements

RFP text:
{full_text[:5000]}
"""
    return get_llm_answer(prompt)

def get_proposal_draft(full_text, company_profile):
    prompt = f"""
You are a proposal writer for a company responding to a government RFP. 
Using the RFP content and the company profile below, draft a professional proposal that includes:

1. Executive Summary
2. Company Overview
3. Understanding of the Requirements
4. Proposed Approach and Methodology
5. Key Personnel and Roles
6. Past Performance and Experience
7. Compliance with Requirements
8. Pricing Summary (if pricing info is available or to be suggested)
9. Conclusion and Call to Action

Make sure the tone is formal, persuasive, and tailored to the government agency's goals.
Ensure the proposal is customized and not generic.

RFP Content:
{full_text[:5000]}

Company Profile:
{company_profile[:2500]}
"""
    return get_llm_answer(prompt)

def get_key_dates(full_text):
    prompt = f"""
Extract all important dates and deadlines from this RFP including:
- RFP release date
- Question/inquiry submission deadline
- Pre-proposal conference date (if any)
- Proposal submission deadline
- Anticipated award date
- Project start and end dates

Present the results in chronological order with specific dates.
If any date is not found, note it as "Not specified".

RFP text:
{full_text[:5000]}
"""
    return get_llm_answer(prompt)

def get_submission_requirements(full_text):
    prompt = f"""
Create a comprehensive submission checklist from this RFP:
1. Required proposal sections with page limits
2. Required forms and attachments
3. Format requirements (margins, font, etc.)
4. Submission method (electronic, hard copy, both)
5. Number of copies required (if physical)
6. Any special submission instructions

Format this as a clear checklist.

RFP text:
{full_text[:5000]}
"""
    return get_llm_answer(prompt)

def get_evaluation_criteria(full_text):
    prompt = f"""
Extract the proposal evaluation criteria from this RFP:
1. List each criterion
2. Include relative weights or importance if specified
3. Note any pass/fail requirements
4. Include any scoring methodology described

Present this as a prioritized list from most to least important criteria.

RFP text:
{full_text[:5000]}
"""
    return get_llm_answer(prompt)

# ------------------- New / Updated Functions -------------------

def get_eligibility(full_text, company_text):
    """Default eligibility analysis based on RFP and company data"""
    prompt = f"""
Compare the following RFP and company data.
Determine if the company is eligible to apply for the contract based on the eligibility criteria found in the RFP.
The criteria may include:
- Registration requirements
- Required certifications or licenses
- Past performance or experience
- Size standards (e.g., small business status)
- Geographic restrictions
- Financial stability or revenue thresholds
- Security clearances
- Technical or operational capabilities
- Any other requirements found in the RFP

Format your response with:
- Overall Eligibility Assessment (Eligible/Potentially Eligible/Not Eligible)
- Matching Requirements
- Potential Issues or Gaps
- Recommended Next Steps

RFP Text:
{full_text[:4000]}

Company Information:
{company_text[:2500]}
"""
    return get_llm_answer(prompt)

def get_concise_eligibility(full_text, company_text):
    """Concise eligibility analysis with a strict format that starts with a clear eligibility result"""
    prompt = f"""
You are a government procurement analyst. Based on the following RFP and company data, respond in this strict format:

Eligibility: ✅ Eligible (or ❌ Not Eligible)

Checklist Highlights:
- Mention only 2–3 key reasons supporting eligibility or ineligibility.
- Keep the explanation concise and action-oriented.

RFP:
{full_text[:3000]}

Company Info:
{company_text[:2500]}
"""
    return get_llm_answer(prompt)

def get_risk_assessment(full_text):
    """Analyze potential risks in the RFP, grouped by risk level with reduction recommendations"""
    prompt = f"""
You are a contract risk analyst.

Your job is to review the following RFP content and identify risky clauses that may negatively impact the company. Categorize and provide specific risk-reduction recommendations.

🔍 Focus on these risk areas:
- Termination: {risk_recommendations['termination']}
- Liability: {risk_recommendations['liability']}
- Insurance: {risk_recommendations['insurance']}
- Risk: {risk_recommendations['risk']}
- Penalty: {risk_recommendations['penalty']}
- Damages: {risk_recommendations['damages']}
- Unilateral: {risk_recommendations['unilateral']}

🧠 For each risk:
1. Write a short description of the clause or issue.
2. Assign a risk level: 🔴 High / 🟠 Medium / 🟢 Low.
3. Give a practical recommendation to reduce or mitigate that risk.

📄 Final format (Group by Risk Level):

### 🔴 High Risk
- **Risk**: [Short description]
  **Recommendation**: [How to mitigate it]

### 🟠 Medium Risk
- **Risk**: [Short description]
  **Recommendation**: [How to mitigate it]

### 🟢 Low Risk
- **Risk**: [Short description]
  **Recommendation**: [How to mitigate it]

Only include the most relevant 3–5 points per category (if any).

RFP Content:
{full_text[:3000]}

Answer:
"""
    return get_llm_answer(prompt)

def get_competitive_analysis(full_text):
    prompt = f"""
Based on this RFP, provide insights on competitive factors:
1. What key capabilities would give a bidder an advantage?
2. Are there incumbent advantages suggested in the RFP?
3. What past performance references would be most relevant?
4. What specific expertise seems most valued?
5. Are there any hints about budget constraints?

Focus on objective insights from the document text.

RFP text:
{full_text[:5000]}
"""
    return get_llm_answer(prompt)

def get_clarification_questions(full_text):
    prompt = f"""
Review this RFP and generate 5-7 important clarification questions:
1. Focus on vague or ambiguous requirements
2. Address potential scope issues
3. Clarify evaluation criteria
4. Question any unusual requirements
5. Seek additional information on key deliverables

Format as specific, well-formed questions that could be submitted to the contracting officer.

RFP text:
{full_text[:5000]}
"""
    return get_llm_answer(prompt)

# ----------------------- Sidebar -----------------------

with st.sidebar:
    st.markdown('<div class="sidebar-header">📄 Document Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload RFP Document", type="pdf")
    company_file = st.file_uploader("Upload Company Data", type="pdf")
    concise_mode = st.checkbox("🧾 Use Concise Report Format", value=True)
    
    if uploaded_file and st.button("🚀 Generate Full Report", use_container_width=True):
        st.session_state['run_analysis'] = True
    else:
        if 'run_analysis' not in st.session_state:
            st.session_state['run_analysis'] = False

def save_history_to_db(user_query, response, rfp_name, company_name=None, eligibility_result=None, checklist=None):
    doc = {
        "timestamp": datetime.utcnow(),
        "query": user_query,
        "response": response,
        "rfp_pdf": rfp_name,
        "company_pdf": company_name,
        "eligibility_result": eligibility_result,
        "checklist": checklist,
    }
    collection.insert_one(doc)

# Sidebar History
st.sidebar.header("🕘 History")
if st.sidebar.button("Show Last 5"):
    for item in collection.find().sort("timestamp", -1).limit(5):
        st.sidebar.markdown("---")
        st.sidebar.write(f"🕐 {item['timestamp']}")
        st.sidebar.write(f"🔍 {item['query']}")
        if "response" in item:
            st.sidebar.code(item['response'][:300])

# ----------------------- Main Content -----------------------

if not uploaded_file:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Welcome to RFP Analyzer Pro")
    st.markdown("""
    This tool automatically generates comprehensive reports from government RFPs, saving you hours of analysis time.
    
    *How it works:*
    1. Upload your RFP document in the sidebar
    2. Upload your company data PDF for eligibility analysis
    3. Click "Generate Full Report" to create a comprehensive analysis
    
    *The report includes:*
    - Executive summary and key information
    - Important dates and deadlines
    - Eligibility assessment (if company data provided)
    - Submission requirements checklist
    - Evaluation criteria analysis
    - Risk assessment and red flags (with grouped risk levels and recommendations)
    - Competitive analysis insights
    - Recommended clarification questions
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
elif st.session_state['run_analysis']:
    # Process the files and generate report
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name
    
    company_path = None
    company_text = ""
    if company_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(company_file.read())
            company_path = tmp.name
            company_text = extract_text_from_pdf(company_path)
    
    # Process files
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Extracting text from PDF...")
    full_text = extract_text_from_pdf(file_path)
    if not full_text:
        st.error("Failed to extract text from the PDF. Please try another file.")
        st.stop()
    progress_bar.progress(10)
    
    status_text.text("Chunking document for analysis...")
    chunks = chunk_text(full_text)
    if not chunks:
        st.error("Failed to chunk the document text. Please try another file.")
        st.stop()
    progress_bar.progress(20)
    
    status_text.text("Building vector database...")
    index, all_chunks, embeddings = build_vector_store(chunks)
    if index is None:
        st.error("Failed to build vector store. Please try again.")
        st.stop()
    progress_bar.progress(30)
    
    # Generate report sections
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="main-header" style="text-align: left; font-size: 1.8rem;">RFP Analysis Report</div>', unsafe_allow_html=True)
        st.caption(f"Generated on {time.strftime('%B %d, %Y')} • Document: {uploaded_file.name}")
        
        # Executive Summary and Key Information
        status_text.text("Generating executive summary...")
        with st.expander("📝 RFP Overview", expanded=True):
            st.markdown('<div class="subheader">Executive Summary</div>', unsafe_allow_html=True)
            summary = get_summary(full_text)
            st.markdown(summary)
            
            st.markdown('<div class="subheader">Key Information</div>', unsafe_allow_html=True)
            key_info = get_key_info(full_text)
            st.markdown(key_info)
        progress_bar.progress(40)
        
        # Key Dates
        status_text.text("Extracting key dates...")
        with st.expander("📅 Key Dates & Deadlines", expanded=True):
            dates = get_key_dates(full_text)
            st.markdown(dates)
        progress_bar.progress(50)
        
        # Eligibility Assessment
        if company_text:
            status_text.text("Analyzing eligibility...")
            with st.expander("🛡️ Eligibility Assessment", expanded=True):
                if concise_mode:
                    eligibility = get_concise_eligibility(full_text, company_text)
                else:
                    eligibility = get_eligibility(full_text, company_text)
                # Display a UI badge based on the result
                if "❌ Not Eligible" in eligibility:
                    st.error("❌ Not Eligible")
                elif "✅ Eligible" in eligibility:
                    st.success("✅ Eligible")
                else:
                    st.warning("⚠️ Eligibility status unclear.")
                st.markdown(eligibility)
        progress_bar.progress(60)
        
        # Submission Requirements
        status_text.text("Extracting submission requirements...")
        with st.expander("📋 Submission Requirements", expanded=True):
            requirements = get_submission_requirements(full_text)
            st.markdown(requirements)
        progress_bar.progress(70)
        
        # Evaluation Criteria
        status_text.text("Analyzing evaluation criteria...")
        with st.expander("⚖️ Evaluation Criteria", expanded=True):
            criteria = get_evaluation_criteria(full_text)
            st.markdown(criteria)
        progress_bar.progress(80)
        
        # Risk Assessment (with grouped risk levels and recommendations)
        status_text.text("Performing risk assessment...")
        with st.expander("⚠️ Risk Assessment", expanded=True):
            risks = get_risk_assessment(full_text)
            st.markdown(risks)
        progress_bar.progress(85)
        
        # Competitive Analysis
        status_text.text("Creating competitive analysis...")
        with st.expander("🏆 Competitive Analysis", expanded=True):
            competitive = get_competitive_analysis(full_text)
            st.markdown(competitive)
        progress_bar.progress(90)
        
        # Clarification Questions (optional)
        status_text.text("Generating clarification questions...")
        with st.expander("❓ Recommended Clarification Questions", expanded=True):
            questions = get_clarification_questions(full_text)
            st.markdown(questions)
        progress_bar.progress(95)
        
    # Side panel with summary and key information
    with col2:
        st.markdown('<div class="status-card">', unsafe_allow_html=True)
        st.markdown("### Document Analysis")
        st.markdown(f"*Pages:* ~{len(chunks)//2}")
        st.markdown(f"*Content Chunks:* {len(chunks)}")
        st.markdown(f"*Analysis Date:* {time.strftime('%Y-%m-%d')}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="summary-box">', unsafe_allow_html=True)
        st.markdown("### Key Takeaways")
        takeaways_prompt = f"""
Based on the RFP text, provide 5 key takeaways or important points that a bidder should focus on.
Keep each takeaway to 1-2 sentences.

RFP text:
{full_text[:3000]}
"""
        takeaways = get_llm_answer(takeaways_prompt)
        st.markdown(takeaways)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("### Critical Deadlines")
        deadlines_prompt = f"""
Extract only the 2-3 most critical upcoming deadlines from this RFP.
Format as: "Deadline Name: Date"
Only include specific dates mentioned in the document.

RFP text:
{full_text[:3000]}
"""
        deadlines = get_llm_answer(deadlines_prompt)
        st.markdown(deadlines)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown("### Win Strategy Tips")
        win_prompt = f"""
Based on the RFP, provide 3-4 specific win strategy tips that would help create a competitive proposal.
Each tip should be actionable and specific to this opportunity.

RFP text:
{full_text[:3000]}
"""
        win_tips = get_llm_answer(win_prompt)
        st.markdown(win_tips)
        st.markdown('</div>', unsafe_allow_html=True)
    
    progress_bar.progress(100)
    status_text.success("✅ Report Generation Complete!")
    
    # Cleanup temporary files
    os.remove(file_path)
    if company_path:
        os.remove(company_path)
