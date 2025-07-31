import streamlit as st
import random
import pandas as pd
from PyPDF2 import PdfReader
from typing import List
import numpy as np
import faiss
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import requests
import json
import io

# --- Streamlit page config and CSS ---
st.set_page_config(page_title="🧠 Office Chatbot", layout="wide")
st.markdown("""
    <style>
    .main { padding-left: 1vw !important; padding-right: 1vw !important; max-width: 100vw; }
    .block-container { padding-top: 1.2rem; padding-bottom: 0.7rem; padding-left: 0.7rem !important; padding-right: 0.7rem !important; width: 100vw; }
    .element-container { width: 100%; }
    .stMarkdown { font-size: 16px; }
    .section-header { font-weight:700; font-size:17px; margin-bottom: 8px; margin-top: 2px; }
    .chat-row { margin-top: 1.1em; margin-bottom: 1.4em; }
    .user-block { margin-bottom: 0.5em; margin-top: 0.7em; }
    .attachment-block { margin-bottom: 0.7em; color: #68696b; font-size: 15px; }
    </style>
""", unsafe_allow_html=True)

# --- API keys --- (unchanged)
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
PERPLEXITY_API_KEY = st.secrets.get("PERPLEXITY_API_KEY", "")
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
if not GEMINI_API_KEY or not PERPLEXITY_API_KEY or not GOOGLE_API_KEY:
    st.error("Missing API key(s). Please check your .streamlit/secrets.toml.")
    st.stop()
genai.configure(api_key=GEMINI_API_KEY)

# --- Greetings ---
greetings_list = [
    "📈 Meetings don’t end. They just get forwarded. Let’s be useful for once.",
    "🧾 Today’s agenda: 1. Survive. 2. Assist you. 3. Coffee.",
    "👨‍💼 Corporate drama pending, but answers? Already brewing.",
    "🚨 Deadline seen. Panic mode activated (internally).",
    "🧠 The only assistant here that won’t ghost you before EOD.",
    "☕ Caffeine: 97%. Sarcasm: 3%. Ready to help!",
    "📊 If Excel had emotions, I’d be it. Precise. Overwhelmed. Reliable.",
    "💡 We don’t vibe here—we align on deliverables.",
    "🤖 No small talk. Just large productivity.",
    "📎 Clippy retired. I took over. Let’s get you un-confused.",
    "🎯 Working smarter so you don’t have to work harder.",
    "💬 Conversations here don’t get ‘Seen 2 hours ago’.",
    "🏃‍♂️ Sprinting through tasks like it’s Q4.",
    "🔍 Your search bar in formalwear.",
    "📦 Deliverables? Let’s package them with panache.",
    "👨‍🔬 Coffee is my fuel. Queries are my experiment.",
    "📚 Well-read. Overworked. Ready.",
    "🚪 Open door policy. Closed tab energy.",
    "📌 You ask. I’ll pin down the answers.",
    "🔐 NDA-safe, HR-proof replies incoming.",
    "🛠️ No bugs here. Only byte-sized solutions.",
    "🎢 Today’s vibe: Corporate rollercoaster. Buckle up.",
    "🧾 I speak fluent ‘action items’.",
    "📁 You bring chaos. I bring folders.",
    "🧭 Lost in workflow? I’ve got directions.",
    "🪑 Watercooler chat’s overrated. Ask me instead.",
    "🌪️ Inbox a mess? Let me be your calm.",
    "🧾 Procrastinating professionally since 9:01am. Let’s fix that.",
    "🤝 Let’s pretend your boss is watching—ask smart.",
    "🖋️ We write. We rewrite. We conquer corporate clutter."
]

# -- PDF helper
@st.cache_data(show_spinner=False)
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    reader = PdfReader(pdf_bytes)
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text)

# -- CSV/Excel helper
@st.cache_data(show_spinner=False)
def extract_text_from_tabular(file_data: bytes, filetype: str) -> str:
    # filetype: 'csv', 'xlsx'
    if filetype == 'csv':
        df = pd.read_csv(io.BytesIO(file_data))
    elif filetype == 'xlsx':
        df = pd.read_excel(io.BytesIO(file_data), engine="openpyxl")
    # Show preview table
    with st.expander(f"🔎 Data preview ({filetype.upper()})", expanded=False):
        st.dataframe(df.head(10))
    # Export first 20 rows as Markdown table for LLM context
    limited_df = df.head(20)
    return f"Data preview (first 20 rows):\n{limited_df.to_markdown(index=False)}"

# -- Text chunking (unchanged)
def text_to_chunks(text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(" ".join(tokens[start:end]))
        start += chunk_size - overlap
    return chunks

# -- Faiss vector index (unchanged)
@st.cache_resource(show_spinner=False)
def build_faiss_index(chunks: List[str]):
    embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    embeddings = embedder.embed_documents(chunks)
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))
    return index, np.array(embeddings).astype('float32')

def query_index(query: str, chunks: List[str], index, embeddings_array, k: int = 3) -> str:
    embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    q_emb = embedder.embed_query(query)
    q_vec = np.array(q_emb).astype('float32').reshape(1, -1)
    _, I = index.search(q_vec, k)
    return "\n---\n".join(chunks[i] for i in I[0])

# -- Ask Gemini, especially for data-analysis queries
def ask_gemini(question, context=None, is_data=False):
    if is_data and context:
        prompt = (
            f"The following is a preview of a data table. Please analyze it in detail: summarize main trends, distribution, any anomalies, outliers, and statistics; then answer the specific user query, referring to the data.\n\n{context}\n\nUser Question:\n{question}"
        )
    else:
        prompt = f"{context}\n\nQuestion:\n{question}" if context else question
    try:
        model = genai.GenerativeModel("models/gemini-2.5-pro")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gemini error: {e}"

# -- Ask Perplexity (with better data analysis prompting)
def ask_perplexity(question, context=None, is_data=False):
    if is_data and context:
        prompt = (
            "Analyze the below table. Summarize visible trends, stats, outliers or correlations, and answer the user question as a data analyst, referencing the data.\n\n"
            f"{context}\n\nUser Question: {question}"
        )
    else:
        prompt = f"{context}\n\nQuestion:\n{question}" if context else question
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "Reply straightforwardly and thoroughly as a data-savvy AI assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=40)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Perplexity error: {e}"

# -- Summarize common points between LLMs
def summarize_common_points(gemini_resp, perplexity_resp):
    system_prompt = (
        "Here are two different AI responses to the same question. "
        "Answer in detailed manner ONLY the points that are common between both answers. "
        "Label this: 'Common Crux'."
    )
    combined = (
        f"AI Response 1:\n{gemini_resp}\n\n"
        f"AI Response 2:\n{perplexity_resp}\n\n"
        "Now, " + system_prompt
    )
    try:
        model = genai.GenerativeModel("models/gemini-2.5-pro")
        resp = model.generate_content(combined)
        return resp.text.strip()
    except Exception as e:
        return f"Crux error: {e}"

# --- UI Layout ---
st.markdown("<h1 style='text-align:center; margin-bottom:0.1em;'>💬Chat Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; margin-top:0em;'>Chatbot powered by Gemini & Perplexity</p>", unsafe_allow_html=True)

with st.expander("🧭 What can this assistant do?"):
    st.markdown("""
    - Answer work-related queries (HR, tech, communication, research).
    - Provide perspectives from **Gemini** and **Perplexity** AI.
    - Help you write, rewrite, or summarize content.
    - Support file uploads (PDF, Excel, CSV) for in-depth analysis.
    """)

# --- Greeting ---
if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.greeting = random.choice(greetings_list)
st.markdown(f"<div class='greeting'>{st.session_state.greeting}</div>", unsafe_allow_html=True)

# --- File Upload & Processing ---
uploaded_files = st.file_uploader(
    "📎 Attach PDF, CSV or Excel files (optional):",
    type=["pdf", "csv", "xlsx"],
    accept_multiple_files=True
)
file_type_texts = []
file_types_present = []
for file in uploaded_files or []:
    filetype = file.name.split('.')[-1].lower()
    if filetype == "pdf":
        file_types_present.append("pdf")
        with st.spinner(f"Extracting from PDF: {file.name}..."):
            file_type_texts.append(extract_text_from_pdf(file))
    elif filetype in ("csv", "xlsx"):
        file_types_present.append(filetype)
        with st.spinner(f"Extracting from {filetype.upper()}: {file.name}..."):
            file_type_texts.append(extract_text_from_tabular(file.read(), filetype))

file_text = "\n".join(file_type_texts) if file_type_texts else ''
data_attached = any(ft in ["csv", "xlsx"] for ft in file_types_present)

if uploaded_files and file_text:
    if not data_attached:  # Only chunk-and-index if NOT a spreadsheet
        chunks = text_to_chunks(file_text)
        index, embeddings_array = build_faiss_index(chunks)
    else:
        chunks, index, embeddings_array = None, None, None   # For tabular, not chunking!
else:
    chunks, index, embeddings_array = None, None, None


# --- Input Space/Layout
st.markdown("<div style='margin-top: 1.2em'></div>", unsafe_allow_html=True)
query = st.text_input("📝 Type your question:")
ask_btn = st.button("Ask")

if ask_btn and query.strip():
    # For PDF, classic retrieval; for tabular, pass table data into LLM
    if uploaded_files and file_text:
        if data_attached:
            context = file_text  # summarized table text (markdown)
            context_is_data = True
        else:
            context = query_index(query, chunks, index, embeddings_array, k=3)
            context_is_data = False
    else:
        context = None
        context_is_data = False
    with st.spinner("Thinking..."):
        gemini_resp = ask_gemini(query, context, is_data=context_is_data)
        perplexity_resp = ask_perplexity(query, context, is_data=context_is_data)
        crux = summarize_common_points(gemini_resp, perplexity_resp)
    st.session_state.history.append({
        "query": query,
        "gemini": gemini_resp,
        "perplexity": perplexity_resp,
        "crux": crux,
        "files": [f.name for f in uploaded_files] if uploaded_files else []
    })

# -- Chat History, side-by-side
for chat in reversed(st.session_state.history):
    st.markdown("<div class='chat-row'></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='user-block'><strong>🧑‍💼 You:</strong> {chat['query']}</div>", unsafe_allow_html=True)
    if chat.get("files"):
        st.markdown(f"<div class='attachment-block'>Attached files: {', '.join(chat['files'])}</div>", unsafe_allow_html=True)
    cols = st.columns(2, gap="large")
    with cols[0]:
        st.markdown("<div class='section-header'>🔮 Gemini 2.5 Pro</div>", unsafe_allow_html=True)
        st.markdown(chat['gemini'])
    with cols[1]:
        st.markdown("<div class='section-header'>🛰️ Perplexity Sonar</div>", unsafe_allow_html=True)
        st.markdown(chat['perplexity'])
    st.markdown('<div style="height: 12px;"></div>', unsafe_allow_html=True)
    st.markdown("<div class='section-header' style='color:  #FFFFFF;font-weight: 700; font-size: 18px;'>🤝COMMON CRUX:</div>", unsafe_allow_html=True)
    st.write(chat.get('crux', 'No common points found.'))
    st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)