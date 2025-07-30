import streamlit as st
import random
from PyPDF2 import PdfReader
from typing import List
import numpy as np
import faiss
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import requests
import json

# --- Load API Keys ---
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
PERPLEXITY_API_KEY = st.secrets.get("PERPLEXITY_API_KEY", "")
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
if not GEMINI_API_KEY or not PERPLEXITY_API_KEY or not GOOGLE_API_KEY:
    st.error("Missing API key(s). Please check your .streamlit/secrets.toml.")
    st.stop()
genai.configure(api_key=GEMINI_API_KEY)

# --- Fun Greetings (Randomize) ---
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

# --- PDF Helpers ---
@st.cache_data(show_spinner=False)
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    reader = PdfReader(pdf_bytes)
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text)

def text_to_chunks(text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(" ".join(tokens[start:end]))
        start += chunk_size - overlap
    return chunks

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

def ask_gemini(question, context=None):
    prompt = f"{context}\n\nQuestion:\n{question}" if context else question
    try:
        model = genai.GenerativeModel("models/gemini-2.5-pro")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gemini error: {e}"

def ask_perplexity(question, context=None):
    prompt = f"{context}\n\nQuestion:\n{question}" if context else question
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "Reply usefully as an AI assistant addressing everything without missing important aspects."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=40)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Perplexity error: {e}"

# --- UI Styling ---
st.set_page_config(page_title="🧠 Office Chatbot", layout="centered")
st.markdown("""
    <style>
    .stMarkdown { font-size: 16px; }
    .section-header {
        font-weight:700;
        font-size:17px;
        margin-bottom: 8px;
        margin-top: 2px;
    }
    .chat-row {
        margin-top: 1.1em;
        margin-bottom: 1.4em;
    }
    .user-block {
        margin-bottom: 0.5em;
        margin-top: 0.7em;
    }
    .attachment-block {
        margin-bottom: 0.7em;
        color: #68696b;
        font-size: 15px;
    }
    /* No color override! Leave as inherited! */
    </style>
""", unsafe_allow_html=True)

# --- Central Heading & Tagline ---
st.markdown("<h1 style='text-align:center; margin-bottom:0.1em;'>💬Chat Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; margin-top:0em;'>Chatbot powered by Gemini & Perplexity</p>", unsafe_allow_html=True)

with st.expander("🧭 What can this assistant do?"):
    st.markdown("""
    - Answer work-related queries (HR, tech, communication, research).
    - Provide perspectives from **Gemini** and **Perplexity** AI.
    - Help you write, rewrite, or summarize content.
    - Support file uploads (for reference or review).
    """)

# --- Greeting: show at top, reset if session starts ---
if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.greeting = random.choice(greetings_list)
st.markdown(f"<div class='greeting'>{st.session_state.greeting}</div>", unsafe_allow_html=True)

# --- File Upload & Processing ---
uploaded_files = st.file_uploader("📎 Attach PDF files (optional):", type=["pdf"], accept_multiple_files=True)
file_text = ""
if uploaded_files:
    for file in uploaded_files:
        with st.spinner(f"Extracting from {file.name}..."):
            file_text += extract_text_from_pdf(file)
    if file_text:
        with st.expander("🔎 Preview extracted content"):
            st.write(file_text[:1500] + ("..." if len(file_text) > 1500 else ""))

if uploaded_files and file_text:
    chunks = text_to_chunks(file_text)
    index, embeddings_array = build_faiss_index(chunks)
else:
    chunks, index, embeddings_array = None, None, None

# --- Input Space/Layout
st.markdown("<div style='margin-top: 1.2em'></div>", unsafe_allow_html=True)
query = st.text_input("📝 Type your question:")
ask_btn = st.button("Ask")

if ask_btn and query.strip():
    if uploaded_files and chunks and index is not None:
        context = query_index(query, chunks, index, embeddings_array, k=3)
    else:
        context = None
    with st.spinner("Thinking..."):
        gemini_resp = ask_gemini(query, context)
        perplexity_resp = ask_perplexity(query, context)
    st.session_state.history.append({
        "query": query,
        "gemini": gemini_resp,
        "perplexity": perplexity_resp,
        "files": [f.name for f in uploaded_files] if uploaded_files else []
    })

# --- Chat History, side-by-side, spaced nicely ---
for chat in reversed(st.session_state.history):
    st.markdown("<div class='chat-row'></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='user-block'><strong>🧑‍💼 You:</strong> {chat['query']}</div>", unsafe_allow_html=True)
    if chat.get("files"):
        st.markdown(f"<div class='attachment-block'>Attached files: {', '.join(chat['files'])}</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("<div class='section-header'>🔮 Gemini 2.5 Pro</div>", unsafe_allow_html=True)
        st.markdown(chat['gemini'])
    with col2:
        st.markdown("<div class='section-header'>🛰️ Perplexity Sonar</div>", unsafe_allow_html=True)
        st.markdown(chat['perplexity'])
    st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)  # Space between turns
