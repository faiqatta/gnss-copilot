import os
import sys
import httpx
import asyncio
import streamlit as st
import qdrant_client
from openai import AsyncOpenAI
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="NCGSA GNSS Copilot",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM GROK-GALACTIC CSS STYLING ---
st.markdown("""
<style>
    /* Cosmic Void Background with High-Contrast Starfield */
    .stApp {
        background-color: #010103;
        background-image: 
            radial-gradient(circle at 50% 12%, rgba(168, 85, 247, 0.16) 0%, transparent 45%),
            radial-gradient(circle at 80% 80%, rgba(56, 189, 248, 0.10) 0%, transparent 50%),
            radial-gradient(1.5px 1.5px at 25px 35px, #ffffff, rgba(0,0,0,0)),
            radial-gradient(2px 2px at 140px 90px, #ffffff, rgba(0,0,0,0)),
            radial-gradient(1.5px 1.5px at 270px 130px, #e2e8f0, rgba(0,0,0,0)),
            radial-gradient(2px 2px at 420px 210px, #ffffff, rgba(0,0,0,0)),
            radial-gradient(1.5px 1.5px at 610px 330px, #ffffff, rgba(0,0,0,0)),
            radial-gradient(2.5px 2.5px at 750px 110px, #ffffff, rgba(0,0,0,0)),
            radial-gradient(1.5px 1.5px at 910px 270px, #e2e8f0, rgba(0,0,0,0)),
            radial-gradient(2px 2px at 1100px 180px, #ffffff, rgba(0,0,0,0));
        background-size: 100% 100%, 100% 100%, 320px 320px, 260px 260px, 380px 380px, 450px 450px, 300px 300px, 500px 500px, 370px 370px, 420px 420px;
        color: #f1f5f9;
    }

    /* Transparent top bar */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Sleek Galactic Dark Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #08080d !important;
        border-right: 1px solid rgba(168, 85, 247, 0.15) !important;
    }

    /* Centered Grok-Galactic Hero Header */
    .grok-hero {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-top: 3vh;
        margin-bottom: 2.5rem;
        text-align: center;
    }

    .grok-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 35%, #c084fc 70%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.03em;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        filter: drop-shadow(0 0 20px rgba(192, 132, 252, 0.25));
    }

    .grok-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-top: 0.5rem;
        font-weight: 400;
        letter-spacing: 0.02em;
    }

    /* Status Telemetry Cards with Nebular Glow */
    .status-card {
        background: rgba(13, 13, 20, 0.75);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 3px solid #c084fc;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.75rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }

    /* Pill-Shaped Grok Chat Input Bar with Cosmic Accent */
    div[data-testid="stChatInput"] {
        background-color: rgba(12, 12, 18, 0.92) !important;
        border: 1px solid rgba(192, 132, 252, 0.25) !important;
        border-radius: 28px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6), 0 0 20px rgba(168, 85, 247, 0.08) !important;
        padding: 4px 8px !important;
        backdrop-filter: blur(12px);
    }

    div[data-testid="stChatInput"]:focus-within {
        border-color: #c084fc !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.7), 0 0 25px rgba(192, 132, 252, 0.25) !important;
    }

    div[data-testid="stChatInput"] textarea {
        color: #f8fafc !important;
        font-size: 0.98rem !important;
    }

    /* Chat Messages Layout */
    .stChatMessage {
        background-color: transparent !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04) !important;
        padding: 1rem 0 !important;
    }

    /* Raw Model Output Expander */
    .streamlit-expanderHeader {
        background-color: rgba(13, 13, 20, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        color: #cbd5e1 !important;
    }

    /* Center Suggestion Prompt Cards */
    div[data-testid="stColumn"] button {
        background-color: rgba(15, 15, 23, 0.75) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        height: 100% !important;
        text-align: left !important;
        color: #e2e8f0 !important;
        transition: all 0.2s ease-in-out !important;
        backdrop-filter: blur(8px) !important;
    }

    div[data-testid="stColumn"] button:hover {
        border-color: #c084fc !important;
        background-color: rgba(24, 24, 38, 0.9) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 20px rgba(192, 132, 252, 0.15) !important;
    }
</style>
""", unsafe_allow_html=True)

load_dotenv()

def get_secret(key):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key)

QDRANT_URL = get_secret("QDRANT_URL")
QDRANT_API_KEY = get_secret("QDRANT_API_KEY")
OPENROUTER_API_KEY = get_secret("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    st.error("⚠️ `OPENROUTER_API_KEY` is missing in environment variables or `.env` file.")
    st.stop()

ai_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "http://localhost:8501",
    }
)

## --- VECTOR DATABASE INITIALIZATION ---
@st.cache_resource
def init_vector_db():
    url = get_secret("QDRANT_URL")
    key = get_secret("QDRANT_API_KEY")
    
    if not url or not key:
        raise ValueError(f"Qdrant credentials missing on Streamlit Cloud! URL present: {bool(url)}, Key present: {bool(key)}")
        
    client = qdrant_client.QdrantClient(url=url, api_key=key)
    vector_store = QdrantVectorStore(client=client, collection_name="gnss_dataset")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
try:
    index = init_vector_db()
    db_status = "Online (Local/Cloud)"
except Exception as e:
    index = None
    db_status = f"Error loading DB: {str(e)}"

# --- DOMAIN ROUTER / CLASSIFIER ---
async def check_gnss_relevance(user_query: str) -> bool:
    gnss_keywords = [=
        "gnss", "gps", "rtk", "ppp", "satellit", "ephemeris", 
        "ublox", "septentrio", "positioning", "ionospher", "tropospher", 
        "carrier-phase", "multipath", "ambiguity", "receiver", "geodesy"
    ]
    if any(k in user_query.lower() for k in gnss_keywords):
        return True

    try:
        res = await ai_client.chat.completions.create(
            model="meta-llama/llama-3.3-70b-instruct",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an intent classifier. Is the user's question related to Global Navigation Satellite Systems (GNSS), GPS, satellite positioning, receivers, or geodesy? Respond with ONLY 'YES' or 'NO'."
                },
                {"role": "user", "content": user_query[:1000]}
            ],
            max_tokens=5,
            temperature=0
        )
        content = res.choices[0].message.content
        return "YES" in content.strip().upper() if content else False
    except Exception:
        return True

# --- MULTI-LLM CONSENSUS ENGINE ---
MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "openrouter/free"
]
SYNTHESIS_MODEL = "openrouter/free"

async def fetch_draft(model_name: str, query: str, context: str):
    try:
        res = await ai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": f"You are an expert GNSS research assistant. Context: {context}"},
                {"role": "user", "content": query}
            ],
            timeout=12
        )
        return {"model": model_name, "content": res.choices[0].message.content}
    except Exception as e:
        return {"model": model_name, "content": f"Error: {str(e)}"}

async def run_consensus_pipeline(user_query: str, context: str):
    tasks = [fetch_draft(m, user_query, context) for m in MODELS]
    drafts = await asyncio.gather(*tasks)
    
    valid_drafts = [d for d in drafts if not d['content'].startswith("Error:")]
    combined_drafts = "\n\n---\n\n".join([f"**{d['model']}**:\n{d['content']}" for d in valid_drafts])
    synthesis_prompt = f"""
You are the Master Editor for NCGSA GNSS Copilot.
Synthesize these candidate model responses for query "{user_query}" into ONE optimized, clear, and accurate expert answer.

Formatting rules:
- Format ALL inline mathematical expressions, variable names, and numerical values with single dollar signs (e.g., $f$, $\text{{TEC}}$, $40.3 \text{{ m}}^3/\text{{s}}^2$).
- Format ALL standalone or multi-line equations using double dollar signs $$ ... $$.
- NEVER use brackets like [ ] or \\[ \\], and NEVER use plain parentheses like ( \\text{{...}} ) for math.

Research Context:
{context}

Candidate Responses:
{combined_drafts}
"""
    
    final_res = await ai_client.chat.completions.create(
        model=SYNTHESIS_MODEL,
        messages=[{"role": "user", "content": synthesis_prompt}]
    )

    return final_res.choices[0].message.content, valid_drafts

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>🛰️</h1>", unsafe_allow_html=True)
    st.title("        Workspace")
    st.caption("NCGSA GNSS Decision Assistant")
    
    st.markdown("---")
    st.subheader("System Telemetry")
    st.markdown(f"""
    <div class="status-card">
        <small style="color:#94a3b8">Vector Database</small><br>
        <strong style="color:#38bdf8">{db_status}</strong>
    </div>
    <div class="status-card">
        <small style="color:#94a3b8">Consensus Panel</small><br>
        <strong style="color:#818cf8">Nemotron-3 & Llama-3.3</strong>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("Suggested Topics")
    if st.button("🛰️ Precise Point Positioning (PPP)"):
        st.session_state.preset_query = "What are the main advantages of PPP over RTK positioning?"
    if st.button("📡 Ionospheric Delays & Errors"):
        st.session_state.preset_query = "How do dual-frequency receivers eliminate ionospheric delay?"
    if st.button("⚙️ Carrier-Phase Ambiguity"):
        st.session_state.preset_query = "Explain integer ambiguity resolution in GNSS receivers."

# --- MAIN HEADER ---
st.markdown("""
<div class="grok-hero">
    <div class="grok-title">
        <span>🪐</span> NCGSA GNSS Copilot
    </div>
    <div class="grok-subtitle">
        Specialized Research & Decision Assistant for Global Navigation Satellite Systems
    </div>
</div>
""", unsafe_allow_html=True)

# --- CHAT INTERFACE ---

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display sample starter cards when the chat is empty
if not st.session_state.messages:
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📡 **Ionospheric Delays**\n\nHow do dual-frequency receivers eliminate delay?"):
            st.session_state.preset_query = "How do dual-frequency receivers eliminate ionospheric delay?"
            st.rerun()
            
        if st.button("🛰️ **PPP vs RTK Positioning**\n\nWhat are the main advantages of PPP over RTK?"):
            st.session_state.preset_query = "What are the main advantages of PPP over RTK positioning?"
            st.rerun()

    with col2:
        if st.button("⚙️ **Carrier-Phase Ambiguity**\n\nExplain integer ambiguity resolution methods."):
            st.session_state.preset_query = "Explain integer ambiguity resolution in GNSS receivers."
            st.rerun()
            
        if st.button("🌐 **Multipath Mitigation**\n\nHow to minimize multipath errors in urban canyons?"):
            st.session_state.preset_query = "How do you minimize multipath errors in urban canyons?"
            st.rerun()
# Display prior messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capture input (or preset query from sidebar)
preset_input = st.session_state.pop("preset_query", None)
user_query = st.chat_input("Ask a question about GNSS, satellite positioning, or dataset metrics...") or preset_input

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)
        
    with st.chat_message("assistant"):
        with st.spinner("🛰️ Verifying domain relevance..."):
            is_valid = asyncio.run(check_gnss_relevance(user_query))
            
        if not is_valid:
            restricted_msg = "⚠️ **Domain Restriction Notice:** I am specifically configured to assist with GNSS (Global Navigation Satellite Systems) and satellite positioning topics. Please ask a GNSS-related query."
            st.warning(restricted_msg)
            st.session_state.messages.append({"role": "assistant", "content": restricted_msg})
        else:
            with st.spinner("🌌 Querying Qdrant Vector DB & Synthesizing Multi-LLM Consensus..."):
                if index:
                    retriever = index.as_retriever(similarity_top_k=3)
                    nodes = retriever.retrieve(user_query)
                    context_str = "\n".join([n.get_content() for n in nodes]) if nodes else "No direct dataset match found. Use general GNSS expert knowledge."
                else:
                    context_str = "Vector database indexing in progress. Use general GNSS expert knowledge."
                
                final_answer, drafts = asyncio.run(run_consensus_pipeline(user_query, context_str))
                
                st.markdown(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                
                with st.expander("🔍 View Raw Outputs from Individual Models (Nemotron, Llama)"):
                    for d in drafts:
                        st.markdown(f"**Model:** `{d['model']}`")
                        st.caption(d['content'])