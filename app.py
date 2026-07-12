import os
import sys
import logging
import streamlit as st
import chromadb
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter

# # 0. Enable Real-Time Console Logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# # 1. Setup Streamlit Page UI
st.set_page_config(page_title="NCGSA GNSS Copilot", page_icon="🛰️", layout="wide")
st.title("🛰️ NCGSA GNSS Copilot")
st.subheader("Research Assistant for Global Navigation Satellite Systems")

DATA_DIR = "data"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "gnss_knowledge_base"

@st.cache_resource
def initialize_local_system():
    try:
        # # 2. Configure Global Settings
        import os

        # Check if running locally or deployed on the cloud internet
        if os.environ.get("STREAMLIT_RUNTIME_ENV") == "cloud":
            from llama_index.llms.groq import Groq
            groq_api_key = os.environ.get("GROQ_API_KEY")
            local_llm = Groq(model="llama3-8b-8192", api_key=groq_api_key)
        else:
            local_llm = Ollama(model="llama3.2:1b", request_timeout=60.0, context_window=2048)

        local_embed = OllamaEmbedding(model_name="nomic-embed-text")

        Settings.llm = local_llm
        Settings.embed_model = local_embed

        # # 3. Setup Local ChromaDB Vector DB Storage
        db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        
        # Check if collection exists and has existing vectors
        collections = [c.name for c in db.list_collections()]
        db_exists = COLLECTION_NAME in collections
        
        chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Reusable strict system prompt configuration
        strict_system_prompt = (
            "You are a strict GNSS and Ionospheric Research Assistant for NCGSA. "
            "Your task is to answer questions using ONLY the provided text chunks. "
            "VTEC stands for Vertical Total Electron Content. Do not use external general knowledge "
            "or mention vehicle communications. If the answer cannot be found in the context, "
            "say 'I cannot find that specific detail in the loaded documents.'"
        )

        #  Optimization: Check files in the collection directly
        try:
            existing_count = chroma_collection.count()
        except Exception:
            existing_count = 0

        if db_exists and existing_count > 0:
            print(f"\n=== SUCCESS: FOUND {existing_count} VECTORS IN CHROMADB ===")
            index = VectorStoreIndex.from_vector_store(
                vector_store,
                storage_context=storage_context
            )
            print("=== DATABASE LOADED INSTANTLY ===\n")
            return index.as_query_engine(
                similarity_top_k=1,
                system_prompt=strict_system_prompt
            )

        # # 4. Read and Index PDF Documents Step-by-Step (Only runs if DB is empty)
        print("\n=== STARTING DATA LOADING ===")
        reader = SimpleDirectoryReader(DATA_DIR, filename_as_id=True, recursive=False)
        documents = reader.load_data()
        print(f"Successfully loaded {len(documents)} document pages/sections.")

        print("Splitting documents into text chunks...")
        text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=30)
        nodes = text_splitter.get_nodes_from_documents(documents)

        print("Building vector index chunk-by-chunk...")
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            show_progress=True
        )
        print("=== INDEXING COMPLETE ===\n")
        
        return index.as_query_engine(
            similarity_top_k=1,
            system_prompt=strict_system_prompt
        )
        
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        return None

# # Initialize engine
query_engine = initialize_local_system()

# # 5. Build Simple Chat History Interface
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your local GNSS Copilot. Ask me anything about your research papers."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask about GNSS anomalies, VTEC, or thesis metrics..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if query_engine:
        with st.spinner("Searching local vector database..."):
            response = query_engine.query(prompt)
            st.session_state.messages.append({"role": "assistant", "content": str(response)})
            st.chat_message("assistant").write(str(response))
    else:
        st.error("Engine failed to respond. Please verify Ollama is running.")