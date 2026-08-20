import os
import torch
import qdrant_client
from collections import defaultdict
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 0. CPU Multi-Threading Setup
num_cores = os.cpu_count() or 4
torch.set_num_threads(num_cores)
torch.set_num_interop_threads(num_cores)

# 1. Connect to Qdrant Cloud
load_dotenv(override=True)

url = os.getenv("QDRANT_URL")
api_key = os.getenv("QDRANT_API_KEY")

print(f"🔑 Loaded Key Ending: ...{api_key[-10:]}" if api_key else "❌ ERROR: QDRANT_API_KEY NOT FOUND")
print(f"☁️ Connecting to Qdrant Cloud at '{url}'...")

# Initialize Client explicitly
client = qdrant_client.QdrantClient(
    url=url,
    api_key=api_key,
    timeout=60
)

collection_name = "gnss_dataset"

# Safe collection recreation without triggering 404 on collection_exists()
try:
    client.delete_collection(collection_name=collection_name)
    print(f"🗑️ Deleted existing collection: {collection_name}")
except Exception:
    pass

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)
print(f"✅ Created fresh collection: {collection_name}")

# 2. Load Model
print("🧠 Loading BAAI/bge-small-en-v1.5 Model...")
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    embed_batch_size=64
)

# 3. Read PDFs & Merge Pages
print("📄 Parsing documents from ./data folder...")
raw_docs = SimpleDirectoryReader("./data").load_data()

docs_by_file = defaultdict(str)
for doc in raw_docs:
    fname = doc.metadata.get("file_name", "document.pdf")
    docs_by_file[fname] += doc.text + "\n"

# 4. Strict 3,500-Character Chunks (~600 words each)
chunks = []
metadatas = []

for fname, full_text in docs_by_file.items():
    if not full_text.strip():
        continue
    # Slice text into 3500-char blocks
    for i in range(0, len(full_text), 3500):
        chunk = full_text[i:i+3500]
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
            metadatas.append(fname)

print(f"🧩 Processing EXACTLY {len(chunks)} total text chunks...")

# 5. Embed & Upsert
embeddings = embed_model.get_text_embedding_batch(chunks, show_progress=True)

# Correctly build points list
points = []
for idx, (chunk, vector, fname) in enumerate(zip(chunks, embeddings, metadatas), start=1):
    points.append(
        PointStruct(
            id=idx,
            vector=vector,
            payload={
                "text": chunk,
                "file_name": fname
            }
        )
    )

print("💾 Uploading vectors to Qdrant Cloud in batches...")

# Batch upsert to prevent request size failures
batch_size = 100
for i in range(0, len(points), batch_size):
    batch = points[i:i + batch_size]
    client.upsert(collection_name=collection_name, points=batch)

print("\n✅ All PDFs successfully indexed to Qdrant Cloud!")