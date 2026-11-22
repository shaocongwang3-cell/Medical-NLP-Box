import logging
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from dotenv import load_dotenv
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
from config import DB_DIR  # noqa: E402
file_path = BACKEND_DIR / "data" / "SNOMED_5000.csv"
persist_dir = DB_DIR / "snomed_bge_m3"
collection_name = "concepts_only_name"
batch_size = 64

persist_dir.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
# 优先使用本地已下载目录，避免联网；本地不存在时才回退到 HuggingFace 仓库名。
MODEL_REPO = "BAAI/bge-m3"
LOCAL_MODEL_DIR = os.getenv("BGE_M3_LOCAL_DIR", "E:/bge-m3-local")
model_name = LOCAL_MODEL_DIR if Path(LOCAL_MODEL_DIR).exists() else MODEL_REPO
logging.info("Loading embedding model %s on %s", model_name, device)
model = SentenceTransformer(model_name, device=device, trust_remote_code=True)

logging.info("Loading data from %s", file_path)
df = pd.read_csv(file_path, dtype=str, low_memory=False).fillna("NA")

client = PersistentClient(path=str(persist_dir))
if collection_name in [c.name for c in client.list_collections()]:
    client.delete_collection(collection_name)
collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

for start_idx in tqdm(range(0, len(df), batch_size), desc="Indexing SNOMED"):
    end_idx = min(start_idx + batch_size, len(df))
    batch_df = df.iloc[start_idx:end_idx]
    docs = [str(row["concept_name"]) for _, row in batch_df.iterrows()]
    embeddings = model.encode(docs, normalize_embeddings=True).tolist()
    ids = [f"concept-{start_idx + idx}" for idx in range(len(batch_df))]
    metadatas = [
        {
            "concept_id": str(row["concept_id"]),
            "concept_name": str(row["concept_name"]),
            "domain_id": str(row["domain_id"]),
            "vocabulary_id": str(row["vocabulary_id"]),
            "concept_class_id": str(row["concept_class_id"]),
            "standard_concept": str(row["standard_concept"]),
            "concept_code": str(row["concept_code"]),
        }
        for _, row in batch_df.iterrows()
    ]
    collection.add(ids=ids, embeddings=embeddings, documents=docs, metadatas=metadatas)

logging.info("Indexed %s concepts into %s", collection.count(), persist_dir)

query = "SOB"
query_embedding = model.encode([query], normalize_embeddings=True).tolist()
search_result = collection.query(query_embeddings=query_embedding, n_results=5, include=["metadatas", "documents", "distances"])
logging.info("Search result for '%s': %s", query, search_result)
