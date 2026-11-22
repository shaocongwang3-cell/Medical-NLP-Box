from dotenv import load_dotenv
from utils.embedding_factory import EmbeddingFactory
from utils.embedding_config import EmbeddingProvider, EmbeddingConfig
from config import chroma_persist_dir
from typing import List, Dict
import logging
import chromadb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class StdService:
    """
    医学术语标准化服务
    使用 Chroma 向量库进行医学术语的标准化和相似度搜索
    """

    def __init__(
        self,
        provider="huggingface",
        model="BAAI/bge-m3",
        db_path="db/snomed_bge_m3.db",
        collection_name="concepts_only_name",
    ):
        provider_mapping = {
            "openai": EmbeddingProvider.OPENAI,
            "bedrock": EmbeddingProvider.BEDROCK,
            "huggingface": EmbeddingProvider.HUGGINGFACE,
        }

        embedding_provider = provider_mapping.get(provider.lower())
        if embedding_provider is None:
            raise ValueError(f"Unsupported provider: {provider}")

        config = EmbeddingConfig(
            provider=embedding_provider,
            model_name=model,
        )
        self.embedding_func = EmbeddingFactory.create_embedding_function(config)

        persist_name = db_path.replace("\\", "/").split("/")[-1]
        persist_dir = chroma_persist_dir(persist_name)
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Chroma collection %s at %s, count=%s",
            collection_name,
            persist_dir,
            self.collection.count(),
        )

    def search_similar_terms(self, query: str, limit: int = 5) -> List[Dict]:
        query_embedding = self.embedding_func.embed_query(query)
        n_results = min(limit, max(self.collection.count(), 1))
        if self.collection.count() == 0:
            logger.warning("Chroma collection is empty. Run tools/create_chroma_db.py first.")
            return []

        search_result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["metadatas", "distances", "documents"],
        )

        results = []
        metadatas = search_result.get("metadatas", [[]])[0]
        distances = search_result.get("distances", [[]])[0]
        documents = search_result.get("documents", [[]])[0]
        for metadata, distance, document in zip(metadatas, distances, documents):
            metadata = metadata or {}
            results.append(
                {
                    "concept_id": metadata.get("concept_id"),
                    "concept_name": metadata.get("concept_name") or document,
                    "domain_id": metadata.get("domain_id"),
                    "vocabulary_id": metadata.get("vocabulary_id"),
                    "concept_class_id": metadata.get("concept_class_id"),
                    "standard_concept": metadata.get("standard_concept"),
                    "concept_code": metadata.get("concept_code"),
                    "synonyms": metadata.get("synonyms"),
                    "distance": float(distance) if distance is not None else None,
                }
            )
        return results
