import dotenv
dotenv.load_dotenv()
from langchain_community.embeddings import BedrockEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
import boto3
import os
from utils.embedding_config import EmbeddingProvider, EmbeddingConfig
from config import resolve_model_path

class EmbeddingFactory:
    @staticmethod
    def create_embedding_function(config: EmbeddingConfig):
        if config.provider == EmbeddingProvider.BEDROCK:
            bedrock_client = boto3.client(
                service_name='bedrock-runtime',
                region_name=config.aws_region,
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
            )
            return BedrockEmbeddings(
                client=bedrock_client,
                model_id=config.model_name
            )
            
        elif config.provider == EmbeddingProvider.OPENAI:
            return OpenAIEmbeddings(
                model=config.model_name,
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )
            
        elif config.provider == EmbeddingProvider.HUGGINGFACE:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
            return HuggingFaceEmbeddings(
                model_name=resolve_model_path(config.model_name),
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )
            
        raise ValueError(f"Unsupported embedding provider: {config.provider}")