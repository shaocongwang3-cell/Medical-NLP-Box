import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
# Chroma 向量库根目录。默认放在用户主目录（纯 ASCII），
# 避免中文路径导致 hnswlib 无法持久化/加载索引（Cannot open header file）。
# 可用环境变量 MEDNLP_DB_DIR 覆盖。
DB_DIR = Path(os.getenv("MEDNLP_DB_DIR", str(Path.home() / "medical_nlp_db")))

# 本地已下载模型目录映射：把 HuggingFace 仓库名解析为本地路径，避免联网下载。
# 可通过环境变量覆盖默认目录（例如 BGE_M3_LOCAL_DIR）。
LOCAL_MODEL_DIRS = {
    "BAAI/bge-m3": os.getenv("BGE_M3_LOCAL_DIR", "E:/bge-m3-local"),
}


def resolve_model_path(model_name: str) -> str:
    """若配置了该模型的本地目录且存在，则返回本地路径，否则原样返回仓库名。"""
    local = LOCAL_MODEL_DIRS.get(model_name)
    if local and Path(local).exists():
        return local
    return model_name


def chroma_persist_dir(db_name: str) -> Path:
    """把前端传入的库名（可带 .db）转成 Chroma 持久化目录。"""
    name = db_name[:-3] if db_name.endswith(".db") else db_name
    path = DB_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path
