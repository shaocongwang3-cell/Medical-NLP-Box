import os
import logging

logger = logging.getLogger(__name__)


class LLMNotConfiguredError(RuntimeError):
    pass


def get_llm(llm_options: dict):
    """按配置创建 LLM。未安装 Ollama / 未配置 API Key 时给出明确错误。"""
    provider = (llm_options or {}).get("provider", "ollama")
    model = (llm_options or {}).get("model", "qwen2.5:7b")

    if provider == "ollama":
        try:
            from langchain_community.llms import Ollama
            return Ollama(model=model)
        except Exception as exc:
            raise LLMNotConfiguredError(
                "当前未配置可用的 Ollama。可稍后安装 Ollama 并拉取模型，或改用 OpenAI。"
            ) from exc

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMNotConfiguredError("未设置 OPENAI_API_KEY，无法使用 OpenAI。")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=0, api_key=api_key)

    raise ValueError(f"Unsupported LLM provider: {provider}")
