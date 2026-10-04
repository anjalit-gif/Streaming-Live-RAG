import os


class Settings:
    # Swappable backend config
    USE_HOSTED_BACKEND: bool = os.getenv("USE_HOSTED_BACKEND", "false").lower() == "true"
    HOSTED_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LOCAL_LLM_URL: str = os.getenv("LOCAL_LLM_URL", "http://localhost:11434/api/generate")
    # Small model by default - an 8B model is too slow for a responsive CPU demo.
    LOCAL_LLM_MODEL: str = os.getenv("LOCAL_LLM_MODEL", "llama3.2:1b")

    # RAG tuning parameters
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))
    RRF_K: int = int(os.getenv("RRF_K", "60"))
    MIN_WORDS_BEFORE_RETRIEVAL: int = int(os.getenv("MIN_WORDS_BEFORE_RETRIEVAL", "3"))


settings = Settings()
