import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    telegram_bot_token: str
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    embedding_model: str
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_namespace: str


def load_config() -> Config:
    load_dotenv()

    names = (
        "TELEGRAM_BOT_TOKEN",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "EMBEDDING_MODEL",
        "PINECONE_API_KEY",
        "PINECONE_INDEX_NAME",
        "PINECONE_NAMESPACE",
    )
    values = {name: os.getenv(name, "").strip() for name in names}
    missing = [name for name, value in values.items() if not value]

    if missing:
        missing_names = ", ".join(missing)
        raise ValueError(
            f"Не заданы обязательные переменные окружения: {missing_names}. "
            "Добавьте их в файл .env."
        )

    return Config(
        telegram_bot_token=values["TELEGRAM_BOT_TOKEN"],
        openai_api_key=values["OPENAI_API_KEY"],
        openai_base_url=values["OPENAI_BASE_URL"],
        openai_model=values["OPENAI_MODEL"],
        embedding_model=values["EMBEDDING_MODEL"],
        pinecone_api_key=values["PINECONE_API_KEY"],
        pinecone_index_name=values["PINECONE_INDEX_NAME"],
        pinecone_namespace=values["PINECONE_NAMESPACE"],
    )