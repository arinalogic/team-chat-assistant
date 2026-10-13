import hashlib
from datetime import datetime, timezone
from typing import Any

from haystack import Document, Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

from config import Config


class PipelineManager:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.document_store = PineconeDocumentStore(
            api_key=Secret.from_token(config.pinecone_api_key),
            index=config.pinecone_index_name,
            namespace=config.pinecone_namespace,
            dimension=1536,
        )

        self.indexing_pipeline = self._build_indexing_pipeline()
        self.search_pipeline = self._build_search_pipeline()
        self.summary_pipeline = self._build_summary_pipeline()

    def _build_indexing_pipeline(self) -> Pipeline:
        pipeline = Pipeline()
        pipeline.add_component(
            "embedder",
            OpenAIDocumentEmbedder(
                api_key=Secret.from_token(self.config.openai_api_key),
                api_base_url=self.config.openai_base_url,
                model=self.config.embedding_model,
                dimensions=1536,
                raise_on_failure=True,
            ),
        )
        pipeline.add_component(
            "writer",
            DocumentWriter(
                document_store=self.document_store,
                policy=DuplicatePolicy.OVERWRITE,
            ),
        )
        pipeline.connect("embedder.documents", "writer.documents")
        return pipeline

    def _build_search_pipeline(self) -> Pipeline:
        pipeline = Pipeline()
        pipeline.add_component(
            "text_embedder",
            OpenAITextEmbedder(
                api_key=Secret.from_token(self.config.openai_api_key),
                api_base_url=self.config.openai_base_url,
                model=self.config.embedding_model,
                dimensions=1536,
            ),
        )
        pipeline.add_component(
            "retriever",
            PineconeEmbeddingRetriever(document_store=self.document_store, top_k=5),
        )
        pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        return pipeline

    def _build_summary_pipeline(self) -> Pipeline:
        prompt_template = [
            ChatMessage.from_system(
                "Ты командный помощник. Всегда отвечай по-русски. "
                "Опирайся только на переданный контекст, не выдумывай факты и источники. "
                "Для ответа на вопрос указывай автора, время и содержание каждого использованного сообщения. "
                "Если контекста недостаточно, честно скажи об этом."
            ),
            ChatMessage.from_user(
                "Режим: {{ mode }}\n"
                "Вопрос: {{ question }}\n"
                "Контекст:\n{{ context }}\n\n"
                "Если режим — итог сессии: определи, обсуждали ли участники спор или решение о действиях. "
                "При споре кратко изложи позиции сторон и добавь взвешенное мнение. "
                "При решении перечисли договорённости в формате «кто — что делает». "
                "В остальных случаях дай короткое резюме. "
                "Если режим — ответ на вопрос, ответь на вопрос по контексту и приведи ссылки на сообщения "
                "в формате [автор, время]."
            ),
        ]

        pipeline = Pipeline()
        pipeline.add_component(
            "prompt_builder",
            ChatPromptBuilder(
                template=prompt_template,
                required_variables=["mode", "question", "context"],
            ),
        )
        pipeline.add_component(
            "chat_generator",
            OpenAIChatGenerator(
                api_key=Secret.from_token(self.config.openai_api_key),
                api_base_url=self.config.openai_base_url,
                model=self.config.openai_model,
            ),
        )
        pipeline.connect("prompt_builder.prompt", "chat_generator.messages")
        return pipeline

    def index_message(
        self,
        content: str,
        author_name: str,
        author_id: int,
        chat_id: int,
        message_id: int,
        timestamp: str,
        username: str | None = None,
    ) -> None:
        document_id = hashlib.sha256(f"{chat_id}:{message_id}".encode("utf-8")).hexdigest()
        metadata: dict[str, Any] = {
            "author_name": author_name,
            "author_id": author_id,
            "chat_id": chat_id,
            "message_id": message_id,
            "timestamp": timestamp,
        }
        if username:
            metadata["username"] = username

        document = Document(id=document_id, content=content, meta=metadata)
        self.indexing_pipeline.run({"embedder": {"documents": [document]}})

    def search_context(self, query: str, chat_id: int) -> list[Document]:
        chat_filter = {"field": "meta.chat_id", "operator": "==", "value": chat_id}
        result = self.search_pipeline.run(
            {
                "text_embedder": {"text": query},
                "retriever": {"filters": chat_filter, "top_k": 5},
            }
        )
        return result["retriever"]["documents"]

    def summarize_session(self, messages: list[str]) -> str:
        context = "\n".join(messages)
        return self._generate_text(mode="итог сессии", question="", context=context)

    def answer_question(self, question: str, found_docs: list[Document]) -> str:
        context_lines = []
        for document in found_docs:
            metadata = document.meta or {}
            author = str(metadata.get("author_name", "Неизвестный автор"))
            username = metadata.get("username")
            if username:
                author = f"{author} (@{username})"
            timestamp = str(metadata.get("timestamp", "Время не указано"))
            context_lines.append(f"[{author}, {timestamp}] {document.content or ''}")

        context = "\n".join(context_lines) or "Подходящие сообщения не найдены."
        return self._generate_text(mode="ответ на вопрос", question=question, context=context)

    def _generate_text(self, mode: str, question: str, context: str) -> str:
        result = self.summary_pipeline.run(
            {
                "prompt_builder": {
                    "template_variables": {
                        "mode": mode,
                        "question": question,
                        "context": context,
                    }
                }
            }
        )
        replies = result["chat_generator"]["replies"]
        if not replies:
            return "Не удалось подготовить ответ."
        return replies[0].text or "Не удалось подготовить ответ."


def format_timestamp(unix_time: int) -> str:
    return datetime.fromtimestamp(unix_time, tz=timezone.utc).isoformat()