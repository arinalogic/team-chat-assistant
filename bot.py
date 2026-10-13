import collections
import logging
import re

from haystack import Document
import telebot

from config import Config, load_config
from logging_setup import setup_logging
from pipelines import PipelineManager, format_timestamp


def main() -> None:
    logger = setup_logging()

    try:
        config = load_config()
        manager = PipelineManager(config)
        bot = telebot.TeleBot(config.telegram_bot_token, threaded=False)
        bot_info = bot.get_me()
        bot_username = bot_info.username or ""
        bot_user_id = bot_info.id
        sessions: dict[int, list[str]] = {}
        recent_messages: dict[int, collections.deque[dict[str, str | int]]] = {}

        def send_text(chat_id: int, text: str) -> None:
            try:
                bot.send_message(chat_id, text)
            except Exception:
                logger.exception("Не удалось отправить ответ в чат %s", chat_id)
                raise
            logger.info("Ответ отправлен в чат %s", chat_id)

        def send_error(chat_id: int) -> None:
            try:
                send_text(chat_id, "Не удалось обработать сообщение. Попробуйте ещё раз позже.")
            except Exception:
                logger.exception("Не удалось отправить сообщение об ошибке в чат %s", chat_id)

        @bot.message_handler(commands=["start"])
        def handle_start(message: telebot.types.Message) -> None:
            try:
                send_text(
                    message.chat.id,
                    "Я сохраняю сообщения чата и помогаю подводить итоги. "
                    "Для сессии используйте /start_listening и /stop_listening.",
                )
            except Exception:
                logger.exception("Ошибка команды /start в чате %s", message.chat.id)
                send_error(message.chat.id)

        @bot.message_handler(commands=["start_listening"])
        def start_listening(message: telebot.types.Message) -> None:
            try:
                sessions[message.chat.id] = []
                logger.info("Сессия обсуждения начата в чате %s", message.chat.id)
                send_text(message.chat.id, "Начала собирать сообщения для итогов обсуждения.")
            except Exception:
                logger.exception("Ошибка команды /start_listening в чате %s", message.chat.id)
                send_error(message.chat.id)

        @bot.message_handler(commands=["stop_listening"])
        def stop_listening(message: telebot.types.Message) -> None:
            chat_id = message.chat.id
            session_messages = None
            try:
                session_messages = sessions.pop(chat_id, None)
                if session_messages is None:
                    send_text(chat_id, "Сейчас сессия обсуждения не запущена.")
                    return

                if not session_messages:
                    send_text(chat_id, "В сессии пока нет сообщений для итогов.")
                    return

                logger.info("Суммаризация сессии в чате %s", chat_id)
                summary = manager.summarize_session(session_messages)
                send_text(chat_id, summary)
            except Exception:
                logger.exception("Ошибка при суммаризации сессии в чате %s", chat_id)
                if session_messages:
                    sessions[chat_id] = session_messages
                send_error(chat_id)

        @bot.message_handler(content_types=["text"])
        def handle_text(message: telebot.types.Message) -> None:
            text = message.text or ""
            chat_id = message.chat.id
            logger.info("Получено текстовое сообщение из чата %s", chat_id)

            if not text or text.startswith("/"):
                return

            author = message.from_user
            if author is None or author.is_bot:
                return

            should_answer = _should_answer(message, bot_username, bot_user_id)

            try:
                timestamp = format_timestamp(message.date)
                author_name = " ".join(
                    part for part in (author.first_name, author.last_name) if part
                ) or str(author.id)

                logger.info("Индексация сообщения из чата %s", chat_id)
                manager.index_message(
                    content=text,
                    author_name=author_name,
                    author_id=author.id,
                    chat_id=chat_id,
                    message_id=message.message_id,
                    timestamp=timestamp,
                    username=author.username,
                )

                chat_messages = recent_messages.setdefault(
                    chat_id, collections.deque(maxlen=15)
                )
                recent_message: dict[str, str | int] = {
                    "author_name": author_name,
                    "timestamp": timestamp,
                    "text": text,
                    "message_id": message.message_id,
                }
                if author.username:
                    recent_message["username"] = author.username
                chat_messages.append(recent_message)

                if chat_id in sessions:
                    sessions[chat_id].append(f"[{author_name}, {timestamp}] {text}")
                    logger.info(
                        "Сообщение добавлено в сессию чата %s: всего сообщений=%d",
                        chat_id,
                        len(sessions[chat_id]),
                    )

                if should_answer:
                    question = _remove_bot_mention(text, bot_username).strip() or text
                    logger.info("Поиск контекста в чате %s", chat_id)
                    documents = manager.search_context(question, chat_id)
                    context_messages, search_count, recent_count = _combine_answer_context(
                        documents,
                        list(recent_messages[chat_id]),
                        message.message_id,
                    )
                    logger.info(
                        "Контекст ответа: найдено поиском=%d, добавлено из последних=%d, всего=%d",
                        search_count,
                        recent_count,
                        len(context_messages),
                    )
                    logger.info("Подготовка ответа на вопрос в чате %s", chat_id)
                    answer = manager.answer_question(question, context_messages)
                    send_text(chat_id, answer)
            except Exception:
                logger.exception("Ошибка при обработке сообщения из чата %s", chat_id)
                if should_answer:
                    send_error(chat_id)

        logger.info("Запуск Telegram-бота")
        bot.infinity_polling()
    except Exception:
        logger.exception("Бот не смог запуститься")
        raise


def _should_answer(message: telebot.types.Message, bot_username: str, bot_user_id: int) -> bool:
    if message.chat.type == "private":
        return True

    if bot_username and re.search(rf"@{re.escape(bot_username)}\b", message.text or "", re.IGNORECASE):
        return True

    for entity in message.entities or []:
        if entity.type == "text_mention" and entity.user and entity.user.id == bot_user_id:
            return True

    replied_message = message.reply_to_message
    replied_author = getattr(replied_message, "from_user", None)
    return replied_author is not None and replied_author.id == bot_user_id


def _remove_bot_mention(text: str, bot_username: str) -> str:
    if not bot_username:
        return text
    return re.sub(rf"@{re.escape(bot_username)}\b", "", text, flags=re.IGNORECASE)


def _combine_answer_context(
    found_documents: list[Document],
    recent_messages: list[dict[str, str | int]],
    current_message_id: int,
) -> tuple[list[dict[str, str | int]], int, int]:
    messages_by_id: dict[int, dict[str, str | int]] = {}
    search_count = 0

    for document in found_documents:
        metadata = document.meta or {}
        message_id = metadata.get("message_id")
        if message_id is None or int(message_id) == current_message_id:
            continue

        numeric_message_id = int(message_id)
        if numeric_message_id in messages_by_id:
            continue

        pinecone_message: dict[str, str | int] = {
            "author_name": str(metadata.get("author_name", "Неизвестный автор")),
            "timestamp": str(metadata.get("timestamp", "")),
            "text": document.content or "",
            "message_id": numeric_message_id,
        }
        username = metadata.get("username")
        if username:
            pinecone_message["username"] = str(username)
        messages_by_id[numeric_message_id] = pinecone_message
        search_count += 1

    recent_count = 0
    for message in recent_messages:
        message_id = int(message["message_id"])
        if message_id == current_message_id or message_id in messages_by_id:
            continue
        messages_by_id[message_id] = message
        recent_count += 1

    context_messages = sorted(
        messages_by_id.values(),
        key=lambda message: (str(message["timestamp"]), int(message["message_id"])),
    )
    return context_messages, search_count, recent_count


if __name__ == "__main__":
    main()