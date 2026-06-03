import os
import logging
from openai import AsyncOpenAI
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
MODEL = "meta-llama/llama-3.3-70b-instruct"
MAX_TOKENS = 8192
SYSTEM_PROMPT = (
    "You are a helpful, concise, and friendly AI assistant. "
    "Answer clearly and accurately. If you don't know something, say so."
)

client = AsyncOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

user_histories: dict[int, list[dict]] = {}


def get_history(user_id: int) -> list[dict]:
    if user_id not in user_histories:
        user_histories[user_id] = []
    return user_histories[user_id]


def trim_history(history: list[dict], max_turns: int = 20) -> list[dict]:
    if len(history) > max_turns * 2:
        return history[-(max_turns * 2):]
    return history


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    await update.message.reply_text(
        f"Hi {user.first_name}! I'm your AI assistant powered by Llama 3.3 70B via OpenRouter.\n\n"
        "Just send me a message and I'll respond. Use /clear to reset the conversation history."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Commands:\n"
        "/start — Welcome message\n"
        "/help — Show this help\n"
        "/clear — Clear conversation history\n"
        "/model — Show current AI model\n\n"
        "Just type anything to chat with the AI!"
    )


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_histories.pop(user_id, None)
    await update.message.reply_text("Conversation history cleared.")


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(f"Current model: `{MODEL}`", parse_mode="Markdown")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_text = update.message.text

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )

    history = get_history(user_id)
    history.append({"role": "user", "content": user_text})
    history = trim_history(history)
    user_histories[user_id] = history

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    try:
        response = await client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=messages,
        )
        reply = response.choices[0].message.content or "(No response)"
        history.append({"role": "assistant", "content": reply})
        user_histories[user_id] = history

        if len(reply) > 4096:
            for i in range(0, len(reply), 4096):
                await update.message.reply_text(reply[i:i + 4096])
        else:
            await update.message.reply_text(reply)

    except Exception as e:
        logger.error("OpenRouter error: %s", e)
        await update.message.reply_text(
            "Sorry, I ran into an error contacting the AI. Please try again."
        )


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Unhandled exception: %s", context.error, exc_info=context.error)


def main() -> None:
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(CommandHandler("model", model_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    logger.info("Bot started. Polling for updates...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
  
