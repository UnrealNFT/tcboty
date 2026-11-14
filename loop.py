import os
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

import brain
import agent

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

ALLOWED_USER_IDS = {
    int(x.strip())
    for x in os.getenv("ALLOWED_USER_IDS", "").split(",")
    if x.strip().isdigit()
}

MEMORY_DIR = Path("memory")
MEMORY_DIR.mkdir(exist_ok=True)

AGENT = agent.Agent(token=os.getenv("GITHUB_TOKEN"))


def load_history(chat_id: int):
    path = MEMORY_DIR / f"{chat_id}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))["messages"]
        except Exception:
            pass
    return [{"role": "system", "content": brain.SYSTEM_PROMPT}]


def save_history(chat_id: int, messages: list):
    path = MEMORY_DIR / f"{chat_id}.json"
    path.write_text(
        json.dumps({"messages": messages}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def _trim_history(messages: list, max_chars: int = 80000):
    total = sum(len(str(m.get("content", ""))) for m in messages)
    if total <= max_chars:
        return messages
    kept = [messages[0]]
    for m in reversed(messages[1:]):
        kept.insert(1, m)
        if sum(len(str(x.get("content", ""))) for x in kept) > max_chars:
            kept.pop(1)
            break
    return kept


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    logger.info(f"/start recu de user_id={user.id}")
    await update.message.reply_text(
        f"Salut {user.first_name} !\n"
        "Donne-moi des ordres sur ton repo GitHub (format `owner/repo`).\n"
        "Exemple : *dans mon repo toto/projet, ajoute un fichier README.md*",
        parse_mode="Markdown",
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if user.id not in ALLOWED_USER_IDS:
        await update.message.reply_text("Acces refuse.")
        return
    chat_id = update.effective_chat.id
    save_history(chat_id, [{"role": "system", "content": brain.SYSTEM_PROMPT}])
    await update.message.reply_text("Memoire reinitialisee.")


async def safe_reply(update, text):
    text = text or "[reponse vide]"
    try:
        await update.message.reply_text(text, parse_mode="Markdown")
        logger.info(f"Reponse envoyee ({len(text)} chars)")
    except Exception as exc:
        logger.warning(f"Markdown failed: {exc}")
        try:
            await update.message.reply_text(text)
        except Exception as exc2:
            logger.exception("Envoi message failed")
            await update.message.reply_text(f"Erreur envoi: {exc2}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    if user_id not in ALLOWED_USER_IDS:
        await update.message.reply_text("Acces refuse. Ton user ID n'est pas autorise.")
        return

    incoming = update.message.text
    logger.info(f"Message de {user_id}: {incoming[:120]}...")

    messages = load_history(chat_id)
    messages.append({"role": "user", "content": incoming})

    await update.message.chat.send_action(action="typing")

    final_answer = ""
    max_steps = 15

    try:
        for step in range(max_steps):
            logger.info(f"=== Boucle LLM step {step + 1}/{max_steps} ===")
            assistant_msg = await brain.chat(messages)
            messages.append(brain.assistant_to_dict(assistant_msg))

            if getattr(assistant_msg, "tool_calls", None):
                logger.info(f"Tool calls: {[tc.function.name for tc in assistant_msg.tool_calls]}")
                tool_results = await AGENT.execute_tool_calls(assistant_msg.tool_calls)
                logger.info(f"Tool results: {[r['name'] for r in tool_results]}")
                messages.extend(tool_results)
            else:
                final_answer = assistant_msg.content or ""
                logger.info(f"Reponse finale recue ({len(final_answer)} chars)")
                break
        else:
            final_answer = (
                "J'ai atteint la limite d'actions automatiques. "
                "Tu peux reprendre la conversation exactement ou on en est."
            )

        messages = _trim_history(messages)
        save_history(chat_id, messages)

        logger.info(f"Envoi reponse: {final_answer[:200]}...")
        await safe_reply(update, final_answer)

    except Exception as exc:
        logger.exception("Erreur boucle")
        await update.message.reply_text(f"Erreur : {exc}")


if __name__ == "__main__":
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN manquant")

    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot lance")
    app.run_polling()