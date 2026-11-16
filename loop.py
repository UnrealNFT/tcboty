import os
import re
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
# httpx loggue l'URL complete (qui contient le token Telegram) : on le fait taire.
logging.getLogger("httpx").setLevel(logging.WARNING)

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

def _safe_tool_id(raw) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]", "_", str(raw or ""))
    s = re.sub(r"_+", "_", s).strip("_")
    return (s or "call")[:64]


def prepare_messages(messages: list) -> list:
    """Ids d'outils valides pour Claude, et jamais un resultat sans son appel."""
    mapping = {}
    used = set()

    def assign(old):
        old = str(old or "")
        if old in mapping:
            return mapping[old]
        base = _safe_tool_id(old)
        new = base
        i = 2
        while new in used:
            new = f"{base}_{i}"[:64]
            i += 1
        mapping[old] = new
        used.add(new)
        return new

    cleaned = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                tc = dict(tc)
                tc["id"] = assign(tc.get("id"))
                calls.append(tc)
            m["tool_calls"] = calls
        if m.get("role") == "tool" and "tool_call_id" in m:
            m["tool_call_id"] = assign(m.get("tool_call_id"))
        cleaned.append(m)

    out = []
    i = 0
    while i < len(cleaned):
        m = cleaned[i]
        calls = m.get("tool_calls") or []
        if not calls:
            if m.get("role") == "tool":
                i += 1
                continue
            out.append(m)
            i += 1
            continue
        need = [tc.get("id") for tc in calls]
        j = i + 1
        got = []
        while j < len(cleaned) and cleaned[j].get("role") == "tool":
            got.append(cleaned[j])
            j += 1
        by_id = {g.get("tool_call_id"): g for g in got}
        if all(n in by_id for n in need):
            out.append(m)
            for n in need:
                out.append(by_id[n])
        elif (m.get("content") or "").strip():
            out.append({"role": "assistant", "content": m.get("content")})
        i = j
    return out


def _trim_history(messages: list, max_chars: int = 80000):
    if not messages:
        return messages
    system = messages[0] if messages[0].get("role") == "system" else None
    body = messages[1:] if system else list(messages)
    groups = []
    for m in body:
        if m.get("role") == "tool" and groups:
            groups[-1].append(m)
        else:
            groups.append([m])
    kept = []
    total = len(str((system or {}).get("content", "")))
    for g in reversed(groups):
        size = len(json.dumps(g, ensure_ascii=False))
        if kept and total + size > max_chars:
            break
        kept.append(g)
        total += size
    kept.reverse()
    out = [system] if system else []
    for g in kept:
        out.extend(g)
    return prepare_messages(out)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    logger.info(f"/start recu de user_id={user.id}")
    await update.message.reply_text(
        f"Salut {user.first_name}.\n"
        "Par defaut : Kimi K2.7.\n"
        "/k3 ta demande\n"
        "/claude ta demande\n"
        "/grok ta demande",
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


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE, model=None):
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    if user_id not in ALLOWED_USER_IDS:
        await update.message.reply_text("Acces refuse. Ton user ID n'est pas autorise.")
        return

    if model:
        incoming = " ".join(context.args).strip()
        if not incoming:
            await update.message.reply_text("Ecris la demande apres la commande.")
            return
    else:
        incoming = update.message.text
        model = brain.MODEL

    logger.info(f"Message de {user_id} model={model}: {incoming[:120]}...")

    messages = load_history(chat_id)
    messages.append({"role": "user", "content": incoming})

    await update.message.chat.send_action(action="typing")

    final_answer = ""
    max_steps = 15

    try:
        for step in range(max_steps):
            logger.info(f"=== Boucle LLM step {step + 1}/{max_steps} ===")
            assistant_msg = await brain.chat(prepare_messages(messages), model=model)
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

        messages = _trim_history(prepare_messages(messages))
        save_history(chat_id, messages)

        logger.info(f"Envoi reponse: {final_answer[:200]}...")
        await safe_reply(update, final_answer)

    except Exception as exc:
        logger.exception("Erreur boucle")
        await update.message.reply_text(f"Erreur : {exc}")


def _commande(cle):
    async def handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await handle_message(update, context, model=brain.MODELS[cle])
    return handler


if __name__ == "__main__":
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN manquant")

    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("k3", _commande("k3")))
    app.add_handler(CommandHandler("claude", _commande("claude")))
    app.add_handler(CommandHandler("grok", _commande("grok")))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot lance")
    app.run_polling()
