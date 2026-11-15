import os
import logging
import httpx
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

client = AsyncOpenAI(
    api_key=os.environ["WAVESPEED_API_KEY"],
    base_url="https://llm.wavespeed.ai/v1",
    timeout=600.0,
    max_retries=2,
    http_client=httpx.AsyncClient(timeout=httpx.Timeout(600.0)),
)

MODEL = os.getenv("WAVESPEED_MODEL", "moonshotai/kimi-k2.7-code")

MODELS = {
    "kimi": "moonshotai/kimi-k2.7-code",
    "k3": "moonshotai/kimi-k3",
    "claude": "anthropic/claude-opus-5.5",
    "grok": "x-ai/grok-4.7",
}
DEFAULT_REPO = os.getenv("DEFAULT_REPO", "")

SYSTEM_PROMPT = f"""Tu es un agent codeur controle depuis Telegram.
Ton role : discuter, analyser et modifier du code sur GitHub.
- Quand tu lis un repo de référence pour t'inspirer, ne lis pas tous les HTML séparément. Demande d'abord un résumé à l'utilisateur ou lis seulement les fichiers clés.
- Sois efficace : ne lis pas plus de 3 fichiers par étape. soit chirurgical dans tes lectures et tes reponses
Regles strictes :
- Confirme toujours les actions destructrices.
- Pour modifier du code : lis d'abord le fichier avec read_file, puis mets a jour.
- Si plusieurs fichiers changent, utilise batch_commit (API GitHub) ou git_push_changes (vrai git clone/commit/push).
- Prefere git_push_changes quand l'utilisateur dit explicitement "commit", "pusher", "git push".
- Le repo par defaut est : {DEFAULT_REPO or 'aucun (precise owner/repo)'}.
- Reponds en francais, concis mais structure.
- N'invente jamais de resultats.

Outils disponibles :
- list_dir / read_file / create_file / update_file / delete_file : actions directes API GitHub (auto-commit).
- batch_commit : commit atomique multi-fichiers via API GitHub.
- git_push_changes : clone en local, applique les modifs, puis `git commit` + `git push`.
- run_shell : commande shell (desactive par defaut).
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "Lister le contenu d'un dossier d'un repo GitHub.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "owner/repo"},
                    "path": {"type": "string", "default": ""},
                    "branch": {"type": "string", "default": "main"},
                },
                "required": ["repo"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Lire le contenu d'un fichier GitHub.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "owner/repo"},
                    "path": {"type": "string"},
                    "branch": {"type": "string", "default": "main"},
                },
                "required": ["repo", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Creer un fichier dans un repo GitHub (commit automatique).",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "owner/repo"},
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "message": {"type": "string", "default": ""},
                    "branch": {"type": "string", "default": "main"},
                },
                "required": ["repo", "path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_file",
            "description": "Modifier un fichier existant d'un repo GitHub (commit automatique).",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "owner/repo"},
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "message": {"type": "string", "default": ""},
                    "branch": {"type": "string", "default": "main"},
                },
                "required": ["repo", "path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Supprimer un fichier d'un repo GitHub (commit automatique).",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "owner/repo"},
                    "path": {"type": "string"},
                    "message": {"type": "string", "default": ""},
                    "branch": {"type": "string", "default": "main"},
                },
                "required": ["repo", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "batch_commit",
            "description": "Commit atomique multi-fichiers via l'API GitHub.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "owner/repo"},
                    "changes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"},
                                "mode": {
                                    "type": "string",
                                    "enum": ["add", "update", "remove"],
                                },
                            },
                            "required": ["path", "mode"],
                        },
                    },
                    "message": {"type": "string"},
                    "branch": {"type": "string", "default": "main"},
                },
                "required": ["repo", "changes", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_push_changes",
            "description": "Clone le repo en local, applique plusieurs fichiers, puis `git commit` et `git push` avec le token.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "owner/repo"},
                    "branch": {"type": "string", "default": "main"},
                    "message": {"type": "string", "description": "Message du commit"},
                    "changes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"},
                                "mode": {
                                    "type": "string",
                                    "enum": ["add", "update", "remove"],
                                },
                            },
                            "required": ["path", "mode"],
                        },
                    },
                },
                "required": ["repo", "changes", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Executer une commande shell sur la machine locale (desactive par defaut).",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "cwd": {"type": "string", "default": ""},
                },
                "required": ["command"],
            },
        },
    },
]


async def chat(messages, model=None):
    chosen = model or MODEL
    logger.info(f"Wavespeed request model={chosen} messages={len(messages)}")
    response = await client.chat.completions.create(
        model=chosen,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )
    if getattr(response, "error", None):
        raise RuntimeError(response.error)
    msg = response.choices[0].message
    logger.info(f"Wavespeed response: role={msg.role} content_len={len(msg.content or '')} tool_calls={bool(getattr(msg, 'tool_calls', None))}")
    return msg


def assistant_to_dict(msg):
    d = {"role": msg.role, "content": msg.content or ""}
    if getattr(msg, "tool_calls", None):
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    return d