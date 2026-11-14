import os
import json
import asyncio
import subprocess
import logging
from pathlib import Path

from github import Github, GithubException, InputGitTreeElement

logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, token):
        if not token:
            raise ValueError("GITHUB_TOKEN manquant dans .env")
        self.token = token
        self.g = Github(token, timeout=30)
        self.allowed_shell = os.getenv("ALLOW_SHELL", "false").lower() in (
            "true",
            "1",
            "yes",
        )
        self.workspace = Path("workspace/repos").resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    def _repo(self, name):
        try:
            return self.g.get_repo(name)
        except GithubException as exc:
            raise RuntimeError(f"Repo inaccessible : {exc}") from exc

    def _split_repo(self, repo_name):
        if "/" not in repo_name:
            raise ValueError("Le repo doit etre au format owner/repo")
        owner, repo = repo_name.split("/", 1)
        return owner, repo

    def list_dir(self, repo_name, path="", branch="main"):
        repo = self._repo(repo_name)
        contents = repo.get_contents(path, ref=branch)
        if not isinstance(contents, list):
            contents = [contents]
        result = {
            "repo": repo_name,
            "path": path,
            "branch": branch,
            "items": [
                {
                    "name": c.name,
                    "path": c.path,
                    "type": c.type,
                    "sha": getattr(c, "sha", None),
                }
                for c in contents
            ],
        }
        logger.info(f"list_dir {repo_name}/{path} -> {len(result['items'])} items")
        return result

    def read_file(self, repo_name, path, branch="main"):
        logger.info(f"read_file {repo_name}/{path}")
        repo = self._repo(repo_name)
        content_file = repo.get_contents(path, ref=branch)
        if isinstance(content_file, list):
            return {"error": f"{path} est un dossier, pas un fichier"}
        content = content_file.decoded_content.decode("utf-8", errors="replace")
        return {
            "path": path,
            "branch": branch,
            "size": len(content),
            "content": content,
        }

    def create_file(self, repo_name, path, content, message="", branch="main"):
        logger.info(f"create_file {repo_name}/{path}")
        repo = self._repo(repo_name)
        result = repo.create_file(
            path, message or f"Create {path}", content, branch=branch
        )
        return {
            "created": path,
            "sha": result["content"].sha,
            "commit": result["commit"].sha,
        }

    def update_file(self, repo_name, path, content, message="", branch="main"):
        logger.info(f"update_file {repo_name}/{path}")
        repo = self._repo(repo_name)
        old = repo.get_contents(path, ref=branch)
        if isinstance(old, list):
            return {"error": f"{path} est un dossier"}
        result = repo.update_file(
            path,
            message or f"Update {path}",
            content,
            old.sha,
            branch=branch,
        )
        return {
            "updated": path,
            "sha": result["content"].sha,
            "commit": result["commit"].sha,
        }

    def delete_file(self, repo_name, path, message="", branch="main"):
        logger.info(f"delete_file {repo_name}/{path}")
        repo = self._repo(repo_name)
        old = repo.get_contents(path, ref=branch)
        if isinstance(old, list):
            return {"error": f"{path} est un dossier"}
        result = repo.delete_file(
            path, message or f"Delete {path}", old.sha, branch=branch
        )
        return {"deleted": path, "commit": result["commit"].sha}

    def batch_commit(self, repo_name, changes, message, branch="main"):
        logger.info(f"batch_commit {repo_name} -> {len(changes)} fichiers")
        repo = self._repo(repo_name)

        ref = repo.get_git_ref(f"heads/{branch}")
        base_commit = repo.get_git_commit(ref.object.sha)
        base_tree = repo.get_git_tree(base_commit.tree.sha, recursive=True)

        changed_paths = {c["path"] for c in changes}
        new_tree = []

        for item in base_tree.tree:
            if item.path in changed_paths:
                continue
            new_tree.append(
                InputGitTreeElement(
                    path=item.path, mode=item.mode, type=item.type, sha=item.sha
                )
            )

        for change in changes:
            mode = change.get("mode", "update")
            path = change["path"]

            if mode == "remove":
                continue

            content = change.get("content", "")
            blob = repo.create_git_blob(content, "utf-8")
            new_tree.append(
                InputGitTreeElement(path=path, mode="100644", type="blob", sha=blob.sha)
            )

        tree = repo.create_git_tree(new_tree, base_tree=base_tree)
        new_commit = repo.create_git_commit(message, tree, [base_commit])
        ref.edit(new_commit.sha)

        return {
            "ok": True,
            "commit": new_commit.sha,
            "files": [c["path"] for c in changes],
        }

    def git_push_changes(self, repo_name, changes, message, branch="main"):
        logger.info(f"git_push_changes {repo_name} -> {len(changes)} fichiers")
        try:
            from git import Repo, Actor
        except ImportError:
            return {
                "error": "GitPython non installe. Lance : pip install GitPython"
            }

        owner, repo = self._split_repo(repo_name)
        local_path = self.workspace / owner / repo

        token_url = f"https://{self.token}@github.com/{repo_name}.git"
        clean_url = f"https://github.com/{repo_name}.git"
        env = {"GIT_TERMINAL_PROMPT": "0"}

        try:
            if not local_path.exists():
                local_path.parent.mkdir(parents=True, exist_ok=True)
                Repo.clone_from(
                    token_url,
                    str(local_path),
                    env=env,
                    branch=branch,
                )

            repo_path = str(local_path)
            repo_obj = Repo(repo_path)

            for remote in repo_obj.remotes:
                if "@" in remote.url:
                    remote.set_url(clean_url)

            if branch in repo_obj.heads:
                repo_obj.git.checkout(branch)
            else:
                repo_obj.git.checkout("-b", branch, f"origin/{branch}")

            try:
                repo_obj.git.reset("--hard", f"origin/{branch}")
            except Exception:
                pass

            repo_root = Path(repo_obj.working_tree_dir)
            modified = []
            for change in changes:
                path = change["path"]
                mode = change.get("mode", "update")
                target = repo_root / path

                if mode == "remove":
                    if target.exists():
                        target.unlink()
                        modified.append(path)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(change.get("content", ""), encoding="utf-8")
                    modified.append(path)

            if not modified:
                return {"info": "Aucun fichier fourni a modifier"}

            repo_obj.git.add(*modified)

            if not repo_obj.is_dirty(untracked_files=True):
                return {"info": "Rien a commiter (fichiers identiques)"}

            author = Actor("TG-Agent", "tg-agent@bot.local")
            repo_obj.index.commit(message, author=author, committer=author)

            push_out = repo_obj.git.push(token_url, branch)

            return {
                "ok": True,
                "repo": repo_name,
                "branch": branch,
                "message": message,
                "committed": modified,
                "push_output": push_out.strip(),
            }

        except Exception as exc:
            logger.exception("git_push_changes erreur")
            return {"error": str(exc)}

    def run_shell(self, command, cwd=None):
        logger.info(f"run_shell: {command[:80]}")
        if not self.allowed_shell:
            return {
                "error": "Shell desactive. Active ALLOW_SHELL=true uniquement en environnement isole."
            }
        cwd = cwd or os.getcwd()
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            return {
                "command": command,
                "returncode": proc.returncode,
                "stdout": proc.stdout[:10000],
                "stderr": proc.stderr[:10000],
            }
        except Exception as exc:
            return {"error": str(exc)}

    def _execute_sync(self, name, raw_args):
        logger.info(f"Outil appele: {name} avec args: {raw_args[:200]}")
        args = json.loads(raw_args)

        if name == "list_dir":
            return self.list_dir(
                args["repo"], args.get("path", ""), args.get("branch", "main")
            )
        if name == "read_file":
            return self.read_file(args["repo"], args["path"], args.get("branch", "main"))
        if name == "create_file":
            return self.create_file(
                args["repo"],
                args["path"],
                args["content"],
                args.get("message", ""),
                args.get("branch", "main"),
            )
        if name == "update_file":
            return self.update_file(
                args["repo"],
                args["path"],
                args["content"],
                args.get("message", ""),
                args.get("branch", "main"),
            )
        if name == "delete_file":
            return self.delete_file(
                args["repo"],
                args["path"],
                args.get("message", ""),
                args.get("branch", "main"),
            )
        if name == "batch_commit":
            return self.batch_commit(
                args["repo"],
                args["changes"],
                args["message"],
                args.get("branch", "main"),
            )
        if name == "git_push_changes":
            return self.git_push_changes(
                args["repo"],
                args["changes"],
                args["message"],
                args.get("branch", "main"),
            )
        if name == "run_shell":
            return self.run_shell(args["command"], args.get("cwd"))

        return {"error": f"Outil inconnu : {name}"}

    async def execute_tool_calls(self, tool_calls):
        results = []
        loop = asyncio.get_event_loop()

        for tc in tool_calls:
            try:
                out = await loop.run_in_executor(
                    None,
                    self._execute_sync,
                    tc.function.name,
                    tc.function.arguments,
                )
            except Exception as exc:
                logger.exception(f"Erreur outil {tc.function.name}")
                out = {"error": str(exc)}

            content = json.dumps(out, ensure_ascii=False, default=str)
            if len(content) > 20000:
                content = content[:20000] + "\n... [tronque]"

            results.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": content,
                }
            )
        return results