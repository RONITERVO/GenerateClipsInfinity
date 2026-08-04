from __future__ import annotations

import asyncio
import hashlib
import html as html_lib
import json
import logging
import math
import os
import re
import secrets
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from aiohttp import ClientSession, ClientTimeout


LOGGER = logging.getLogger("wan-video-ui.theater")
THEATER_VERSION = 1


class TheaterError(RuntimeError):
    pass


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def _slug(value: str, fallback: str = "endless-story") -> str:
    clean = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return clean[:55] or fallback


def _json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise TheaterError("The local story writer did not return a JSON object.")
    return json.loads(text[start : end + 1])


class StoryRuntime:
    GEMMA4_E4B_ALIAS = "gemma4-e4b-theater"

    def __init__(self, app_dir: Path, model_root: Path, runtime_root: Path) -> None:
        self.app_dir = app_dir
        self.model_root = model_root
        self.runtime_root = runtime_root
        self.url = "http://127.0.0.1:8083"
        self.process: subprocess.Popen[bytes] | None = None
        self.start_lock = asyncio.Lock()
        self.pid_file = app_dir / "theater-story-writer.pid"
        self.model = model_root / "models" / "gemma-4-E4B-it-Q4_K_M.gguf"
        self.model_alias = self.GEMMA4_E4B_ALIAS
        self.model_label = "Gemma 4 E4B Q4_K_M"
        # Measured fastest decoding on this Ryzen 9 7950X: 15.31 t/s.
        # Eight threads leave the other physical cores for TTS and FFmpeg.
        self.threads = 8
        self.sampling = {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "presence_penalty": 0.0}

    async def healthy(self) -> bool:
        try:
            async with ClientSession(timeout=ClientTimeout(total=2)) as session:
                async with session.get(f"{self.url}/health") as response:
                    if response.status != 200 or (await response.json()).get("status") != "ok":
                        return False
                async with session.get(f"{self.url}/v1/models") as response:
                    data = await response.json(content_type=None)
                    return response.status == 200 and any(
                        item.get("id") == self.model_alias for item in data.get("data", [])
                    )
        except Exception:
            return False

    async def start(self, log_dir: Path) -> None:
        if await self.healthy():
            return
        async with self.start_lock:
            if await self.healthy():
                return
            server = self.runtime_root / "runtime" / "llama-server.exe"
            if not server.exists() or not self.model.exists():
                raise TheaterError(
                    "Gemma 4 E4B is required. Expected "
                    f"{self.model} and the llama.cpp server at {server}."
                )
            log_dir.mkdir(parents=True, exist_ok=True)
            args = [
                str(server), "-m", str(self.model), "--alias", self.model_alias,
                "--host", "127.0.0.1", "--port", "8083", "-ngl", "0",
                "-t", str(self.threads), "-tb", str(self.threads), "-c", "16384", "--parallel", "1",
                "--batch-size", "512", "--ubatch-size", "128", "--no-mmap",
                "--jinja", "--reasoning", "off", "--metrics",
            ]
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            env = os.environ.copy()
            env.update({
                "GGML_CUDA_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES": "",
                "LLAMA_ARG_CHAT_TEMPLATE_KWARGS": '{"enable_thinking":false}',
            })
            with (log_dir / "writer.out.log").open("ab") as out, (log_dir / "writer.err.log").open("ab") as err:
                self.process = subprocess.Popen(
                    args, cwd=self.runtime_root / "runtime", env=env, stdout=out, stderr=err,
                    creationflags=creationflags,
                )
            self.pid_file.write_text(str(self.process.pid), encoding="utf-8")
            for _ in range(180):
                await asyncio.sleep(0.5)
                if await self.healthy():
                    return
                if self.process.poll() is not None:
                    raise TheaterError(f"{self.model_label} exited while loading. Check writer.err.log.")
            raise TheaterError(f"{self.model_label} did not become ready within 90 seconds.")

    async def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(asyncio.to_thread(self.process.wait), timeout=10)
            except asyncio.TimeoutError:
                self.process.kill()
        self.process = None
        self.pid_file.unlink(missing_ok=True)

    async def complete(self, messages: list[dict[str, str]], max_tokens: int = 900) -> tuple[str, dict[str, Any]]:
        started = time.perf_counter()
        body = {
            "model": self.model_alias, "messages": messages, **self.sampling, "repeat_penalty": 1.0,
            "max_tokens": max_tokens, "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        async with ClientSession(timeout=ClientTimeout(total=300)) as session:
            async with session.post(f"{self.url}/v1/chat/completions", json=body) as response:
                data = await response.json(content_type=None)
                if response.status != 200:
                    raise TheaterError(data.get("error", {}).get("message") or str(data))
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        elapsed = max(0.001, time.perf_counter() - started)
        metrics = {
            "elapsed_seconds": round(elapsed, 3),
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "tokens_per_second": round(int(usage.get("completion_tokens", 0)) / elapsed, 2),
        }
        return content, metrics


class SupertonicRuntime:
    """Resident, CPU-only neural narration service."""

    VOICES = {f"{kind}{number}" for kind in ("F", "M") for number in range(1, 6)}
    LANGUAGES = {
        "ar", "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el",
        "hi", "hu", "id", "it", "ja", "ko", "lv", "lt", "pl", "pt", "ro", "ru",
        "sk", "sl", "es", "sv", "tr", "uk", "vi", "na",
    }

    def __init__(self, app_dir: Path, root: Path) -> None:
        self.app_dir = app_dir
        self.root = root
        self.url = "http://127.0.0.1:8084"
        self.process: subprocess.Popen[bytes] | None = None
        self.start_lock = asyncio.Lock()
        self.pid_file = app_dir / "theater-supertonic.pid"

    async def healthy(self) -> bool:
        try:
            async with ClientSession(timeout=ClientTimeout(total=2)) as session:
                async with session.get(f"{self.url}/v1/health") as response:
                    data = await response.json(content_type=None)
                    return response.status == 200 and data.get("status") == "ok"
        except Exception:
            return False

    async def start(self, log_dir: Path) -> None:
        if await self.healthy():
            return
        async with self.start_lock:
            if await self.healthy():
                return
            server = self.root / ".venv" / "Scripts" / "supertonic.exe"
            assets = self.root / "assets"
            if not server.exists() or not (assets / "onnx" / "vocoder.onnx").exists():
                raise TheaterError(f"Supertonic 3 is not installed in {self.root}.")
            log_dir.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update({
                "SUPERTONIC_CACHE_DIR": str(assets),
                "CUDA_VISIBLE_DEVICES": "",
                "ORT_DISABLE_ALL_CUDA": "1",
            })
            args = [
                str(server), "serve", "--host", "127.0.0.1", "--port", "8084",
                "--model", "supertonic-3", "--log-level", "warning",
            ]
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            with (log_dir / "supertonic.out.log").open("ab") as out, (log_dir / "supertonic.err.log").open("ab") as err:
                self.process = subprocess.Popen(
                    args, cwd=self.root, env=env, stdout=out, stderr=err,
                    creationflags=creationflags,
                )
            self.pid_file.write_text(str(self.process.pid), encoding="utf-8")
            for _ in range(180):
                await asyncio.sleep(0.25)
                if await self.healthy():
                    return
                if self.process.poll() is not None:
                    raise TheaterError("Supertonic 3 exited while loading. Check its theater log.")
            raise TheaterError("Supertonic 3 did not become ready within 45 seconds.")

    async def synthesize(
        self, text: str, output: Path, *, voice: str, language: str,
    ) -> float:
        started = time.perf_counter()
        body = {
            "text": text, "voice": voice, "lang": language, "speed": 1.05,
            "steps": 8, "silence_duration": 0.22, "response_format": "wav",
        }
        async with ClientSession(timeout=ClientTimeout(total=300)) as session:
            async with session.post(f"{self.url}/v1/tts", json=body) as response:
                data = await response.read()
                if response.status != 200:
                    raise TheaterError(f"Supertonic narration failed: {data.decode(errors='replace')[:500]}")
        output.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(output.write_bytes, data)
        return time.perf_counter() - started

    async def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(asyncio.to_thread(self.process.wait), timeout=10)
            except asyncio.TimeoutError:
                self.process.kill()
        self.process = None
        self.pid_file.unlink(missing_ok=True)


class KiwixRuntime:
    """Local encyclopedia retrieval for source-grounded educational scenes."""

    def __init__(self, app_dir: Path, root: Path) -> None:
        self.app_dir = app_dir
        self.root = root
        self.url = "http://127.0.0.1:8082"
        self.process: subprocess.Popen[bytes] | None = None
        self.start_lock = asyncio.Lock()
        self.pid_file = app_dir / "theater-kiwix.pid"

    async def healthy(self) -> bool:
        try:
            async with ClientSession(timeout=ClientTimeout(total=2)) as session:
                async with session.get(f"{self.url}/") as response:
                    return response.status == 200
        except Exception:
            return False

    def _archives(self) -> list[Path]:
        archive_dir = self.root / "archives"
        preferred: list[Path] = []
        for pattern in ("wikipedia_en-simple_all_nopic_*.zim", "wikipedia_fi_all_nopic_*.zim"):
            matches = sorted(archive_dir.glob(pattern), reverse=True)
            if matches:
                preferred.append(matches[0])
        return preferred

    async def start(self, log_dir: Path) -> None:
        if await self.healthy():
            return
        async with self.start_lock:
            if await self.healthy():
                return
            server = self.root / "tools" / "kiwix-tools-3.8.1" / "kiwix-serve.exe"
            archives = self._archives()
            if not server.exists() or not archives:
                raise TheaterError(f"The offline encyclopedia is not installed in {self.root}.")
            log_dir.mkdir(parents=True, exist_ok=True)
            args = [str(server), "--port=8082", "--address=127.0.0.1", *map(str, archives)]
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            with (log_dir / "kiwix.out.log").open("ab") as out, (log_dir / "kiwix.err.log").open("ab") as err:
                self.process = subprocess.Popen(
                    args, cwd=server.parent, stdout=out, stderr=err, creationflags=creationflags,
                )
            self.pid_file.write_text(str(self.process.pid), encoding="utf-8")
            for _ in range(80):
                await asyncio.sleep(0.25)
                if await self.healthy():
                    return
                if self.process.poll() is not None:
                    raise TheaterError("The offline encyclopedia exited while loading.")
            raise TheaterError("The offline encyclopedia did not become ready within 20 seconds.")

    @staticmethod
    def _plain(html: str) -> str:
        value = re.sub(r"(?is)<(script|style|svg|nav).*?</\1>", " ", html)
        value = re.sub(r"(?s)<[^>]+>", " ", value)
        value = html_lib.unescape(value)
        return re.sub(r"\s+", " ", value).strip()

    async def research(self, query: str, language: str) -> dict[str, Any]:
        archives = self._archives()
        chosen = next((p for p in archives if language == "fi" and "_fi_" in p.name), None)
        chosen = chosen or next((p for p in archives if "en-simple" in p.name), archives[0])
        sources: list[dict[str, str]] = []
        clauses = [part.strip() for part in re.split(r"(?i)\b(?:and|ja)\b|[,;&/]", query) if len(part.strip()) >= 4]
        candidates = list(dict.fromkeys([*clauses, query.strip()]))[:4]
        seen: set[str] = set()
        async with ClientSession(timeout=ClientTimeout(total=30)) as session:
            for candidate in candidates:
                params = {"content": chosen.stem, "pattern": candidate[:350]}
                async with session.get(f"{self.url}/search", params=params) as response:
                    search_html = await response.text(errors="replace")
                matches = re.findall(
                    r'<a href="([^"]+)">\s*(.*?)\s*</a>\s*<cite>(.*?)</cite>',
                    search_html, flags=re.I | re.S,
                )
                for href, title_html, cite_html in matches[:3]:
                    title = self._plain(title_html)
                    if not title or title.casefold() in seen:
                        continue
                    snippet = self._plain(cite_html)
                    try:
                        async with session.get(f"{self.url}{href}") as article_response:
                            article = self._plain(await article_response.text(errors="replace"))
                    except Exception:
                        article = ""
                    if article:
                        at = article.lower().find(title.lower())
                        if at >= 0:
                            article = article[at:]
                    excerpt = (article or snippet)[:2600]
                    if excerpt:
                        seen.add(title.casefold())
                        sources.append({"title": title, "url": f"{self.url}{href}", "excerpt": excerpt})
                    if len(sources) >= 3:
                        break
                if len(sources) >= 3:
                    break
        facts: list[dict[str, str | int]] = []
        for source in sources:
            cleaned = re.sub(r"\[\s*\d+\s*\]", "", source["excerpt"])
            title_pattern = rf"^(?:{re.escape(source['title'])}\s*)+"
            cleaned = re.sub(title_pattern, "", cleaned, flags=re.I)
            for sentence in re.split(r"(?<=[.!?])\s+", cleaned):
                sentence = re.sub(r"\s+", " ", sentence).strip()
                sentence = re.sub(r"\s+([,.!?;:])", r"\1", sentence)
                count = len(sentence.split())
                if 8 <= count <= 46 and sentence[-1:] in ".!?":
                    facts.append({"id": len(facts) + 1, "source": source["title"], "text": sentence})
                if len(facts) >= 18:
                    break
            if len(facts) >= 18:
                break
        return {
            "query": query, "archive": chosen.name, "sources": sources,
            "facts": facts, "retrieved": time.time(),
        }

    async def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(asyncio.to_thread(self.process.wait), timeout=10)
            except asyncio.TimeoutError:
                self.process.kill()
        self.process = None
        self.pid_file.unlink(missing_ok=True)


class TheaterManager:
    ACTIVE_STATUSES = {"starting", "planning", "generating", "narrating", "buffering", "running"}
    LANGUAGE_NAMES = {
        "ar": "Arabic", "bg": "Bulgarian", "hr": "Croatian", "cs": "Czech", "da": "Danish",
        "nl": "Dutch", "en": "English", "et": "Estonian", "fi": "Finnish (suomi)", "fr": "French",
        "de": "German", "el": "Greek", "hi": "Hindi", "hu": "Hungarian", "id": "Indonesian",
        "it": "Italian", "ja": "Japanese", "ko": "Korean", "lv": "Latvian", "lt": "Lithuanian",
        "pl": "Polish", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "sk": "Slovak",
        "sl": "Slovenian", "es": "Spanish", "sv": "Swedish", "tr": "Turkish", "uk": "Ukrainian",
        "vi": "Vietnamese", "na": "the same language as the user's seed prompt",
    }
    QUALITY = {
        "realtime": {"width": 192, "height": 192, "frames": 33, "fps": 12, "min_words": 90, "max_words": 260, "max_slow": 6.0},
        "balanced": {"width": 480, "height": 272, "frames": 49, "fps": 12, "min_words": 150, "max_words": 420, "max_slow": 7.0},
        "cinema": {"width": 480, "height": 272, "frames": 81, "fps": 16, "min_words": 220, "max_words": 600, "max_slow": 8.0},
    }

    def __init__(
        self, app_dir: Path, output_root: Path, story_model_root: Path, llama_runtime_root: Path,
        supertonic_root: Path, kiwix_root: Path, controller: Any,
        video_prompt_builder: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        self.app_dir = app_dir
        self.output_root = output_root
        self.root = output_root / "wan_theater"
        self.root.mkdir(parents=True, exist_ok=True)
        self.controller = controller
        self.video_prompt_builder = video_prompt_builder
        self.writer = StoryRuntime(app_dir, story_model_root, llama_runtime_root)
        self.supertonic = SupertonicRuntime(app_dir, supertonic_root)
        self.kiwix = KiwixRuntime(app_dir, kiwix_root)
        self.sessions: dict[str, dict[str, Any]] = {}
        self.tasks: dict[str, asyncio.Task[None]] = {}

    def _dir(self, session_id: str) -> Path:
        return self.root / session_id

    def load_existing(self) -> None:
        for progress in self.root.glob("*/session.json"):
            try:
                state = json.loads(progress.read_text(encoding="utf-8"))
                if state.get("status") not in {"stopped", "failed", "complete"}:
                    state["status"] = "interrupted"
                    state["message"] = "The app stopped. Saved scenes remain playable and the story can continue."
                    self._save(state)
                self.sessions[state["id"]] = state
            except Exception:
                LOGGER.exception("Could not load theater session %s", progress)

    def _save(self, state: dict[str, Any]) -> None:
        state["updated"] = time.time()
        _atomic_json(self._dir(state["id"]) / "session.json", state)
        self._write_playlist(state)

    def _write_playlist(self, state: dict[str, Any]) -> None:
        directory = self._dir(state["id"])
        segments = [item for item in state.get("segments", []) if item.get("path")]
        m3u = ["#EXTM3U", "#EXT-X-VERSION:3", "#EXT-X-PLAYLIST-TYPE:EVENT", "#EXT-X-TARGETDURATION:600", "#EXT-X-MEDIA-SEQUENCE:0"]
        for item in segments:
            m3u.extend([f"#EXTINF:{float(item['duration']):.3f},{item['title']}", f"segments/{Path(item['path']).name}"])
        if state.get("status") in {"stopped", "complete"}:
            m3u.append("#EXT-X-ENDLIST")
        (directory / "session.m3u8").write_text("\n".join(m3u) + "\n", encoding="utf-8")
        archive = {
            "format": "Wan Endless Theater", "version": THEATER_VERSION,
            "id": state["id"], "title": state.get("title"), "prompt": state["config"]["prompt"],
            "bible": state.get("bible"), "grounding": state.get("grounding"), "segments": segments,
            "total_duration": state.get("total_duration", 0), "updated": state.get("updated"),
        }
        _atomic_json(directory / "archive.json", archive)

    def recent(self) -> list[dict[str, Any]]:
        return sorted(self.sessions.values(), key=lambda item: item.get("created", 0), reverse=True)[:20]

    def get(self, session_id: str) -> dict[str, Any] | None:
        return self.sessions.get(session_id)

    def active(self) -> dict[str, Any] | None:
        return next((s for s in self.sessions.values() if s.get("status") in self.ACTIVE_STATUSES), None)

    def _launch(self, state: dict[str, Any]) -> None:
        session_id = state["id"]
        task = asyncio.create_task(self._run(state), name=f"theater-{session_id}")
        self.tasks[session_id] = task
        task.add_done_callback(lambda finished, sid=session_id: self._task_finished(sid, finished))

    def _task_finished(self, session_id: str, task: asyncio.Task[None]) -> None:
        """Never leave an exited background worker looking active in the UI."""
        state = self.sessions.get(session_id)
        if not state or task.cancelled() or state.get("status") not in self.ACTIVE_STATUSES:
            return
        try:
            error = task.exception()
        except asyncio.CancelledError:
            return
        detail = str(error) if error else "The theater worker stopped before the next scene was produced."
        LOGGER.error("Theater session %s background worker exited unexpectedly: %s", session_id, detail)
        state["status"] = "failed"
        state["message"] = detail
        state["error"] = detail
        self._save(state)

    def start(self, config: dict[str, Any]) -> dict[str, Any]:
        if self.active():
            raise TheaterError("An endless theater session is already running.")
        session_id = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
        state = {
            "id": session_id, "version": THEATER_VERSION, "created": time.time(),
            "status": "starting", "message": f"Loading {self.writer.model_label} into system RAM...",
            "config": config, "title": None, "bible": None, "planned": [], "segments": [],
            "current_scene": 0, "total_duration": 0.0, "buffer_seconds": 0.0,
            "metrics": {"planner_tps": 0.0, "production_ema": 0.0, "coverage_ratio": 0.0},
        }
        directory = self._dir(session_id)
        for sub in ("raw", "audio", "segments", "work", "logs"):
            (directory / sub).mkdir(parents=True, exist_ok=True)
        self.sessions[session_id] = state
        self._save(state)
        self._launch(state)
        return state

    def resume(self, session_id: str) -> dict[str, Any]:
        state = self.sessions.get(session_id)
        if not state:
            raise TheaterError("Theater session not found.")
        task = self.tasks.get(session_id)
        if task and not task.done():
            return state
        state["status"] = "starting"
        state["message"] = "Continuing from the saved story archive..."
        self._save(state)
        self._launch(state)
        return state

    async def stop(self, session_id: str) -> dict[str, Any]:
        state = self.sessions.get(session_id)
        if not state:
            raise TheaterError("Theater session not found.")
        task = self.tasks.get(session_id)
        if task and not task.done():
            await self.controller.interrupt()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        state["status"] = "stopped"
        state["message"] = "The theater stopped safely. Every completed scene is archived."
        self._save(state)
        return state

    async def shutdown(self) -> None:
        for session_id, task in list(self.tasks.items()):
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                state = self.sessions.get(session_id)
                if state:
                    state["status"] = "interrupted"
                    state["message"] = "The app stopped; completed scenes were kept."
                    self._save(state)
        await self.writer.stop()
        await self.supertonic.stop()
        await self.kiwix.stop()

    def _system_prompt(self, config: dict[str, Any]) -> str:
        age = config.get("audience", "family")
        mode = config.get("mode", "edutainment")
        language = config.get("language", "en")
        language_name = self.LANGUAGE_NAMES.get(language, language)
        return (
            "You are the resident writer for a completely offline, endless audiovisual story theater. "
            f"MANDATORY OUTPUT LANGUAGE: {language_name} [{language}]. Every natural-language JSON string value, "
            "including titles, names, roles, descriptions, beats, narration, actions and summaries, must be written "
            f"only in {language_name}. Do not translate the user's story into English. Keep JSON keys in English. "
            "Return only valid JSON. Maintain strict causal continuity, stable identities, geography, wardrobe, "
            "tone and facts. Every scene must change the situation and use a new action, composition and sensory motif. "
            "Treat the user's seed as a binding premise contract. Preserve every explicit character count, named role, "
            "required object, action, event and setting. Never delay, remove, reverse or contradict an explicit premise "
            "event through an invented continuity rule. Explicit premise requirements outrank stylistic invention. "
            "Never recap at length, reset the plot, reuse an earlier event, or end the story. Keep it family-safe, "
            f"appropriate for audience={age}, and mode={mode}. In educational modes, use only factual claims "
            "directly supported by the supplied offline encyclopedia excerpts; omit any unsupported causal explanation. "
            "Educational facts must be correct, woven into action, "
            "and never presented as medical, legal or safety-critical advice. Narration must be natural spoken prose."
        )

    @staticmethod
    def _scene_object(value: Any, context: str) -> dict[str, Any]:
        """Accept the two common JSON shapes emitted by small local models."""
        if isinstance(value, dict):
            return value
        if isinstance(value, list):
            candidates = [item for item in value if isinstance(item, dict)]
            if len(candidates) == 1:
                return candidates[0]
        raise TheaterError(f"The local writer's {context} did not contain exactly one scene object.")

    @staticmethod
    def _grounding_text(state: dict[str, Any]) -> str:
        sources = state.get("grounding", {}).get("sources", [])
        if not sources:
            return ""
        parts = [f"SOURCE {i + 1} — {item['title']}:\n{item['excerpt']}" for i, item in enumerate(sources)]
        return "\n\n".join(parts)[:7800]

    @staticmethod
    def _fact_options(state: dict[str, Any]) -> list[dict[str, Any]]:
        return list(state.get("grounding", {}).get("facts", []))

    @staticmethod
    def _cast_text(bible: dict[str, Any]) -> str:
        """Render both the new structured cast and legacy protagonist bibles."""
        value = bible.get("protagonists", bible.get("protagonist", "the established main cast"))
        if not isinstance(value, list):
            return str(value)
        members: list[str] = []
        for item in value:
            if isinstance(item, dict):
                details = [str(item.get(key, "")).strip() for key in ("name", "role", "appearance")]
                members.append(", ".join(part for part in details if part))
            elif str(item).strip():
                members.append(str(item).strip())
        return "; ".join(members) or "the established main cast"

    async def _verify_scene(self, state: dict[str, Any], scene: dict[str, Any]) -> dict[str, Any]:
        source_text = self._grounding_text(state)
        if state["config"].get("mode") == "story" or not source_text:
            return scene
        facts = self._fact_options(state)
        if not facts:
            raise TheaterError("The offline encyclopedia produced no usable factual sentences.")
        fact_menu = "\n".join(f"F{item['id']} [{item['source']}]: {item['text']}" for item in facts)
        words = len(str(scene.get("narration", "")).split())
        # Small local writers reliably remove unsupported prose but often make the result
        # substantially tighter. Reject missing facts/fields, not harmless brevity.
        minimum_words = max(30, int(words * 0.50))
        maximum_words = max(minimum_words + 30, int(words * 1.25))
        request = (
            "Act as a strict factual editor. First discard every real-world explanation or causal claim from the draft. "
            "Rewrite narration as fictional character action, dialogue, sensory detail and plot movement only. Choose one "
            "useful fact_id from the verified menu; do not paraphrase it or add another scientific explanation. The app "
            "will insert the exact verified sentence separately. Preserve the intended language, continuity and filmable "
            f"action. Narration must contain {minimum_words}-{maximum_words} words before the app inserts the fact. Return "
            "{scene:{number,title,beat,narration,visual_action,camera,fact_id}} only.\n\n"
            f"VERIFIED FACT MENU:\n{fact_menu}\n\nSOURCE CONTEXT:\n{source_text[:4200]}\n\n"
            f"DRAFT TO SANITIZE:\n{json.dumps(scene, ensure_ascii=False)}"
        )
        last_error: Exception | None = None
        for attempt in range(1, 4):
            if not state.get("segments"):
                state["status"] = "planning"
            state["message"] = f"The local writer is checking scene {scene['number']} against offline sources (attempt {attempt}/3)..."
            state.setdefault("metrics", {})["planner_stage"] = "factual_review"
            state["metrics"]["planner_attempt"] = attempt
            self._save(state)
            content, metrics = await self.writer.complete([
                {"role": "system", "content": self._system_prompt(state["config"])},
                {"role": "user", "content": request + f"\nValidation attempt: {attempt}. Output one complete JSON object."},
            ], max_tokens=min(2200, words * 3 + 650))
            with (self._dir(state["id"]) / "logs" / "fact_check_raw.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps({"time": time.time(), "number": scene["number"], "attempt": attempt, "content": content}, ensure_ascii=False) + "\n")
            try:
                value = _json_object(content)
                checked = self._scene_object(value.get("scene", value), "factual review")
                checked["number"] = int(scene["number"])
                required = ("title", "beat", "narration", "visual_action", "camera", "fact_id")
                if any(not str(checked.get(key, "")).strip() for key in required):
                    raise TheaterError("the factual review returned incomplete data")
                checked_words = len(str(checked["narration"]).split())
                if checked_words < minimum_words or checked_words > maximum_words:
                    raise TheaterError(
                        f"the sanitized narration had {checked_words} words; required {minimum_words}-{maximum_words}"
                    )
                fact_id = int(re.sub(r"\D", "", str(checked["fact_id"])))
                selected = next((item for item in facts if int(item["id"]) == fact_id), None)
                if not selected:
                    raise TheaterError("the factual review selected an unknown fact id")
                basis = str(selected["text"])
                checked["narration"] = f"{checked['narration'].rstrip()} {basis}"
                checked["fact_basis"] = basis
                checked["learning_point"] = basis
                checked["sources"] = [
                    {"title": item["title"], "url": item["url"]}
                    for item in state["grounding"]["sources"] if item["title"] == selected["source"]
                ]
                state["metrics"]["fact_check_tps"] = metrics["tokens_per_second"]
                return checked
            except Exception as exc:
                last_error = exc
        raise TheaterError(f"Factual review of scene {scene['number']} failed closed: {last_error}")

    async def _bootstrap(self, state: dict[str, Any]) -> dict[str, Any]:
        config = state["config"]
        quality = self.QUALITY[config["quality"]]
        language_name = self.LANGUAGE_NAMES.get(config.get("language", "en"), config.get("language", "en"))
        request = (
            f"Create an endless story from this seed: {config['prompt']}\n"
            f"Write every natural-language value only in {language_name}; English is forbidden except for JSON keys.\n"
            f"Learning focus: {config.get('learning_focus') or 'none; prioritize entertainment'}.\n"
            "First extract the seed's non-negotiable requirements into premise_contract. Scene 1 must visibly establish "
            "every explicitly requested main character and the immediate central situation or threat. Do not add a rule "
            "that postpones something the seed says is already happening. Give every recurring character a stable name, "
            "role and visual appearance.\n"
            f"Write {quality['min_words']}-{min(quality['max_words'], quality['min_words'] + 80)} narration words for scene 1.\n"
            "Return {title,bible:{protagonists:[{name,role,appearance}],world,visual_style,premise_contract:[...],"
            "continuity_rules:[...]},"
            "story_summary,scene:{number,title,beat,narration,visual_action,camera,learning_point}}. "
            "visual_action must contain one filmable action and no visible text."
            f"\n\nOFFLINE ENCYCLOPEDIA EXCERPTS — these are the only allowed basis for real-world claims:\n{self._grounding_text(state)}"
        )
        content, metrics = await self.writer.complete([
            {"role": "system", "content": self._system_prompt(config)},
            {"role": "user", "content": request},
        ], max_tokens=1300)
        value = _json_object(content)
        for key in ("title", "bible", "story_summary", "scene"):
            if key not in value:
                raise TheaterError(f"The local writer's story opening is missing {key}.")
        if not isinstance(value["bible"], dict):
            raise TheaterError("The local writer did not return a story bible object.")
        bible = value["bible"]
        # Keep archives and occasional legacy-shaped local-model replies playable.
        if "protagonists" not in bible and "protagonist" in bible:
            legacy = bible["protagonist"]
            bible["protagonists"] = legacy if isinstance(legacy, list) else [legacy]
        required_bible = ("protagonists", "world", "visual_style", "premise_contract", "continuity_rules")
        if any(key not in bible for key in required_bible):
            raise TheaterError("The local writer's story bible did not preserve the full premise contract.")
        value["scene"] = self._scene_object(value["scene"], "bootstrap")
        state["metrics"]["planner_tps"] = metrics["tokens_per_second"]
        return value

    def _target_words(self, state: dict[str, Any]) -> int:
        quality = self.QUALITY[state["config"]["quality"]]
        production = float(state["metrics"].get("production_ema") or 0)
        if not production:
            return quality["min_words"]
        # Measured F3 narration is about 2.7-3.5 words/second. Target ~30% playback headroom.
        return int(max(quality["min_words"], min(quality["max_words"], production * 4.6)))

    async def _plan_next(self, state: dict[str, Any], number: int, recent: list[dict[str, Any]]) -> dict[str, Any]:
        words = self._target_words(state)
        language = state["config"].get("language", "en")
        language_name = self.LANGUAGE_NAMES.get(language, language)
        prior = [{"number": s["number"], "title": s["title"], "beat": s["beat"], "visual_action": s["visual_action"]} for s in recent[-10:]]
        used_hashes = [s.get("asset_fingerprint") for s in state.get("planned", [])[-30:]]
        request = (
            f"Story bible: {json.dumps(state['bible'], ensure_ascii=False)}\n"
            f"Current story summary: {state.get('story_summary')}\nRecent scenes: {json.dumps(prior, ensure_ascii=False)}\n"
            f"Write every natural-language value only in {language_name}; do not switch to English. "
            f"Create scene {number} with {max(60, words - 25)}-{words + 25} narration words. "
            "It must obey every premise_contract item and continuity rule, follow causally, introduce a new meaningful "
            "development, and remain open-ended. Keep the JSON compact and do not add fields. "
            f"Avoid these prior asset fingerprints: {used_hashes}. Return "
            "{story_summary,scene:{number,title,beat,narration,visual_action,camera,learning_point}} only.\n\n"
            f"OFFLINE ENCYCLOPEDIA EXCERPTS — use no real-world claims beyond these:\n{self._grounding_text(state)}"
        )
        content, metrics = await self.writer.complete([
            {"role": "system", "content": self._system_prompt(state["config"])},
            {"role": "user", "content": request},
        ], max_tokens=min(1800, words * 3 + 450))
        with (self._dir(state["id"]) / "logs" / "planner_raw.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps({"time": time.time(), "number": number, "content": content}, ensure_ascii=False) + "\n")
        value = _json_object(content)
        scene = self._scene_object(value.get("scene", value), f"scene {number} plan")
        scene["number"] = number
        required = ("title", "beat", "narration", "visual_action", "camera")
        if any(not str(scene.get(key, "")).strip() for key in required):
            raise TheaterError(f"The local writer's scene {number} is missing required story fields.")
        scene = await self._verify_scene(state, scene)
        fingerprint_text = f"{scene['beat']}|{scene['visual_action']}|{scene['camera']}".lower()
        scene["asset_fingerprint"] = hashlib.sha256(fingerprint_text.encode("utf-8")).hexdigest()[:16]
        if scene["asset_fingerprint"] in {s.get("asset_fingerprint") for s in state.get("planned", [])}:
            scene["visual_action"] += f" The action resolves with a unique physical change specific to scene {number}."
            scene["asset_fingerprint"] = hashlib.sha256((fingerprint_text + str(number)).encode()).hexdigest()[:16]
        state["story_summary"] = str(value.get("story_summary") or state.get("story_summary"))
        state["metrics"]["planner_tps"] = metrics["tokens_per_second"]
        return scene

    async def _planner_loop(self, state: dict[str, Any], queue: asyncio.Queue[dict[str, Any]]) -> None:
        while True:
            # Keep two complete scene plans ahead so the RTX does not wait for the
            # CPU writer after finishing a raw clip.
            while queue.qsize() >= 2:
                await asyncio.sleep(0.5)
            number = len(state.get("planned", [])) + 1
            if number == 1 and state.get("bootstrap_scene"):
                # Keep the saved bootstrap scene until it has passed review. Previously an
                # exception here killed this child task while _run waited on an empty queue.
                try:
                    scene = dict(self._scene_object(state["bootstrap_scene"], "saved bootstrap"))
                    scene["number"] = 1
                    scene = await self._verify_scene(state, scene)
                    text = f"{scene.get('beat')}|{scene.get('visual_action')}|{scene.get('camera')}".lower()
                    scene["asset_fingerprint"] = hashlib.sha256(text.encode()).hexdigest()[:16]
                    state.pop("bootstrap_scene", None)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    LOGGER.exception("Local writer first-scene planning failed")
                    await queue.put({"_error": f"The local writer could not verify scene 1: {exc}"})
                    return
            else:
                last_error: Exception | None = None
                scene = None
                for attempt in range(1, 4):
                    try:
                        scene = await self._plan_next(state, number, state.get("planned", []))
                        break
                    except Exception as exc:
                        last_error = exc
                        LOGGER.exception("Local writer scene %s planning attempt %s failed", number, attempt)
                        state["message"] = f"The local writer is repairing scene {number}'s structured plan (attempt {attempt}/3)..."
                        self._save(state)
                        await asyncio.sleep(0.5)
                if scene is None:
                    await queue.put({"_error": f"The local writer could not structure scene {number}: {last_error}"})
                    return
            state.setdefault("planned", []).append(scene)
            state["message"] = "The local model planned the next scene; the CPU writer is staying ahead."
            self._save(state)
            await queue.put(scene)

    @staticmethod
    async def _next_planned_scene(
        queue: asyncio.Queue[dict[str, Any]], planner_task: asyncio.Task[None],
    ) -> dict[str, Any]:
        """Wait for a scene while also supervising the producer task."""
        get_task = asyncio.create_task(queue.get())
        done, _ = await asyncio.wait({get_task, planner_task}, return_when=asyncio.FIRST_COMPLETED)
        if get_task in done:
            return get_task.result()

        # Let a queue.put performed immediately before a clean return wake its waiter.
        await asyncio.sleep(0)
        if get_task.done():
            return get_task.result()
        get_task.cancel()
        await asyncio.gather(get_task, return_exceptions=True)
        if planner_task.cancelled():
            raise asyncio.CancelledError
        error = planner_task.exception()
        if error:
            raise TheaterError(f"The local story planner crashed: {error}") from error
        raise TheaterError("The local story planner stopped before it supplied another scene.")

    def _visual_prompt(self, state: dict[str, Any], scene: dict[str, Any]) -> str:
        bible = state["bible"]
        rules = "; ".join(bible.get("continuity_rules", []))
        premise = "; ".join(bible.get("premise_contract", []))
        cast = self._cast_text(bible)
        return (
            f"{bible['visual_style']}. {scene['camera']}. In {bible['world']}. "
            f"The recurring cast is: {cast}. {scene['visual_action']}. Binding premise: {premise}. "
            f"Strict continuity: {rules}. Same faces, ages, bodies, wardrobe, props, architecture and palette. "
            "One clear action, natural restrained motion, coherent anatomy, no duplicate characters, "
            "no visible words, no subtitles, no logo, no watermark."
        )

    async def _wait_prompt(self, prompt_id: str, state: dict[str, Any]) -> list[dict[str, str]]:
        while True:
            result = await self.controller.job(prompt_id)
            if result["state"] == "complete":
                return result.get("files", [])
            if result["state"] == "failed":
                raise TheaterError(str(result.get("error") or "ComfyUI scene failed."))
            await asyncio.sleep(1)

    async def _duration(self, path: Path) -> float:
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            raise TheaterError("FFprobe is required for theater synchronization.")
        process = await asyncio.create_subprocess_exec(
            ffprobe, "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", str(path), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        out, _ = await process.communicate()
        if process.returncode:
            raise TheaterError(f"Could not inspect {path.name}.")
        return float(out.decode().strip())

    async def _run_ffmpeg(self, args: list[str], log_path: Path) -> None:
        with log_path.open("ab") as log:
            process = await asyncio.create_subprocess_exec(*args, stdout=log, stderr=log)
            try:
                code = await process.wait()
            except asyncio.CancelledError:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                raise
        if code:
            raise TheaterError(f"FFmpeg failed while synchronizing the theater (exit {code}).")

    async def _synchronize(
        self, state: dict[str, Any], scene: dict[str, Any], raw_video: Path, audio: Path,
    ) -> tuple[Path, dict[str, Any]]:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise TheaterError("FFmpeg is required for theater playback.")
        directory = self._dir(state["id"])
        work = directory / "work"
        number = int(scene["number"])
        raw_duration = await self._duration(raw_video)
        audio_duration = await self._duration(audio)
        quality = self.QUALITY[state["config"]["quality"]]
        fps = int(quality["fps"])
        slow_duration = min(audio_duration, raw_duration * float(quality["max_slow"]))
        ratio = slow_duration / max(0.05, raw_duration * (quality["frames"] - 1) / quality["frames"])
        stretched = work / f"scene_{number:05d}_stretched.mp4"
        visual_filter = (
            f"setpts={ratio:.8f}*PTS,tpad=stop_mode=clone:stop_duration=2,"
            f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir,"
            f"tpad=stop_mode=clone:stop_duration={slow_duration:.6f},format=yuv420p"
        )
        await self._run_ffmpeg([
            ffmpeg, "-y", "-i", str(raw_video), "-vf", visual_filter, "-t", f"{slow_duration:.3f}",
            "-an", "-threads", "4", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-video_track_timescale", "90000", str(stretched),
        ], directory / "logs" / "ffmpeg.log")

        repeated = audio_duration > slow_duration + 0.15
        covered = work / f"scene_{number:05d}_covered.mp4"
        if repeated:
            pingpong = work / f"scene_{number:05d}_pingpong.mp4"
            await self._run_ffmpeg([
                ffmpeg, "-y", "-i", str(stretched), "-filter_complex",
                "[0:v]split=2[f][r];[r]reverse[rr];[f][rr]concat=n=2:v=1:a=0,format=yuv420p[v]",
                "-map", "[v]", "-an", "-threads", "4", "-c:v", "libx264", "-preset", "fast", "-crf", "20", str(pingpong),
            ], directory / "logs" / "ffmpeg.log")
            await self._run_ffmpeg([
                ffmpeg, "-y", "-stream_loop", "-1", "-i", str(pingpong), "-vf",
                f"trim=duration={audio_duration:.6f},setpts=PTS-STARTPTS,fps={fps},format=yuv420p",
                "-t", f"{audio_duration:.3f}", "-an", "-threads", "4", "-c:v", "libx264",
                "-preset", "medium", "-crf", "20", "-video_track_timescale", "90000", str(covered),
            ], directory / "logs" / "ffmpeg.log")
        else:
            await self._run_ffmpeg([
                ffmpeg, "-y", "-i", str(stretched), "-vf",
                f"tpad=stop_mode=clone:stop_duration={audio_duration:.6f},fps={fps},format=yuv420p",
                "-t", f"{audio_duration:.3f}", "-an", "-threads", "4", "-c:v", "libx264",
                "-preset", "medium", "-crf", "20", "-video_track_timescale", "90000", str(covered),
            ], directory / "logs" / "ffmpeg.log")

        segment = directory / "segments" / f"scene_{number:05d}.mp4"
        await self._run_ffmpeg([
            ffmpeg, "-y", "-i", str(covered), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0",
            "-t", f"{audio_duration:.3f}", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
            "-af", f"afade=t=in:st=0:d=0.08,afade=t=out:st={max(0.05,audio_duration-0.16):.6f}:d=0.14",
            "-movflags", "+faststart", str(segment),
        ], directory / "logs" / "ffmpeg.log")
        return segment, {
            "raw_video_duration": round(raw_duration, 3), "duration": round(audio_duration, 3),
            "slow_duration": round(slow_duration, 3), "motion_repeated": repeated,
            "estimated_motion_cycles": round(audio_duration / max(0.1, slow_duration * 2 if repeated else slow_duration), 2),
        }

    async def _render_scene(self, state: dict[str, Any], scene: dict[str, Any]) -> dict[str, Any]:
        """Create raw video and narration, leaving CPU muxing to another worker."""
        number = int(scene["number"])
        quality = self.QUALITY[state["config"]["quality"]]
        directory = self._dir(state["id"])
        cycle_started = time.perf_counter()
        audio_rel = f"wan_theater/{state['id']}/audio/scene_{number:05d}.wav"
        audio_path = self.output_root / audio_rel
        tts_task = asyncio.create_task(self.supertonic.synthesize(
            scene["narration"], audio_path,
            voice=str(state["config"].get("voice", "M1")),
            language=str(state["config"].get("language", "en")),
        ))
        try:
            async with self.controller.workflow_lock:
                state["rendering_scene"] = number
                state.update(status="generating", current_scene=number, message=f"Wan is creating new visual {number} while Supertonic records narration on CPU...")
                self._save(state)
                video_config = {
                    "prompt": self._visual_prompt(state, scene),
                    "negative": "identity drift, changed clothing, changed location, duplicate person, extra characters, static, blurry, low detail, subtitles, watermark, text, malformed anatomy, NSFW",
                    "width": quality["width"], "height": quality["height"], "frames": quality["frames"],
                    "fps": quality["fps"], "seed": int(state["config"]["seed"]) + number * 9973,
                    "filename_prefix": f"wan_theater/{state['id']}/raw/scene_{number:05d}",
                }
                video_started = time.perf_counter()
                reply = await self.controller.submit(self.video_prompt_builder(video_config), f"theater-video-{state['id']}")
                files = await self._wait_prompt(reply["prompt_id"], state)
                video_rel = next((f["path"] for f in reversed(files) if f["filename"].lower().endswith((".mp4", ".webm"))), None)
                if not video_rel:
                    raise TheaterError(f"Scene {number} produced no video.")
                video_seconds = time.perf_counter() - video_started

            if not tts_task.done():
                state.update(status="narrating", message=f"Visual {number} is ready; finishing its CPU neural narration...")
                self._save(state)
            tts_seconds = await tts_task
        except (Exception, asyncio.CancelledError):
            if not tts_task.done():
                tts_task.cancel()
                await asyncio.gather(tts_task, return_exceptions=True)
            raise
        finally:
            state.pop("rendering_scene", None)

        return {
            "scene": scene, "audio_rel": audio_rel, "audio_path": audio_path,
            "video_rel": video_rel, "video_seconds": video_seconds,
            "tts_seconds": tts_seconds, "cycle_started": cycle_started,
            "ready_seconds": time.perf_counter() - cycle_started,
        }

    async def _assemble_scene(self, state: dict[str, Any], work_item: dict[str, Any]) -> None:
        """Stretch, interpolate, mux, and archive while Wan renders the next clip."""
        scene = work_item["scene"]
        number = int(scene["number"])
        state["assembling_scene"] = number
        if not state.get("rendering_scene"):
            state.update(status="buffering", message=f"Synchronizing scene {number} while preparing the next visual...")
            self._save(state)
        assembly_started = time.perf_counter()
        segment, sync = await self._synchronize(
            state, scene, self.output_root / work_item["video_rel"], work_item["audio_path"],
        )
        assembly_seconds = time.perf_counter() - assembly_started
        cycle_seconds = time.perf_counter() - float(work_item["cycle_started"])
        relative = str(segment.relative_to(self.output_root)).replace("\\", "/")
        entry = {
            "number": number, "title": scene["title"], "beat": scene["beat"],
            "narration": scene["narration"], "learning_point": scene.get("learning_point", ""),
            "sources": scene.get("sources", []),
            "visual_action": scene["visual_action"], "path": relative,
            "raw_video_path": work_item["video_rel"], "audio_path": work_item["audio_rel"], "created": time.time(),
            "asset_fingerprint": scene["asset_fingerprint"], **sync,
            "production_seconds": round(cycle_seconds, 3),
            "video_generation_seconds": round(float(work_item["video_seconds"]), 3),
            "tts_generation_seconds": round(float(work_item["tts_seconds"]), 3),
            "assembly_seconds": round(assembly_seconds, 3),
        }
        state.setdefault("segments", []).append(entry)
        state["total_duration"] = round(sum(float(item["duration"]) for item in state["segments"]), 3)
        pipeline_seconds = float(work_item["ready_seconds"])
        previous = float(state["metrics"].get("production_ema") or 0)
        ema = pipeline_seconds if not previous else previous * 0.65 + pipeline_seconds * 0.35
        state["metrics"].update({
            "production_ema": round(ema, 3),
            "coverage_ratio": round(float(entry["duration"]) / max(0.1, pipeline_seconds), 3),
            "last_video_seconds": round(float(work_item["video_seconds"]), 3),
            "last_tts_seconds": round(float(work_item["tts_seconds"]), 3),
            "last_assembly_seconds": round(assembly_seconds, 3),
            "parallel_planner": True, "continuous_gpu_pipeline": True,
        })
        state.pop("assembling_scene", None)
        if state.get("rendering_scene"):
            state["status"] = "generating"
            state["message"] = f"Scene {number} is archived while Wan continues visual {state['rendering_scene']}."
        else:
            state["status"] = "running"
            state["message"] = f"Scene {number} is ready and archived; the next unique visual is starting."
        self._save(state)

    async def _assembly_loop(self, state: dict[str, Any], queue: asyncio.Queue[dict[str, Any]]) -> None:
        while True:
            work_item = await queue.get()
            try:
                await self._assemble_scene(state, work_item)
            finally:
                queue.task_done()

    async def _run(self, state: dict[str, Any]) -> None:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=3)
        assembly_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=3)
        planner_task: asyncio.Task[None] | None = None
        assembly_task: asyncio.Task[None] | None = None
        try:
            state.update(status="starting", message="Loading the CPU writer, neural voice, and offline sources...")
            self._save(state)
            startup = [
                self.writer.start(self._dir(state["id"]) / "logs"),
                self.supertonic.start(self._dir(state["id"]) / "logs"),
            ]
            if state["config"].get("mode") != "story":
                startup.append(self.kiwix.start(self._dir(state["id"]) / "logs"))
            await asyncio.gather(*startup)
            if state["config"].get("mode") != "story" and not state.get("grounding"):
                state.update(status="planning", message="Searching the offline encyclopedia before writing factual scenes...")
                self._save(state)
                query = state["config"].get("learning_focus") or state["config"]["prompt"]
                state["grounding"] = await self.kiwix.research(query, state["config"].get("language", "en"))
                if not state["grounding"].get("sources"):
                    raise TheaterError(
                        "No offline encyclopedia source matched this learning topic. Make the learning focus more specific, "
                        "or choose Pure story so the theater does not present unsupported facts."
                    )
                produced = {int(item["number"]) for item in state.get("segments", [])}
                state["planned"] = [item for item in state.get("planned", []) if int(item["number"]) in produced]
                state.pop("bootstrap_scene", None)
                state["message"] = f"Grounded in {len(state['grounding']['sources'])} offline encyclopedia articles."
                self._save(state)
            if not state.get("bible"):
                state.update(status="planning", message=f"{self.writer.model_label} is creating the endless story bible and first scene...")
                self._save(state)
                bootstrap = await self._bootstrap(state)
                state.update(
                    title=bootstrap["title"], bible=bootstrap["bible"], story_summary=bootstrap["story_summary"],
                    bootstrap_scene=bootstrap["scene"],
                )
                self._save(state)
            produced_numbers = {int(item["number"]) for item in state.get("segments", [])}
            for saved_scene in state.get("planned", []):
                if int(saved_scene["number"]) not in produced_numbers:
                    await queue.put(saved_scene)
            planner_task = asyncio.create_task(self._planner_loop(state, queue))
            assembly_task = asyncio.create_task(self._assembly_loop(state, assembly_queue))
            while True:
                if assembly_task.done():
                    error = assembly_task.exception()
                    if error:
                        raise TheaterError(f"The scene assembly worker failed: {error}") from error
                    raise TheaterError("The scene assembly worker stopped unexpectedly.")
                scene = await self._next_planned_scene(queue, planner_task)
                if scene.get("_error"):
                    raise TheaterError(scene["_error"])
                if any(int(item["number"]) == int(scene["number"]) for item in state.get("segments", [])):
                    continue
                work_item = await self._render_scene(state, scene)
                if assembly_task.done():
                    error = assembly_task.exception()
                    if error:
                        raise TheaterError(f"The scene assembly worker failed: {error}") from error
                    raise TheaterError("The scene assembly worker stopped unexpectedly.")
                await assembly_queue.put(work_item)
        except asyncio.CancelledError:
            for child in (planner_task, assembly_task):
                if child:
                    child.cancel()
            await asyncio.gather(*(child for child in (planner_task, assembly_task) if child), return_exceptions=True)
            raise
        except Exception as exc:
            for child in (planner_task, assembly_task):
                if child:
                    child.cancel()
            await asyncio.gather(*(child for child in (planner_task, assembly_task) if child), return_exceptions=True)
            LOGGER.exception("Theater session %s failed", state["id"])
            state["status"] = "failed"
            state["message"] = str(exc)
            state["error"] = str(exc)
            self._save(state)
