from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import os
import re
import secrets
import shutil
import signal
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from aiohttp import ClientSession, ClientTimeout

from process_utils import terminate_process_tree


LOGGER = logging.getLogger("wan-video-ui.movie")
PROJECT_VERSION = 2


class MovieError(RuntimeError):
    pass


def _slug(value: str, fallback: str = "movie") -> str:
    clean = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return clean[:55] or fallback


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


class LocalPlanner:
    def __init__(self, app_dir: Path, bonsai_root: Path) -> None:
        self.app_dir = app_dir
        self.root = bonsai_root
        self.process: subprocess.Popen[bytes] | None = None
        self.pid_file = app_dir / "movie-planner.pid"
        self.url = "http://127.0.0.1:8082"

    async def _healthy(self) -> bool:
        try:
            async with ClientSession(timeout=ClientTimeout(total=2)) as session:
                async with session.get(f"{self.url}/health") as response:
                    return response.status == 200 and (await response.json()).get("status") == "ok"
        except Exception:
            return False

    async def start(self, log_dir: Path) -> None:
        if await self._healthy():
            return
        server = self.root / "runtime" / "llama-server.exe"
        model = self.root / "models" / "Ternary-Bonsai-27B-Q2_0.gguf"
        if not server.exists() or not model.exists():
            raise MovieError(f"The local Bonsai planner files are missing from {self.root}.")
        log_dir.mkdir(parents=True, exist_ok=True)
        args = [
            str(server), "-m", str(model), "--alias", "bonsai-movie-planner",
            "--host", "127.0.0.1", "--port", "8082", "-ngl", "all",
            "--fit", "off", "-c", "8192", "-np", "1", "-ctk", "q4_0",
            "-ctv", "q4_0", "-fa", "on", "--kv-offload", "--op-offload",
            "--cache-ram", "0", "--jinja", "--temp", "0.2", "--top-p",
            "0.9", "--top-k", "20", "--no-warmup",
        ]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        with (log_dir / "planner.out.log").open("ab") as stdout, (log_dir / "planner.err.log").open("ab") as stderr:
            self.process = subprocess.Popen(
                args, cwd=server.parent, stdout=stdout, stderr=stderr,
                creationflags=creationflags,
            )
        self.pid_file.write_text(str(self.process.pid), encoding="utf-8")
        for _ in range(180):
            if await self._healthy():
                return
            if self.process.poll() is not None:
                raise MovieError("Bonsai exited while loading. See this movie's planner.err.log.")
            await asyncio.sleep(0.5)
        raise MovieError("Bonsai did not become ready within 90 seconds.")

    async def stop(self) -> None:
        if self.process and self.process.poll() is None:
            await terminate_process_tree(self.process, timeout=12)
        elif self.pid_file.exists():
            try:
                os.kill(int(self.pid_file.read_text(encoding="utf-8").strip()), signal.SIGTERM)
                for _ in range(20):
                    if not await self._healthy():
                        break
                    await asyncio.sleep(0.25)
            except (OSError, ValueError):
                pass
        self.process = None
        self.pid_file.unlink(missing_ok=True)

    async def plan(self, sentence: str, shot_count: int) -> dict[str, Any]:
        shot_schema = {
            "type": "object", "additionalProperties": False,
            "required": ["number", "act", "beat", "action", "camera", "narration", "transition"],
            "properties": {
                "number": {"type": "integer"},
                "act": {"type": "integer", "minimum": 1, "maximum": 3},
                "beat": {"type": "string"},
                "action": {"type": "string"},
                "camera": {"type": "string"},
                "narration": {"type": "string"},
                "transition": {"type": "string", "enum": ["cut", "dissolve"]},
            },
        }
        schema = {
            "type": "object", "additionalProperties": False,
            "required": ["title", "logline", "character", "location", "style", "continuity_rules", "shots"],
            "properties": {
                "title": {"type": "string"}, "logline": {"type": "string"},
                "character": {"type": "string"}, "location": {"type": "string"},
                "style": {"type": "string"},
                "continuity_rules": {"type": "array", "minItems": 3, "maxItems": 7, "items": {"type": "string"}},
                "shots": {"type": "array", "minItems": shot_count, "maxItems": shot_count, "items": shot_schema},
            },
        }
        system = (
            "You are the continuity editor for a fully offline narration-led micro-film. Return only schema-valid JSON. "
            "Use one recurring protagonist and one stable main location. The exact character identity, face, hair, age, "
            "wardrobe, location, weather, palette, and visual medium must never change. Build a causal three-act story: "
            "setup, escalating complication, decisive action, and resolved ending. Every shot gets one visually simple "
            "action and one conventional camera setup. Prefer narration; no visible dialogue, readable text, crowds, "
            "complex hand work, costume changes, flashbacks, or unexplained objects. Do not repeat a story beat."
        )
        example = {
            "title": "The Last Bloom",
            "logline": "A rusted gardener robot spends its final charge helping one flower greet the sun.",
            "character": "MOSS, a small dented brass garden robot with round glass eyes, moss on the left shoulder, and a red scarf",
            "location": "an abandoned glass greenhouse with cracked panes and waist-high weeds at pale dawn",
            "style": "cinematic storybook realism, rust orange, sage green and pale dawn blue, soft natural light, restrained motion, no text",
            "continuity_rules": [
                "MOSS always has the same red scarf and mossy left shoulder",
                "the greenhouse geography and dawn lighting remain stable",
                "the only flower is white",
            ],
            "shots": [
                {"number": 1, "act": 1, "beat": "Moss discovers the flower", "action": "Moss crosses the weeds toward one closed white flower", "camera": "wide locked establishing shot", "narration": "At the end of its charge, Moss found one living thing still waiting for morning.", "transition": "cut"},
                {"number": 2, "act": 2, "beat": "Moss chooses sacrifice", "action": "Moss kneels beside the flower as a soft glow passes from its chest into the soil", "camera": "medium side shot", "narration": "It gave the earth the little warmth it had left.", "transition": "dissolve"},
                {"number": 3, "act": 3, "beat": "The sacrifice succeeds", "action": "Sunlight enters and the white flower opens beside motionless Moss", "camera": "wide locked resolution shot", "narration": "When morning came, the garden remembered how to begin.", "transition": "dissolve"},
            ],
        }
        # The example establishes the contract; its shot list is trimmed or repeated only in the prompt,
        # while the schema enforces the requested output length.
        body = {
            "model": "bonsai-movie-planner", "temperature": 0.12,
            "max_tokens": min(7000, 650 + shot_count * 135),
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {"name": "movie_plan", "strict": True, "schema": schema}},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": "Create exactly 3 shots from: An old robot grows one last flower before its battery dies."},
                {"role": "assistant", "content": json.dumps(example, ensure_ascii=False)},
                {"role": "user", "content": f"Create exactly {shot_count} shots from: {sentence}"},
            ],
        }
        async with ClientSession(timeout=ClientTimeout(total=600)) as session:
            async with session.post(f"{self.url}/v1/chat/completions", json=body) as response:
                raw = await response.text()
                if response.status != 200:
                    raise MovieError(f"Bonsai planning failed: {raw[:500]}")
                reply = json.loads(raw)
        content = reply["choices"][0]["message"].get("content")
        if not content:
            raise MovieError("Bonsai returned an empty movie plan.")
        plan = json.loads(content)
        if len(plan.get("shots", [])) != shot_count:
            raise MovieError(f"Bonsai returned {len(plan.get('shots', []))} shots instead of {shot_count}.")
        for index, shot in enumerate(plan["shots"], start=1):
            shot["number"] = index
        return plan


class MovieManager:
    def __init__(
        self,
        app_dir: Path,
        output_root: Path,
        bonsai_root: Path,
        controller: Any,
        video_prompt_builder: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        self.app_dir = app_dir
        self.output_root = output_root
        self.movies_root = output_root / "wan_movies"
        self.movies_root.mkdir(parents=True, exist_ok=True)
        self.controller = controller
        self.video_prompt_builder = video_prompt_builder
        self.planner = LocalPlanner(app_dir, bonsai_root)
        self.jobs: dict[str, dict[str, Any]] = {}
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self.run_lock = asyncio.Lock()

    def load_existing(self) -> None:
        for progress in self.movies_root.glob("*/progress.json"):
            try:
                state = json.loads(progress.read_text(encoding="utf-8"))
                if state.get("status") in {"planning", "rendering", "narrating", "assembling"}:
                    state["status"] = "interrupted"
                    state["message"] = "The app stopped before this movie finished. It can be resumed."
                    _atomic_json(progress, state)
                state.setdefault("project_version", 1)
                self._ensure_editor_state(state)
                self.jobs[state["id"]] = state
            except Exception:
                LOGGER.exception("Could not load movie state from %s", progress)

    def _job_dir(self, job_id: str) -> Path:
        return self.movies_root / job_id

    def _save(self, state: dict[str, Any]) -> None:
        state["updated"] = time.time()
        self.jobs[state["id"]] = state
        _atomic_json(self._job_dir(state["id"]) / "progress.json", state)
        self._write_project(state)

    def _update(self, state: dict[str, Any], **changes: Any) -> None:
        state.update(changes)
        self._save(state)

    def _default_edit(self, state: dict[str, Any], number: int) -> dict[str, Any]:
        config = state["config"]
        return {
            "number": number, "enabled": True, "order": number,
            "sync_mode": config.get("sync_mode", "fit_video_to_audio"),
            "fps": int(config.get("fps", 16)),
            "fill_mode": config.get("fill_mode", "freeze"),
            "audio_mode": "pad", "target_duration": 0.0,
            "video_in": 0.0, "video_out": 0.0,
            "audio_in": 0.0, "audio_out": 0.0,
            "narration_gain": 1.0,
            "motion_interpolation": bool(config.get("motion_interpolation", True)),
        }

    def _ensure_editor_state(self, state: dict[str, Any]) -> None:
        plan = state.get("plan")
        if not plan:
            return
        count = len(plan.get("shots", []))
        existing_edits = {int(edit.get("number", 0)): edit for edit in state.get("edits", [])}
        edits = []
        clips = list(state.get("clips", []))
        clip_map = {int(clip.get("number", 0)): clip for clip in clips}
        shots = state.get("shot_files", [])
        audio = state.get("audio_files", [])
        new_clips = []
        for number in range(1, count + 1):
            edit = self._default_edit(state, number)
            edit.update(existing_edits.get(number, {}))
            edits.append(edit)
            clip = {"number": number, "status": "planned", "video_path": None, "audio_path": None, "motion": None}
            clip.update(clip_map.get(number, {}))
            if number <= len(shots) and shots[number - 1]:
                clip["video_path"] = shots[number - 1]
            if number <= len(audio) and audio[number - 1]:
                clip["audio_path"] = audio[number - 1]
            if clip["video_path"] and clip["audio_path"]:
                clip["status"] = "ready"
            elif clip["video_path"]:
                clip["status"] = "video_ready"
            elif clip["audio_path"]:
                clip["status"] = "audio_ready"
            new_clips.append(clip)
        state["project_version"] = PROJECT_VERSION
        state["edits"] = edits
        state["clips"] = new_clips

    def _write_project(self, state: dict[str, Any]) -> None:
        if not state.get("plan"):
            return
        project = {
            "format": "Wan Video Studio Project", "version": PROJECT_VERSION,
            "id": state["id"], "title": state.get("title"), "logline": state.get("logline"),
            "created": state.get("created"), "updated": state.get("updated"),
            "config": state.get("config"), "plan": state.get("plan"),
            "clips": state.get("clips", []), "edits": state.get("edits", []),
            "final_path": state.get("final_path"), "duration": state.get("duration"),
        }
        _atomic_json(self._job_dir(state["id"]) / "project.json", project)

    def get(self, job_id: str) -> dict[str, Any] | None:
        return self.jobs.get(job_id)

    def recent(self) -> list[dict[str, Any]]:
        return sorted(self.jobs.values(), key=lambda item: item.get("created", 0), reverse=True)[:20]

    def start(self, config: dict[str, Any]) -> dict[str, Any]:
        job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
        state = {
            "id": job_id, "status": "queued", "progress": 0,
            "message": "Waiting to plan the story…", "created": time.time(),
            "project_version": PROJECT_VERSION, "config": config, "shot_files": [], "audio_files": [],
            "clips": [], "edits": [],
            "completed_shots": 0, "completed_audio": 0, "final_path": None,
        }
        job_dir = self._job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(job_dir / "request.json", config)
        self._save(state)
        self.tasks[job_id] = asyncio.create_task(self._run(state))
        return state

    def save_edits(self, job_id: str, raw_edits: list[dict[str, Any]]) -> dict[str, Any]:
        state = self.jobs.get(job_id)
        if not state or not state.get("plan"):
            raise MovieError("Movie project not found or not planned yet.")
        if not isinstance(raw_edits, list) or len(raw_edits) != len(state["plan"]["shots"]):
            raise ValueError("The edit list must contain one entry for every planned shot.")
        allowed_sync = {"fit_video_to_audio", "fit_audio_to_video", "fixed_fps_fill"}
        allowed_fill = {"freeze", "loop", "pingpong", "black"}
        allowed_audio = {"pad", "trim", "fit"}
        normalized = []
        seen: set[int] = set()
        for raw in raw_edits:
            number = int(raw.get("number", 0))
            if number < 1 or number > len(raw_edits) or number in seen:
                raise ValueError("Each edit needs a unique valid shot number.")
            seen.add(number)
            base = self._default_edit(state, number)
            base.update(raw)
            base["number"] = number
            base["order"] = int(base.get("order", number))
            base["enabled"] = bool(base.get("enabled", True))
            base["fps"] = max(1, min(60, int(base.get("fps", state["config"]["fps"]))))
            base["target_duration"] = max(0.0, min(600.0, float(base.get("target_duration", 0))))
            base["narration_gain"] = max(0.0, min(4.0, float(base.get("narration_gain", 1))))
            base["sync_mode"] = str(base.get("sync_mode"))
            base["fill_mode"] = str(base.get("fill_mode"))
            base["audio_mode"] = str(base.get("audio_mode"))
            if base["sync_mode"] not in allowed_sync or base["fill_mode"] not in allowed_fill or base["audio_mode"] not in allowed_audio:
                raise ValueError(f"Shot {number} contains an unsupported edit policy.")
            normalized.append(base)
        state["edits"] = sorted(normalized, key=lambda edit: (edit["order"], edit["number"]))
        self._save(state)
        return state

    def render_edit(self, job_id: str) -> dict[str, Any]:
        state = self.jobs.get(job_id)
        if not state or not state.get("plan"):
            raise MovieError("Movie project not found or not planned yet.")
        if not state.get("shot_files"):
            raise MovieError("No generated video assets are available yet.")
        task = self.tasks.get(job_id)
        if task and not task.done():
            raise MovieError("This movie is still generating or rebuilding.")
        self.tasks[job_id] = asyncio.create_task(self._render_existing(state))
        return state

    async def _render_existing(self, state: dict[str, Any]) -> None:
        async with self.controller.workflow_lock:
            try:
                self._update(state, status="assembling", progress=90, message="Rebuilding the edited movie from existing assets…")
                await self._assemble(state, state["plan"], self._job_dir(state["id"]))
            except Exception as exc:
                LOGGER.exception("Edited movie %s failed", state["id"])
                self._update(state, status="failed", message=str(exc), error=str(exc))

    def resume(self, job_id: str) -> dict[str, Any]:
        state = self.jobs.get(job_id)
        if not state:
            raise MovieError("Movie job not found.")
        if job_id in self.tasks and not self.tasks[job_id].done():
            return state
        if state.get("status") == "complete":
            return state
        self.tasks[job_id] = asyncio.create_task(self._run(state))
        return state

    async def cancel(self, job_id: str) -> None:
        task = self.tasks.get(job_id)
        if task and not task.done():
            await self.controller.interrupt()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def shutdown(self) -> None:
        """Stop owned work while keeping every completed project artifact resumable."""
        active: list[tuple[str, asyncio.Task[None]]] = [
            (job_id, task) for job_id, task in self.tasks.items() if not task.done()
        ]
        for _, task in active:
            task.cancel()
        if active:
            await asyncio.gather(*(task for _, task in active), return_exceptions=True)
        for job_id, _ in active:
            state = self.jobs.get(job_id)
            if state:
                self._update(
                    state,
                    status="interrupted",
                    message="The app exited; completed movie assets and edit decisions were kept.",
                )
        await self.planner.stop()

    async def _run(self, state: dict[str, Any]) -> None:
        async with self.controller.workflow_lock:
            job_dir = self._job_dir(state["id"])
            config = state["config"]
            try:
                plan_path = job_dir / "plan.json"
                if plan_path.exists():
                    plan = json.loads(plan_path.read_text(encoding="utf-8"))
                else:
                    self._update(state, status="planning", progress=2, message="Freeing GPU memory for the local story planner…")
                    await self.controller.ensure_ready()
                    await self.controller.free_models()
                    await asyncio.sleep(1.5)
                    await self.planner.start(job_dir)
                    self._update(state, progress=5, message="Bonsai is writing the screenplay and continuity bible…")
                    plan = await self.planner.plan(config["sentence"], config["shots"])
                    _atomic_json(plan_path, plan)
                    self._update(state, title=plan["title"], logline=plan["logline"], plan=plan, progress=9, message="Story plan passed the continuity schema.")
                    await self.planner.stop()
                    await asyncio.sleep(1)

                state.update(title=plan["title"], logline=plan["logline"], plan=plan)
                self._ensure_editor_state(state)
                self._save(state)
                await self.controller.ensure_ready()
                if config.get("narration", True):
                    await self._render_narration(state, plan, job_dir)
                    await self.controller.free_models()
                    await asyncio.sleep(1)
                await self._render_shots(state, plan, job_dir)
                await self._assemble(state, plan, job_dir)
            except asyncio.CancelledError:
                await self.planner.stop()
                self._update(state, status="cancelled", message="Movie generation was cancelled. Completed files were kept.")
                raise
            except Exception as exc:
                await self.planner.stop()
                LOGGER.exception("Movie %s failed", state["id"])
                self._update(state, status="failed", message=str(exc), error=str(exc))

    def _visual_prompt(self, plan: dict[str, Any], shot: dict[str, Any]) -> str:
        rules = "; ".join(plan["continuity_rules"])
        return (
            f"{plan['style']}. {shot['camera']}. In {plan['location']}. "
            f"The only protagonist is {plan['character']}. {shot['action']}. "
            f"Strict continuity rules: {rules}. Identical face, hair, age, body, wardrobe, props, "
            "architecture, weather, light direction and color palette in every shot. One simple action, "
            "natural restrained motion, coherent anatomy, no new characters, no visible words, no subtitles, no watermark."
        )

    async def _wait_prompt(self, prompt_id: str, state: dict[str, Any], message: str) -> list[dict[str, str]]:
        while True:
            result = await self.controller.job(prompt_id)
            if result["state"] == "complete":
                return result.get("files", [])
            if result["state"] == "failed":
                error = result.get("error") or {}
                raise MovieError(error.get("exception_message") or error.get("exception_type") or "ComfyUI generation failed.")
            self._update(state, message=message)
            await asyncio.sleep(2)

    async def _render_shots(self, state: dict[str, Any], plan: dict[str, Any], job_dir: Path) -> None:
        total = len(plan["shots"])
        shot_files = list(state.get("shot_files", []))
        for index, shot in enumerate(plan["shots"], start=1):
            if index <= len(shot_files) and (self.output_root / shot_files[index - 1]).exists():
                continue
            progress = 28 + int((index - 1) / total * 55)
            self._update(state, status="rendering", progress=progress, current_shot=index, message=f"Rendering shot {index} of {total}: {shot['beat']}")
            prefix = f"wan_movies/{state['id']}/shots/shot_{index:03d}"
            video_config = {
                "prompt": self._visual_prompt(plan, shot),
                "negative": "identity drift, different person, different face, changed clothing, changed location, extra characters, duplicate person, overexposed, static, blurry, low detail, subtitles, watermark, text, painting, malformed anatomy, deformed face, fused limbs, NSFW",
                "width": state["config"]["width"], "height": state["config"]["height"],
                "frames": state["config"]["frames"], "fps": state["config"]["fps"],
                "seed": state["config"]["seed"], "filename_prefix": prefix,
            }
            reply = await self.controller.submit(self.video_prompt_builder(video_config), f"movie-{state['id']}")
            files = await self._wait_prompt(reply["prompt_id"], state, f"Rendering shot {index} of {total}: {shot['beat']}")
            video = next((f["path"] for f in reversed(files) if f["filename"].lower().endswith((".mp4", ".webm"))), None)
            if not video:
                raise MovieError(f"Shot {index} finished without a video file.")
            shot_files.append(video)
            state["clips"][index - 1]["video_path"] = video
            state["clips"][index - 1]["status"] = "ready" if state["clips"][index - 1].get("audio_path") else "video_ready"
            motion = await self._analyze_video(self.output_root / video)
            state["clips"][index - 1]["motion"] = motion
            self._update(state, shot_files=shot_files, completed_shots=len(shot_files))
        self._update(state, progress=84, message="All maximum-frame visual shots are complete.")

    def _tts_prompt(self, text: str, prefix: str, seed: int) -> dict[str, Any]:
        token_limit = max(240, min(1800, len(text.split()) * 16))
        token_limit = int((token_limit + 7) // 8 * 8)
        return {
            "1": {"class_type": "ChatterboxTTS", "inputs": {
                "model_pack_name": "resembleai_default_voice", "text": text,
                "max_new_tokens": token_limit, "flow_cfg_scale": 0.7,
                "exaggeration": 0.42, "temperature": 0.72, "cfg_weight": 0.5,
                "repetition_penalty": 1.2, "min_p": 0.05, "top_p": 1.0,
                "seed": seed, "use_watermark": False,
            }},
            "2": {"class_type": "SaveAudio", "inputs": {"audio": ["1", 0], "filename_prefix": prefix}},
        }

    async def _render_narration(self, state: dict[str, Any], plan: dict[str, Any], job_dir: Path) -> None:
        total = len(plan["shots"])
        audio_files = list(state.get("audio_files", []))
        for index, shot in enumerate(plan["shots"], start=1):
            if index <= len(audio_files) and (self.output_root / audio_files[index - 1]).exists():
                continue
            progress = 10 + int((index - 1) / total * 15)
            self._update(state, status="narrating", progress=progress, current_shot=index, message=f"Recording local narration {index} of {total}…")
            prefix = f"wan_movies/{state['id']}/audio/shot_{index:03d}"
            graph = self._tts_prompt(shot["narration"], prefix, state["config"]["seed"] + index)
            reply = await self.controller.submit(graph, f"movie-audio-{state['id']}")
            files = await self._wait_prompt(reply["prompt_id"], state, f"Recording local narration {index} of {total}…")
            audio = next((f["path"] for f in reversed(files) if f["filename"].lower().endswith((".flac", ".wav", ".mp3", ".ogg"))), None)
            if not audio:
                raise MovieError(f"Narration {index} finished without an audio file.")
            audio_files.append(audio)
            state["clips"][index - 1]["audio_path"] = audio
            state["clips"][index - 1]["status"] = "ready" if state["clips"][index - 1].get("video_path") else "audio_ready"
            self._update(state, audio_files=audio_files, completed_audio=len(audio_files))
        self._update(state, progress=26, message="Narration is complete; loading the maximum-frame video model.")

    async def _analyze_video(self, path: Path) -> dict[str, Any]:
        ffprobe = shutil.which("ffprobe")
        ffmpeg = shutil.which("ffmpeg")
        if not ffprobe or not ffmpeg:
            return {"warning": "FFmpeg diagnostics unavailable"}
        probe = await asyncio.create_subprocess_exec(
            ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=nb_read_frames,avg_frame_rate,duration",
            "-of", "json", str(path), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await probe.communicate()
        data = json.loads(stdout.decode() or "{}") if probe.returncode == 0 else {}
        stream = (data.get("streams") or [{}])[0]
        hashes = await asyncio.create_subprocess_exec(
            ffmpeg, "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "framemd5", "-",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        hash_out, _ = await hashes.communicate()
        lines = [line for line in hash_out.decode(errors="replace").splitlines() if line and not line.startswith("#")]
        frame_hashes = [line.rsplit(",", 1)[-1].strip() for line in lines]
        frames = len(frame_hashes) or int(stream.get("nb_read_frames") or 0)
        unique = len(set(frame_hashes)) if frame_hashes else frames
        return {
            "frames": frames, "unique_frames": unique,
            "unique_ratio": round(unique / frames, 4) if frames else 0,
            "duration": round(float(stream.get("duration") or 0), 4),
            "avg_frame_rate": stream.get("avg_frame_rate"),
            "warning": "Low temporal change" if frames and unique / frames < 0.35 else None,
        }

    async def _run_process(self, args: list[str], log_path: Path) -> None:
        with log_path.open("ab") as log:
            process = await asyncio.create_subprocess_exec(*args, stdout=log, stderr=log)
            code = await process.wait()
        if code:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
            raise MovieError(f"FFmpeg failed with exit code {code}.\n{tail}")

    async def _duration(self, ffprobe: str, path: Path) -> float:
        process = await asyncio.create_subprocess_exec(
            ffprobe, "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await process.communicate()
        if process.returncode:
            raise MovieError(f"Could not inspect media duration: {path.name}")
        return float(stdout.decode().strip())

    async def _assemble(self, state: dict[str, Any], plan: dict[str, Any], job_dir: Path) -> None:
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        if not ffmpeg or not ffprobe:
            raise MovieError("FFmpeg and ffprobe are required but were not found.")
        self._update(state, status="assembling", progress=90, message="Editing shots and narration into the final movie…")
        segments_dir = job_dir / "segments"
        segments_dir.mkdir(exist_ok=True)
        segment_paths: list[Path] = []
        timeline: list[tuple[float, float, str]] = []
        cursor = 0.0
        for index, video_rel in enumerate(state["shot_files"], start=1):
            video = self.output_root / video_rel
            audio = self.output_root / state["audio_files"][index - 1] if state.get("audio_files") else None
            video_duration = await self._duration(ffprobe, video)
            audio_duration = await self._duration(ffprobe, audio) if audio else 0.0
            duration = max(video_duration, audio_duration + 0.8, 1.0)
            pad = max(0.0, duration - video_duration)
            segment = segments_dir / f"segment_{index:03d}.mp4"
            fade_out = max(0.2, duration - 0.25)
            if audio:
                args = [
                    ffmpeg, "-y", "-i", str(video), "-i", str(audio),
                    "-filter_complex",
                    f"[0:v]setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration={pad:.3f},fps={state['config']['fps']},format=yuv420p,fade=t=in:st=0:d=0.12,fade=t=out:st={fade_out:.3f}:d=0.2[v];"
                    f"[1:a]adelay=250|250,apad=pad_dur={duration:.3f},atrim=duration={duration:.3f},afade=t=in:st=0.1:d=0.15,afade=t=out:st={max(0.1,duration-0.3):.3f}:d=0.25[a]",
                    "-map", "[v]", "-map", "[a]", "-t", f"{duration:.3f}",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                    "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(segment),
                ]
            else:
                args = [
                    ffmpeg, "-y", "-i", str(video), "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
                    "-filter_complex", f"[0:v]setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration={pad:.3f},fps={state['config']['fps']},format=yuv420p[v]",
                    "-map", "[v]", "-map", "1:a", "-t", f"{duration:.3f}",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-c:a", "aac", "-b:a", "128k", str(segment),
                ]
            await self._run_process(args, job_dir / "ffmpeg.log")
            segment_paths.append(segment)
            timeline.append((cursor, cursor + duration, plan["shots"][index - 1]["narration"]))
            cursor += duration
            self._update(state, progress=90 + int(index / len(state["shot_files"]) * 8), message=f"Editing segment {index} of {len(state['shot_files'])}…")

        concat = job_dir / "concat.txt"
        concat.write_text("".join(f"file '{str(path).replace(chr(39), chr(39)*2)}'\n" for path in segment_paths), encoding="utf-8")
        final_name = _slug(plan["title"]) + ".mp4"
        final_path = job_dir / final_name
        await self._run_process([
            ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
            "-c", "copy", "-movflags", "+faststart", "-metadata", f"title={plan['title']}", str(final_path),
        ], job_dir / "ffmpeg.log")

        srt_lines: list[str] = []
        for index, (start, end, text) in enumerate(timeline, start=1):
            def stamp(value: float) -> str:
                millis = int(round(value * 1000))
                hours, millis = divmod(millis, 3_600_000)
                minutes, millis = divmod(millis, 60_000)
                seconds, millis = divmod(millis, 1000)
                return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"
            srt_lines.extend([str(index), f"{stamp(start)} --> {stamp(end)}", text, ""])
        (job_dir / (_slug(plan["title"]) + ".srt")).write_text("\n".join(srt_lines), encoding="utf-8")
        relative = str(final_path.relative_to(self.output_root)).replace("\\", "/")
        self._update(state, status="complete", progress=100, message="Movie complete.", final_path=relative, duration=cursor)

    @staticmethod
    def _atempo(factor: float) -> str:
        """Return a legal FFmpeg atempo chain for any positive speed factor."""
        factor = max(0.01, factor)
        values: list[float] = []
        while factor > 2.0:
            values.append(2.0)
            factor /= 2.0
        while factor < 0.5:
            values.append(0.5)
            factor /= 0.5
        values.append(factor)
        return ",".join(f"atempo={value:.6f}" for value in values)

    async def _video_frame_count(self, ffprobe: str, path: Path) -> int:
        process = await asyncio.create_subprocess_exec(
            ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await process.communicate()
        if process.returncode:
            raise MovieError(f"Could not count video frames: {path.name}")
        try:
            return int(stdout.decode().strip())
        except ValueError as exc:
            raise MovieError(f"FFprobe returned no frame count for {path.name}") from exc

    @staticmethod
    def _timecode(seconds: float, fps: int) -> str:
        total_frames = max(0, int(round(seconds * fps)))
        frames = total_frames % fps
        total_seconds = total_frames // fps
        return f"{total_seconds // 3600:02d}:{(total_seconds // 60) % 60:02d}:{total_seconds % 60:02d}:{frames:02d}"

    def _write_edit_exports(
        self, job_dir: Path, title: str, timeline: list[dict[str, Any]], fps: int,
    ) -> None:
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer)
        writer.writerow([
            "order", "shot", "timeline_in", "timeline_out", "duration_seconds",
            "sync_mode", "playback_fps", "fill_mode", "audio_mode", "source_video", "source_audio",
        ])
        for item in timeline:
            edit = item["edit"]
            writer.writerow([
                item["order"], item["number"], f"{item['start']:.3f}", f"{item['end']:.3f}",
                f"{item['duration']:.3f}", edit["sync_mode"], edit["fps"], edit["fill_mode"],
                edit["audio_mode"], item["video_path"], item.get("audio_path") or "",
            ])
        (job_dir / "shot-list.csv").write_text("\ufeff" + buffer.getvalue(), encoding="utf-8")
        edl = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""]
        for item in timeline:
            source_out = self._timecode(item["duration"], fps)
            record_in = self._timecode(item["start"], fps)
            record_out = self._timecode(item["end"], fps)
            edl.extend([
                f"{item['order']:03d}  SHOT{item['number']:03d} V     C        00:00:00:00 {source_out} {record_in} {record_out}",
                f"* FROM CLIP NAME: shot_{item['number']:03d}.mp4",
            ])
        (job_dir / "timeline.edl").write_text("\n".join(edl) + "\n", encoding="utf-8")

    async def _assemble(self, state: dict[str, Any], plan: dict[str, Any], job_dir: Path) -> None:
        """Render edit decisions in two passes so audio muxing cannot collapse video timing."""
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        if not ffmpeg or not ffprobe:
            raise MovieError("FFmpeg and ffprobe are required but were not found.")
        self._ensure_editor_state(state)
        self._update(state, status="assembling", progress=90, message="Rendering the non-destructive movie edit...")
        segments_dir = job_dir / "segments"
        segments_dir.mkdir(exist_ok=True)
        segment_paths: list[Path] = []
        timeline: list[dict[str, Any]] = []
        cursor = 0.0
        delivery_fps = max(1, int(state["config"].get("fps", 12)))
        clip_map = {int(clip["number"]): clip for clip in state.get("clips", [])}
        edits = sorted(
            (edit for edit in state.get("edits", []) if edit.get("enabled", True)),
            key=lambda item: (int(item.get("order", item["number"])), int(item["number"])),
        )
        if not edits:
            raise MovieError("The edit has no enabled clips.")

        for position, edit in enumerate(edits, start=1):
            number = int(edit["number"])
            clip = clip_map.get(number)
            if not clip or not clip.get("video_path"):
                raise MovieError(f"Shot {number} has no generated video yet.")
            video_rel = clip["video_path"]
            audio_rel = clip.get("audio_path")
            video = self.output_root / video_rel
            audio = self.output_root / audio_rel if audio_rel else None
            raw_video_duration = await self._duration(ffprobe, video)
            raw_audio_duration = await self._duration(ffprobe, audio) if audio else 0.0
            video_in = min(float(edit.get("video_in", 0)), max(0.0, raw_video_duration - 0.05))
            video_out = min(raw_video_duration, max(video_in + 0.05, float(edit.get("video_out", 0)) or raw_video_duration))
            video_span = video_out - video_in
            audio_in = min(float(edit.get("audio_in", 0)), max(0.0, raw_audio_duration - 0.05)) if audio else 0.0
            audio_out = min(raw_audio_duration, max(audio_in + 0.05, float(edit.get("audio_out", 0)) or raw_audio_duration)) if audio else 0.0
            audio_span = max(0.0, audio_out - audio_in)
            source_frames = int((clip.get("motion") or {}).get("frames") or max(1, round(raw_video_duration * delivery_fps)))
            playback_fps = max(1, int(edit.get("fps", delivery_fps)))
            native_duration = max(0.1, source_frames / playback_fps)
            sync_mode = edit.get("sync_mode", "fit_video_to_audio")
            custom_duration = float(edit.get("target_duration", 0))
            if custom_duration > 0:
                duration = custom_duration
            elif sync_mode == "fit_video_to_audio" and audio:
                duration = max(1.0, audio_span + 0.5)
            elif sync_mode == "fit_audio_to_video":
                duration = native_duration
            else:
                duration = max(1.0, native_duration, audio_span + 0.5 if audio else 0.0)
            duration = round(duration, 3)

            video_only = segments_dir / f"video_{position:03d}_shot_{number:03d}.mp4"
            trim = f"trim=start={video_in:.6f}:end={video_out:.6f},setpts=PTS-STARTPTS"
            interpolate = bool(edit.get("motion_interpolation", True))
            rate_filter = f"minterpolate=fps={delivery_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir" if interpolate else f"fps={delivery_fps}"
            fill_mode = edit.get("fill_mode", "freeze")
            if sync_mode == "fit_video_to_audio":
                # Input duration includes the final frame interval, while setpts works on
                # timestamps ending one interval earlier.  Account for that interval and
                # give minterpolate a cloned look-ahead tail so it cannot truncate output.
                timestamp_span = video_span * max(1, source_frames - 1) / max(1, source_frames)
                ratio = duration / max(0.05, timestamp_span)
                visual = f"{trim},setpts={ratio:.8f}*PTS,tpad=stop_mode=clone:stop_duration=2.0,{rate_filter},tpad=stop_mode=clone:stop_duration={duration:.6f}"
                video_args = [ffmpeg, "-y", "-i", str(video), "-vf", visual]
            elif sync_mode == "fixed_fps_fill" and fill_mode in {"loop", "pingpong"}:
                repeat_source = segments_dir / f"repeat_source_{position:03d}_shot_{number:03d}.mp4"
                if fill_mode == "pingpong":
                    graph = (
                        f"[0:v]{trim},setpts=N/{playback_fps}/TB,split=2[f][r];"
                        f"[r]reverse[rr];[f][rr]concat=n=2:v=1:a=0,format=yuv420p[v]"
                    )
                    prepare_args = [
                        ffmpeg, "-y", "-i", str(video), "-filter_complex", graph, "-map", "[v]", "-an",
                        "-c:v", "libx264", "-preset", "fast", "-crf", "18", str(repeat_source),
                    ]
                else:
                    prepare_args = [
                        ffmpeg, "-y", "-i", str(video), "-vf",
                        f"{trim},setpts=N/{playback_fps}/TB,format=yuv420p", "-an",
                        "-c:v", "libx264", "-preset", "fast", "-crf", "18", str(repeat_source),
                    ]
                await self._run_process(prepare_args, job_dir / "ffmpeg.log")
                visual = f"trim=duration={duration:.6f},setpts=PTS-STARTPTS,{rate_filter},tpad=stop_mode=clone:stop_duration={duration:.6f}"
                video_args = [ffmpeg, "-y", "-stream_loop", "-1", "-i", str(repeat_source), "-vf", visual]
            else:
                pad_mode = "add:color=black" if fill_mode == "black" else "clone"
                pad = max(0.0, duration - native_duration + 0.25)
                visual = f"{trim},setpts=N/{playback_fps}/TB,tpad=stop_mode={pad_mode}:stop_duration={pad:.6f},{rate_filter},tpad=stop_mode=clone:stop_duration={duration:.6f}"
                video_args = [ffmpeg, "-y", "-i", str(video), "-vf", visual]
            video_args.extend([
                "-t", f"{duration:.3f}", "-an", "-r", str(delivery_fps), "-pix_fmt", "yuv420p",
                "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-video_track_timescale", "90000",
                "-movflags", "+faststart", str(video_only),
            ])
            await self._run_process(video_args, job_dir / "ffmpeg.log")

            # Some FFmpeg temporal filters legitimately end before the requested
            # -t because they retain look-ahead frames. Normalize the video in a
            # separate video-only pass before audio is ever introduced.
            minimum_video_frames = max(1, int(duration * delivery_fps) - 2)
            if await self._video_frame_count(ffprobe, video_only) < minimum_video_frames:
                normalized = segments_dir / f"normalized_{position:03d}_shot_{number:03d}.mp4"
                await self._run_process([
                    ffmpeg, "-y", "-i", str(video_only), "-vf",
                    f"tpad=stop_mode=clone:stop_duration={duration:.6f},fps={delivery_fps},format=yuv420p",
                    "-t", f"{duration:.3f}", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                    "-video_track_timescale", "90000", "-movflags", "+faststart", str(normalized),
                ], job_dir / "ffmpeg.log")
                normalized.replace(video_only)

            segment = segments_dir / f"segment_{position:03d}.mp4"
            if audio:
                content_duration = max(0.2, duration - 0.35)
                audio_mode = "fit" if sync_mode == "fit_audio_to_video" else edit.get("audio_mode", "pad")
                audio_filters = [f"atrim=start={audio_in:.6f}:end={audio_out:.6f}", "asetpts=PTS-STARTPTS"]
                if audio_mode == "fit" and audio_span > 0:
                    audio_filters.append(self._atempo(audio_span / content_duration))
                audio_filters.extend([
                    f"volume={float(edit.get('narration_gain', 1.0)):.4f}", "adelay=175|175",
                    f"apad=pad_dur={duration:.6f}", f"atrim=duration={duration:.6f}",
                    "afade=t=in:st=0.05:d=0.12", f"afade=t=out:st={max(0.05, duration - 0.22):.6f}:d=0.18",
                ])
                mux_args = [
                    ffmpeg, "-y", "-i", str(video_only), "-i", str(audio),
                    "-filter_complex", f"[1:a]{','.join(audio_filters)}[a]", "-map", "0:v:0", "-map", "[a]",
                    "-t", f"{duration:.3f}", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
                    "-movflags", "+faststart", str(segment),
                ]
            else:
                mux_args = [
                    ffmpeg, "-y", "-i", str(video_only), "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
                    "-map", "0:v:0", "-map", "1:a:0", "-t", f"{duration:.3f}", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(segment),
                ]
            await self._run_process(mux_args, job_dir / "ffmpeg.log")

            actual_frames = await self._video_frame_count(ffprobe, segment)
            expected_frames = max(1, int(duration * delivery_fps) - 2)
            if actual_frames < expected_frames:
                raise MovieError(f"Temporal validation failed for shot {number}: {actual_frames} frames, expected at least {expected_frames}.")
            segment_paths.append(segment)
            timeline.append({
                "order": position, "number": number, "start": cursor, "end": cursor + duration,
                "duration": duration, "frames": actual_frames, "edit": edit,
                "video_path": video_rel, "audio_path": audio_rel,
            })
            cursor += duration
            self._update(
                state, progress=90 + int(position / len(edits) * 8), timeline=timeline,
                message=f"Rendered and validated segment {position} of {len(edits)}...",
            )

        concat = job_dir / "concat.txt"
        concat.write_text("".join(f"file '{str(path).replace(chr(39), chr(39)*2)}'\n" for path in segment_paths), encoding="utf-8")
        final_path = job_dir / (_slug(plan["title"]) + ".mp4")
        await self._run_process([
            ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy",
            "-movflags", "+faststart", "-metadata", f"title={plan['title']}", str(final_path),
        ], job_dir / "ffmpeg.log")

        def stamp(value: float) -> str:
            millis = int(round(value * 1000))
            hours, millis = divmod(millis, 3_600_000)
            minutes, millis = divmod(millis, 60_000)
            seconds, millis = divmod(millis, 1000)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

        srt_lines: list[str] = []
        for index, item in enumerate(timeline, start=1):
            shot = next((shot for shot in plan["shots"] if int(shot.get("number", 0)) == item["number"]), {})
            srt_lines.extend([str(index), f"{stamp(item['start'])} --> {stamp(item['end'])}", shot.get("narration", ""), ""])
        (job_dir / (_slug(plan["title"]) + ".srt")).write_text("\n".join(srt_lines), encoding="utf-8")
        self._write_edit_exports(job_dir, plan["title"], timeline, delivery_fps)
        relative = str(final_path.relative_to(self.output_root)).replace("\\", "/")
        self._update(
            state, status="complete", progress=100, message="Movie edit complete and temporally validated.",
            final_path=relative, duration=cursor, timeline=timeline,
        )
