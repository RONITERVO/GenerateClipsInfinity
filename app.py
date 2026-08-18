from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import subprocess
import sys
import time
import uuid
import webbrowser
from pathlib import Path
from typing import Any

from aiohttp import ClientSession, ClientTimeout, web

from process_utils import terminate_process_tree
from theater_pipeline import SupertonicRuntime, TheaterError, TheaterManager


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


AI_ROOT = _env_path("WAN_AI_ROOT", Path(r"D:\AI"))
LOCAL_AI_ROOT = _env_path("WAN_LOCAL_AI_ROOT", Path(r"D:\LocalAI"))
COMFY_ROOT = _env_path("WAN_COMFY_ROOT", AI_ROOT / "ComfyUI")
STORY_MODEL_ROOT = _env_path("WAN_GEMMA4_ROOT", LOCAL_AI_ROOT / "Gemma4E4B")
LLAMA_RUNTIME_ROOT = _env_path("WAN_LLAMA_RUNTIME_ROOT", STORY_MODEL_ROOT)
CUDA_LLAMA_RUNTIME_ROOT = _env_path("WAN_CUDA_LLAMA_RUNTIME_ROOT", LOCAL_AI_ROOT / "Bonsai27B")
SUPERTONIC_ROOT = _env_path("WAN_SUPERTONIC_ROOT", LOCAL_AI_ROOT / "Supertonic3")
KIWIX_ROOT = _env_path("WAN_KIWIX_ROOT", LOCAL_AI_ROOT / "OfflineWikipedia")
COMFY_URL = os.environ.get("WAN_COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
OUTPUT_ROOT = _env_path("WAN_OUTPUT_ROOT", COMFY_ROOT / "output")
HOST = os.environ.get("WAN_HOST", "127.0.0.1")
PORT = int(os.environ.get("WAN_PORT", "7868"))

LOG_DIR = APP_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "wan-video-ui.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
LOGGER = logging.getLogger("wan-video-ui")

DEFAULT_NEGATIVE = (
    "overexposed, static, blurry, low detail, subtitles, watermark, text, painting, "
    "still image, washed out, worst quality, low quality, JPEG artifacts, distorted "
    "hands, malformed fingers, deformed face, fused limbs, cluttered background, NSFW"
)

RUNTIME_STATE = web.AppKey("runtime_state", dict)


def build_prompt(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config["seed"])
    width = int(config["width"])
    height = int(config["height"])
    frames = int(config["frames"])
    fps = float(config["fps"])
    positive = str(config["prompt"]).strip()
    negative = str(config.get("negative", DEFAULT_NEGATIVE)).strip()
    prefix = str(config["filename_prefix"])

    return {
        "1": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                "type": "wan",
                "device": "default",
            },
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": positive, "clip": ["1", 0]},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative, "clip": ["1", 0]},
        },
        "4": {
            "class_type": "EmptyHunyuanLatentVideo",
            "inputs": {
                "width": width,
                "height": height,
                "length": frames,
                "batch_size": 1,
            },
        },
        "5": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
                "weight_dtype": "default",
            },
        },
        "6": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["5", 0],
                "lora_name": "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors",
                "strength_model": 1.0,
            },
        },
        "7": {
            "class_type": "ModelSamplingSD3",
            "inputs": {"model": ["6", 0], "shift": 5.0},
        },
        "8": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["7", 0],
                "add_noise": "enable",
                "noise_seed": seed,
                "steps": 4,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
                "start_at_step": 0,
                "end_at_step": 2,
                "return_with_leftover_noise": "enable",
            },
        },
        "9": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
                "weight_dtype": "default",
            },
        },
        "10": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["9", 0],
                "lora_name": "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors",
                "strength_model": 1.0,
            },
        },
        "11": {
            "class_type": "ModelSamplingSD3",
            "inputs": {"model": ["10", 0], "shift": 5.0},
        },
        "12": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["11", 0],
                "add_noise": "disable",
                "noise_seed": seed,
                "steps": 4,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["8", 0],
                "start_at_step": 2,
                "end_at_step": 4,
                "return_with_leftover_noise": "disable",
            },
        },
        "13": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "wan_2.1_vae.safetensors"},
        },
        "14": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["12", 0], "vae": ["13", 0]},
        },
        "15": {
            "class_type": "CreateVorbisAudio",
            "inputs": {
                "sound": "silence",
                "duration": frames / fps,
                "sample_rate": 44100,
            },
        },
        "16": {
            "class_type": "SaveVideo",
            "inputs": {
                "images": ["14", 0],
                "audio": ["15", 0],
                "filename_prefix": prefix,
                "fps": fps,
                "format": "mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_output": True,
            },
        },
    }


class ComfyController:
    def __init__(self) -> None:
        self.session: ClientSession | None = None
        self.process: subprocess.Popen[bytes] | None = None
        self.start_lock = asyncio.Lock()
        self.workflow_lock = asyncio.Lock()
        self.starting = False

    async def open(self) -> None:
        self.session = ClientSession(timeout=ClientTimeout(total=15))

    async def close(self) -> None:
        if self.session:
            await self.session.close()
            self.session = None

    async def stop_owned_process(self) -> None:
        """Unload ComfyUI and stop it only when this app launched the process."""
        await terminate_process_tree(self.process)
        self.process = None

    async def stats(self) -> dict[str, Any] | None:
        assert self.session
        try:
            async with self.session.get(f"{COMFY_URL}/system_stats", timeout=3) as response:
                if response.status != 200:
                    return None
                return await response.json()
        except Exception:
            return None

    async def ensure_ready(self) -> dict[str, Any]:
        current = await self.stats()
        if current:
            return current

        async with self.start_lock:
            current = await self.stats()
            if current:
                return current
            self.starting = True
            try:
                self._start_process()
                for _ in range(120):
                    await asyncio.sleep(0.75)
                    current = await self.stats()
                    if current:
                        return current
                    if self.process and self.process.poll() is not None:
                        raise RuntimeError(
                            "ComfyUI exited during startup. Check the UI logs folder."
                        )
                raise RuntimeError("ComfyUI did not become ready within 90 seconds.")
            finally:
                self.starting = False

    def _start_process(self) -> None:
        python = COMFY_ROOT / ".venv" / "Scripts" / "python.exe"
        if not python.exists():
            raise RuntimeError(f"ComfyUI Python was not found at {python}")
        env = os.environ.copy()
        env.update(
            {
                "PYTHONUTF8": "1",
                "HF_HOME": str(COMFY_ROOT / ".cache" / "huggingface"),
                "TRANSFORMERS_CACHE": str(COMFY_ROOT / ".cache" / "huggingface"),
                "HF_XET_HIGH_PERFORMANCE": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "CUDA_VISIBLE_DEVICES": "0",
            }
        )
        args = [
            str(python),
            "main.py",
            "--listen",
            "127.0.0.1",
            "--port",
            "8188",
            "--cuda-device",
            "0",
            "--enable-dynamic-vram",
            "--lowvram",
            "--reserve-vram",
            "1.5",
            "--cache-none",
            "--preview-method",
            "auto",
            "--fast-disk",
        ]
        stdout_path = LOG_DIR / "comfyui.out.log"
        stderr_path = LOG_DIR / "comfyui.err.log"
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
            self.process = subprocess.Popen(
                args,
                cwd=COMFY_ROOT,
                env=env,
                stdout=stdout,
                stderr=stderr,
                creationflags=creationflags,
            )
        LOGGER.info("Started ComfyUI PID %s", self.process.pid)

    async def submit(self, prompt: dict[str, Any], client_id: str) -> dict[str, Any]:
        assert self.session
        await self.ensure_ready()
        async with self.session.post(
            f"{COMFY_URL}/prompt",
            json={"prompt": prompt, "client_id": client_id},
            timeout=30,
        ) as response:
            data = await response.json(content_type=None)
            if response.status != 200:
                raise RuntimeError(data.get("error", {}).get("message") or json.dumps(data))
            return data

    async def free_models(self) -> None:
        assert self.session
        try:
            async with self.session.post(
                f"{COMFY_URL}/free",
                json={"unload_models": True, "free_memory": True},
                timeout=30,
            ) as response:
                await response.read()
        except Exception as exc:
            LOGGER.warning("Could not explicitly unload ComfyUI models: %s", exc)

    async def wait_until_idle(self, timeout_seconds: float = 900) -> None:
        """Wait for any already-queued ComfyUI work before reclaiming its VRAM."""
        assert self.session
        deadline = time.monotonic() + timeout_seconds
        while True:
            async with self.session.get(f"{COMFY_URL}/queue", timeout=10) as response:
                queue = await response.json(content_type=None)
            if not queue.get("queue_running") and not queue.get("queue_pending"):
                return
            if time.monotonic() >= deadline:
                raise RuntimeError("Timed out waiting for the existing ComfyUI queue to become idle.")
            await asyncio.sleep(1)

    async def interrupt(self) -> None:
        assert self.session
        try:
            async with self.session.post(f"{COMFY_URL}/interrupt", timeout=10) as response:
                await response.read()
            async with self.session.post(f"{COMFY_URL}/queue", json={"clear": True}, timeout=10) as response:
                await response.read()
        except Exception as exc:
            LOGGER.warning("Could not interrupt the current ComfyUI job: %s", exc)

    async def job(self, prompt_id: str) -> dict[str, Any]:
        assert self.session
        async with self.session.get(f"{COMFY_URL}/history/{prompt_id}") as response:
            history = await response.json(content_type=None)
        if prompt_id in history:
            item = history[prompt_id]
            status = item.get("status", {})
            completed = bool(status.get("completed"))
            files: list[dict[str, str]] = []
            for output in item.get("outputs", {}).values():
                for value in output.values():
                    if not isinstance(value, list):
                        continue
                    for entry in value:
                        if isinstance(entry, dict) and entry.get("filename"):
                            rel = str(Path(entry.get("subfolder", "")) / entry["filename"])
                            files.append({"path": rel.replace("\\", "/"), "filename": entry["filename"]})
            messages = status.get("messages", [])
            error = next(
                (m[1] for m in reversed(messages) if isinstance(m, list) and m and m[0] == "execution_error"),
                None,
            )
            return {
                "state": "failed" if error else ("complete" if completed else "running"),
                "files": files,
                "error": error,
            }

        async with self.session.get(f"{COMFY_URL}/queue") as response:
            queue = await response.json(content_type=None)
        for item in queue.get("queue_running", []):
            if len(item) > 1 and item[1] == prompt_id:
                return {"state": "running", "files": []}
        for item in queue.get("queue_pending", []):
            if len(item) > 1 and item[1] == prompt_id:
                return {"state": "queued", "files": []}
        return {"state": "waiting", "files": []}


CONTROLLER = ComfyController()
THEATER = TheaterManager(
    APP_DIR, OUTPUT_ROOT, STORY_MODEL_ROOT, LLAMA_RUNTIME_ROOT,
    SUPERTONIC_ROOT, KIWIX_ROOT, CONTROLLER, build_prompt,
    cuda_llama_runtime_root=CUDA_LLAMA_RUNTIME_ROOT,
)


async def index(_: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC_DIR / "index.html")


async def api_status(_: web.Request) -> web.Response:
    stats = await CONTROLLER.stats()
    if not stats:
        return web.json_response({"ready": False, "starting": CONTROLLER.starting})
    device = (stats.get("devices") or [{}])[0]
    system = stats.get("system", {})
    return web.json_response(
        {
            "ready": True,
            "starting": False,
            "device": device.get("name", "NVIDIA GPU"),
            "vram_gb": round(float(device.get("vram_total", 0)) / (1024**3), 1),
            "comfy_version": system.get("comfyui_version", "unknown"),
        }
    )


async def api_config(_: web.Request) -> web.Response:
    """Expose only non-secret browser connection settings."""
    return web.json_response({"comfy_url": COMFY_URL})


async def api_video(request: web.Request) -> web.StreamResponse:
    relative = request.query.get("path", "")
    if not relative:
        raise web.HTTPBadRequest(text="Missing path")
    candidate = (OUTPUT_ROOT / relative).resolve()
    try:
        candidate.relative_to(OUTPUT_ROOT.resolve())
    except ValueError as exc:
        raise web.HTTPForbidden(text="Invalid output path") from exc
    if candidate.suffix.lower() not in {".mp4", ".webm", ".wav", ".flac", ".mp3", ".ogg", ".m4a"} or not candidate.is_file():
        raise web.HTTPNotFound(text="Media not found")
    return web.FileResponse(candidate)


def validate_theater_payload(raw: dict[str, Any]) -> dict[str, Any]:
    mode = str(raw.get("mode", "edutainment"))
    prompt = str(raw.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("Whisper a dream seed or write a story or learning idea first.")
    if len(prompt) > 1200:
        raise ValueError("The theater prompt is too long (maximum 1,200 characters).")
    audience = str(raw.get("audience", "family"))
    if mode not in {"dream", "interactive", "story", "edutainment", "lesson"}:
        raise ValueError("Choose dream, interactive character, story, edutainment, or lesson mode.")
    if audience not in {"young", "family", "teen", "adult"}:
        raise ValueError("Choose a supported audience level.")
    voice = str(raw.get("voice", "M1")).upper()
    language = str(raw.get("language", "en")).lower()
    translation_language = str(raw.get("translation_language") or "").lower()
    if voice not in SupertonicRuntime.VOICES:
        raise ValueError("Choose a supported Supertonic voice.")
    if language not in SupertonicRuntime.LANGUAGES:
        raise ValueError("Choose a supported narration language.")
    if translation_language and translation_language not in SupertonicRuntime.TRANSLATION_LANGUAGES:
        raise ValueError("Choose a supported translation language or turn translation off.")
    if translation_language == language:
        raise ValueError("The translation language must differ from the story language.")
    seed = int(raw.get("seed", -1))
    if seed < 0:
        seed = secrets.randbelow(2**31 - 1)
    supplied_settings = raw.get("quality_settings", {})
    if not isinstance(supplied_settings, dict):
        raise ValueError("Advanced generation settings must be an object.")
    defaults = TheaterManager.CINEMA_DEFAULTS
    try:
        width = int(supplied_settings.get("width", defaults["width"]))
        height = int(supplied_settings.get("height", defaults["height"]))
        frames = int(supplied_settings.get("frames", defaults["frames"]))
        fps = int(supplied_settings.get("fps", defaults["fps"]))
        min_words = int(supplied_settings.get("min_words", defaults["min_words"]))
        max_words = int(supplied_settings.get("max_words", defaults["max_words"]))
        max_slow = float(supplied_settings.get("max_slow", defaults["max_slow"]))
        context_compaction_scenes = int(
            raw.get("context_compaction_scenes", TheaterManager.DEFAULT_CONTEXT_COMPACTION_SCENES)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Advanced generation values must be numeric.") from exc
    if width < 192 or width > 832 or width % 16:
        raise ValueError("Theater width must be a multiple of 16 between 192 and 832.")
    if height < 192 or height > 832 or height % 16:
        raise ValueError("Theater height must be a multiple of 16 between 192 and 832.")
    if frames < 9 or frames > 81 or (frames - 1) % 4:
        raise ValueError("Theater source frames must be 9-81 and follow the 4n+1 rule.")
    if fps < 1 or fps > 60:
        raise ValueError("Theater playback FPS must be between 1 and 60.")
    if min_words < 30 or min_words > 1200:
        raise ValueError("Minimum narration words must be between 30 and 1,200.")
    if max_words < min_words or max_words > 2400:
        raise ValueError("Maximum narration words must be at least the minimum and no more than 2,400.")
    if max_slow < 1 or max_slow > 20:
        raise ValueError("Maximum slow-motion must be between 1x and 20x.")
    if context_compaction_scenes != 0 and not 5 <= context_compaction_scenes <= 200:
        raise ValueError("Context compaction must be 0 (interval off) or between 5 and 200 scenes.")
    quality_settings = {
        "width": width, "height": height, "frames": frames, "fps": fps,
        "min_words": min_words, "max_words": max_words, "max_slow": max_slow,
    }
    return {
        "prompt": prompt,
        "learning_focus": "" if mode == "dream" else str(raw.get("learning_focus", "")).strip()[:800],
        "quality": "custom", "quality_settings": quality_settings,
        "mode": mode, "audience": audience, "seed": seed,
        "voice": voice, "language": language, "translation_language": translation_language,
        "context_compaction_scenes": context_compaction_scenes,
    }


async def api_theater_start(request: web.Request) -> web.Response:
    try:
        return web.json_response(THEATER.start(validate_theater_payload(await request.json())))
    except (TheaterError, ValueError, TypeError) as exc:
        return web.json_response({"error": str(exc)}, status=400)


async def api_theater_recent(_: web.Request) -> web.Response:
    return web.json_response({"sessions": THEATER.recent()})


async def api_theater_status(request: web.Request) -> web.Response:
    state = THEATER.get(request.match_info["session_id"])
    if not state:
        raise web.HTTPNotFound(text="Theater session not found")
    return web.json_response(state)


async def api_theater_stop(request: web.Request) -> web.Response:
    try:
        return web.json_response(await THEATER.stop(request.match_info["session_id"]))
    except TheaterError as exc:
        return web.json_response({"error": str(exc)}, status=400)


async def api_theater_resume(request: web.Request) -> web.Response:
    try:
        return web.json_response(THEATER.resume(request.match_info["session_id"]))
    except TheaterError as exc:
        return web.json_response({"error": str(exc)}, status=400)


async def api_theater_live_directive(request: web.Request) -> web.Response:
    try:
        raw = await request.json()
        if not isinstance(raw, dict):
            raise ValueError("Request body must be a JSON object.")
        return web.json_response(THEATER.add_live_directive(
            request.match_info["session_id"], raw.get("text", ""), str(raw.get("scope", "next_scene")),
            str(raw.get("delivery", "after_buffer")),
        ))
    except (TheaterError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return web.json_response({"error": str(exc)}, status=400)


async def api_theater_remove_directive(request: web.Request) -> web.Response:
    try:
        return web.json_response(THEATER.remove_live_directive(
            request.match_info["session_id"], request.match_info["directive_id"],
        ))
    except TheaterError as exc:
        return web.json_response({"error": str(exc)}, status=400)


async def api_theater_voice_preview(request: web.Request) -> web.Response:
    try:
        raw = await request.json()
        voice = str(raw.get("voice", "M1")).upper()
        language = str(raw.get("language", "en")).lower()
        if voice not in SupertonicRuntime.VOICES or language not in SupertonicRuntime.LANGUAGES:
            raise ValueError("Choose a supported voice and language.")
        samples = {
            "fi": "Tervetuloa loputtomaan teatteriin. Jokainen uusi kohtaus jatkaa tarinaa ja säilyy omalla tietokoneellasi.",
            "en": "Welcome to the endless theater. Every new scene continues the story and stays safely on your own computer.",
        }
        text = samples.get(language, samples["en"])
        relative = f"wan_theater/_voice_previews/{voice}_{language}.wav"
        output = OUTPUT_ROOT / relative
        if not output.exists():
            await THEATER.supertonic.start(APP_DIR / "logs")
            await THEATER.supertonic.synthesize(text, output, voice=voice, language=language)
        return web.json_response({"path": relative, "voice": voice, "language": language})
    except (TheaterError, ValueError, TypeError) as exc:
        return web.json_response({"error": str(exc)}, status=400)


def _is_local_exit_request(request: web.Request) -> bool:
    remote = request.remote or ""
    same_machine = remote in {"127.0.0.1", "::1"} or remote.startswith("::ffff:127.")
    return same_machine and request.headers.get("X-Wan-Local-Exit") == "release-owned-resources"


async def _release_owned_resources(app: web.Application) -> None:
    """Archive active work, release accelerators, and terminate only owned helpers."""
    runtime = app[RUNTIME_STATE]

    async def cancel_background_start() -> None:
        task = runtime.get("comfy_start_task")
        if task and task is not asyncio.current_task() and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    steps = (
        ("cancel background ComfyUI startup", cancel_background_start),
        ("interrupt ComfyUI work", CONTROLLER.interrupt),
        ("archive Theater work and stop Theater helpers", THEATER.shutdown),
        ("unload ComfyUI models", CONTROLLER.free_models),
        ("stop app-owned ComfyUI", CONTROLLER.stop_owned_process),
    )
    for description, action in steps:
        try:
            await action()
        except Exception:
            LOGGER.exception("One-click exit could not %s", description)
    LOGGER.info("All reachable app-owned generation resources were released by the exit action")
    runtime["shutdown_complete"] = True
    runtime["shutdown_event"].set()


async def api_shutdown(request: web.Request) -> web.Response:
    if not _is_local_exit_request(request):
        raise web.HTTPForbidden(text="Local exit authorization required")
    runtime = request.app[RUNTIME_STATE]
    task = runtime.get("resource_release_task")
    if not task:
        task = asyncio.create_task(
            _release_owned_resources(request.app), name="release-owned-resources",
        )
        runtime["resource_release_task"] = task
    return web.json_response(
        {"status": "shutting_down", "message": "Archiving work and releasing RAM and VRAM."},
        status=202,
    )


async def on_startup(app: web.Application) -> None:
    await CONTROLLER.open()
    THEATER.load_existing()
    (APP_DIR / "wan-video-ui.pid").write_text(str(os.getpid()), encoding="utf-8")
    app[RUNTIME_STATE]["comfy_start_task"] = asyncio.create_task(
        CONTROLLER.ensure_ready(), name="comfy-background-start",
    )


async def on_cleanup(app: web.Application) -> None:
    task = app[RUNTIME_STATE].get("resource_release_task")
    if task:
        await task
    else:
        await _release_owned_resources(app)
    await CONTROLLER.close()
    (APP_DIR / "wan-video-ui.pid").unlink(missing_ok=True)


def create_app() -> web.Application:
    app = web.Application(client_max_size=1024 * 1024)
    app[RUNTIME_STATE] = {
        "shutdown_event": asyncio.Event(),
        "shutdown_complete": False,
        "resource_release_task": None,
        "comfy_start_task": None,
    }
    app.router.add_get("/", index)
    app.router.add_get("/api/config", api_config)
    app.router.add_get("/api/status", api_status)
    app.router.add_get("/api/video", api_video)
    app.router.add_post("/api/theater", api_theater_start)
    app.router.add_get("/api/theater", api_theater_recent)
    app.router.add_post("/api/theater/voice-preview", api_theater_voice_preview)
    app.router.add_get("/api/theater/{session_id}", api_theater_status)
    app.router.add_post("/api/theater/{session_id}/stop", api_theater_stop)
    app.router.add_post("/api/theater/{session_id}/resume", api_theater_resume)
    app.router.add_post("/api/theater/{session_id}/directives", api_theater_live_directive)
    app.router.add_delete("/api/theater/{session_id}/directives/{directive_id}", api_theater_remove_directive)
    app.router.add_post("/api/shutdown", api_shutdown)
    app.router.add_static("/static/", STATIC_DIR, show_index=False)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


async def _serve() -> None:
    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=HOST, port=PORT)
    await site.start()
    try:
        await app[RUNTIME_STATE]["shutdown_event"].wait()
    finally:
        await runner.cleanup()


def main() -> None:
    if not COMFY_ROOT.exists():
        raise SystemExit(f"ComfyUI was not found at {COMFY_ROOT}")
    LOGGER.info("Starting Wan Endless Theater on http://%s:%s", HOST, PORT)
    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
