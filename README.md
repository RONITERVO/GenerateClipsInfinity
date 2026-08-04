# Wan Video Studio

A Windows-first, fully local interface for Wan 2.2 video generation, one-sentence movie production, and an endless narrated story theater. After the models are installed, generation uses no cloud API, account, analytics, or telemetry.

This repository contains the application code only. Model weights, Wikipedia archives, generated media, and third-party runtimes are intentionally excluded.

## Features

- Single Wan 2.2 text-to-video clips with the tested two-stage LightX2V workflow.
- One-sentence movie planning, progressive generation, editable per-clip timing, subtitles, EDL, CSV shot list, and final MP4 export.
- Endless offline theater with Gemma 4 E4B writing on CPU, Supertonic narration on CPU, and Wan rendering on the GPU.
- Optional bilingual language-learning playback: every story sentence is displayed and spoken first in the selected story language and then in the learner's translation language.
- Progressive playback: completed scenes appear while later scenes are still being planned and generated.
- Optional fail-closed educational grounding from local Kiwix ZIM archives.
- Durable projects made from ordinary JSON, MP4, WAV, SRT, CSV, EDL, and M3U8 files.

## Tested hardware and performance

The reference system is Windows 11 with a Ryzen 9 7950X, 64 GB RAM, and an NVIDIA RTX 5070 12 GB.

For the measured end-to-end baseline, current bottleneck analysis, telemetry limitations, and a reproducible tuning procedure, see [`PERFORMANCE.md`](PERFORMANCE.md). The figures below describe this reference machine and are not universal performance guarantees.

- Gemma 4 E4B Q4_K_M: 15.31 output tokens/second and 146.73 prompt tokens/second at 8 CPU threads; about 4.62 GiB resident RAM.
- Live Theater planning: 13.29 output tokens/second with the video and speech services available.
- Supertonic 3: approximately 0.20 real-time factor in the measured narration test.
- **Cinema preview (`480 x 272 / 81 frames / 16 FPS`) is the user-tested Theater default on this PC.** There is no preset selector. Advanced custom generation exposes width, height, source frames, playback FPS, narration word limits, maximum slow-motion, and seed. Saved sessions created by older versions retain their original render settings.

## Required local components

The default paths match the tested machine. Every path can be overridden with environment variables.

### Core application

- Windows 10 or 11.
- Python 3.13 with the packages in `requirements.txt`.
- FFmpeg and FFprobe on `PATH`.
- A working ComfyUI installation and NVIDIA/PyTorch environment.

### Endless Theater

Gemma 4 E4B is the only supported story writer. The application intentionally has no smaller-model fallback.

```text
D:\LocalAI\Gemma4E4B\
├── models\
│   └── gemma-4-E4B-it-Q4_K_M.gguf
└── runtime\
    ├── llama-server.exe
    └── llama.cpp DLLs
```

The required instruction-tuned model is derived from [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it). The tested GGUF filename is available from [`unsloth/gemma-4-E4B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF). The tested file SHA-256 is:

```text
85a896a047553e842f25297ee5b031d64ff30147d9c4af17b1e4b394cd1fab87
```

Supertonic 3 is expected at `D:\LocalAI\Supertonic3`. Pure story mode needs Gemma, Supertonic, ComfyUI, Wan, and FFmpeg. Educational modes additionally require Kiwix tools and English Simple Wikipedia or Finnish Wikipedia ZIM archives under `D:\LocalAI\OfflineWikipedia\archives`.

### Movie editor

The one-sentence movie planner expects Bonsai 27B at `D:\LocalAI\Bonsai27B`. Narration uses the installed ComfyUI `ChatterboxTTS` node. These are not required for single clips or Pure story Theater.

## ComfyUI models

The backend uses ComfyUI's **Text to Video (Wan 2.2)** graph with:

- Wan 2.2 T2V high-noise and low-noise 14B FP8 models;
- matching LightX2V four-step LoRAs;
- UMT5 text encoder;
- Wan VAE.

Sampling stays at the validated blueprint values. The application changes only user-facing inputs such as prompt, resolution, frame count, FPS, and seed.

## Installation

```powershell
git clone <your-repository-url> Wan-Video-UI
cd Wan-Video-UI
python -m pip install -r requirements.txt
python app.py
```

On the tested layout, `Start Wan Video UI.cmd` uses ComfyUI's virtual-environment Python. `Launch Wan Video UI.vbs` starts the application hidden and opens `http://127.0.0.1:7868`.

The launchers inherit `WAN_*` environment variables. `.env.example` is a reference file; the application does not silently load `.env` files.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `WAN_AI_ROOT` | `D:\AI` | Base AI application directory |
| `WAN_LOCAL_AI_ROOT` | `D:\LocalAI` | Base local-model directory |
| `WAN_COMFY_ROOT` | `D:\AI\ComfyUI` | ComfyUI installation |
| `WAN_GEMMA4_ROOT` | `D:\LocalAI\Gemma4E4B` | Gemma model and llama.cpp runtime |
| `WAN_LLAMA_RUNTIME_ROOT` | same as `WAN_GEMMA4_ROOT` | Optional separate llama.cpp runtime |
| `WAN_SUPERTONIC_ROOT` | `D:\LocalAI\Supertonic3` | Offline neural speech service |
| `WAN_KIWIX_ROOT` | `D:\LocalAI\OfflineWikipedia` | Optional educational sources |
| `WAN_BONSAI_ROOT` | `D:\LocalAI\Bonsai27B` | Optional movie planner |
| `WAN_COMFY_URL` | `http://127.0.0.1:8188` | ComfyUI HTTP/WebSocket endpoint |
| `WAN_OUTPUT_ROOT` | `<ComfyUI>\output` | Generated-project root |
| `WAN_HOST` | `127.0.0.1` | Application bind address |
| `WAN_PORT` | `7868` | Application port |
| `WAN_PYTHON_EXE` | ComfyUI venv Python | Visible CMD launcher only |
| `WAN_PYTHONW_EXE` | ComfyUI venv PythonW | Hidden VBS launcher only |

Example for the current PowerShell session:

```powershell
$env:WAN_COMFY_ROOT = "E:\ComfyUI"
$env:WAN_GEMMA4_ROOT = "E:\Models\Gemma4E4B"
python app.py
```

## Output directories

Under `WAN_OUTPUT_ROOT`:

- `wan_ui` contains single clips;
- `wan_movies` contains editable movie projects;
- `wan_theater` contains saved endless-story sessions.

Movie projects retain the original request, screenplay plan, versioned `project.json`, source shots, narration, logs, rendered segments, final MP4, subtitles, shot-list CSV, and EDL. Rebuilding an edit reuses source assets and does not rerun Wan.

## Theater synchronization

Gemma planning, Supertonic speech, Wan rendering, and FFmpeg assembly overlap. Playback waits for synchronized media rather than speaking over a missing visual. The editor first stretches unique forward motion; if narration remains longer, it uses forward/backward coverage instead of silently freezing most of the scene.

The live duration controller measures the interval between fully archived scenes, learns the actual seconds per spoken word and bilingual expansion ratio for the selected languages and voice, and targets 1.08× playback coverage inside the configured word budget. Gemma's sentence plan and a broad safe duration envelope are validated without retrying ordinary language-dependent word variation. Supertonic remains within a natural 0.96–1.05 speed range and changes pace only for the residual duration error; pacing is not randomized. The synchronization graph performs interpolation, forward/reverse coverage, encoding, and audio muxing in one FFmpeg pass, avoiding repeated lossy H.264 encodes.

The writer stores a structured cast, world, visual style, binding premise contract, and continuity rules. Explicit character counts, objects, actions, threats, and settings in the user's prompt are treated as non-negotiable.

### Language-learning mode

Choose the language of the story under **Narration language**, then choose a different **Language-learning translation**. Gemma keeps the story itself in the first language and produces a validated one-to-one translation for every sentence. The Theater UI displays the paired sentences and Supertonic speaks each original sentence immediately followed by its translation.

The Cinema preview default is 80–110 total spoken words per clip. The Advanced minimum and maximum values use the same total-spoken-word definition. When translation is enabled, the planner reduces source prose before generation, then learns the measured source-to-translation expansion ratio. Completed-scene cadence and actual narration duration select the required part of the budget. The sentence count is validated, gross duration-envelope misses use the existing bounded repair path, and normal language-dependent word variation is preserved. This keeps bilingual scenes close to the generation cadence instead of doubling them or disguising a short scene with arbitrarily slow speech.

Gemma uses two bounded CPU request slots: one can plan the next source scene while the other produces the current scene's validated sentence translations. Each slot retains a 16K context ceiling. The separate translation gate still fails closed on missing, reordered, combined, or split sentences; concurrency does not weaken archive or TTS alignment.

Sentence pairs, language codes, translated titles, the complete configuration, word counts, model timing/context, and GPU-feed wait are stored in the ordinary session and archive JSON. Version 1 sessions without bilingual fields remain playable and resumable.

## Development

Run the same checks used by GitHub Actions:

```powershell
python -m py_compile app.py movie_pipeline.py theater_pipeline.py test_prompt.py
python -m unittest -v test_prompt.py
```

Tests do not download models or require a GPU. See `CONTRIBUTING.md` and `SECURITY.md` before opening a pull request.

## Security and privacy

All services bind to loopback by default. There is no authentication layer; do not expose these ports directly to an untrusted network. The application forces Hugging Face and Transformers offline mode when it launches ComfyUI.

## Troubleshooting

- UI log: `logs/wan-video-ui.log`
- ComfyUI logs: `logs/comfyui.out.log` and `logs/comfyui.err.log`
- Story-session writer logs: `<output>/wan_theater/<session>/logs/`
- If Theater reports a missing writer, verify both the exact GGUF filename and `runtime/llama-server.exe` beneath `WAN_GEMMA4_ROOT`.
- If FFmpeg is not found, add both `ffmpeg.exe` and `ffprobe.exe` to `PATH`.
