# Wan Endless Learning Stream

A Windows-first, fully local interface for an endless narrated story theater powered by Wan 2.2, Gemma 4 E4B, and Supertonic 3. After the models are installed, generation uses no cloud API, account, analytics, or telemetry. The application is served directly at `http://127.0.0.1:7868/`.

This repository contains the application code only. Model weights, Wikipedia archives, generated media, and third-party runtimes are intentionally excluded.

## Features

- **Endless Offline Theater**: A resident story world that writes, speaks, moves, and archives itself at `http://127.0.0.1:7868/`.
- **Endless Dream Mode**: A one-word pre-sleep cue becomes an entirely invented, ungrounded stream of evolving imagery.
- **Interactive Character Show**: A stable resident host continues an activity, receives delayed viewer chat or decisions, and answers in later synchronized scenes.
- **Live World Direction**: Inject a one-scene event or a persistent future rule without replacing completed media.
- **Bilingual Language-Learning Playback**: Every story sentence is displayed and spoken first in the selected story language and then in the learner's translation language.
- **Progressive Playback**: Completed scenes appear while later scenes are still being planned and generated.
- **Optional Educational Grounding**: Local Kiwix ZIM archives provide verified facts for educational adventures.
- **Durable Archives**: Made from ordinary JSON, MP4, WAV, SRT, and M3U8 files.

## Tested hardware and performance

The reference system is Windows 11 with a Ryzen 9 7950X, 64 GB RAM, and an NVIDIA RTX 5070 12 GB.

For the measured end-to-end baseline, current bottleneck analysis, telemetry limitations, and a reproducible tuning procedure, see [`PERFORMANCE.md`](PERFORMANCE.md). The figures below describe this reference machine and are not universal performance guarantees.

- Gemma 4 E4B Q4_K_M: 15.31 output tokens/second and 146.73 prompt tokens/second at 8 CPU threads; about 4.62 GiB resident RAM.
- Live Theater planning: 13.29 output tokens/second with the video and speech services available.
- Supertonic 3: approximately 0.20 real-time factor in the measured narration test.
- **Cinema preview (`480 x 272 / 81 frames / 16 FPS`) is the user-tested Theater default on this PC.** Advanced custom generation exposes width, height, source frames, playback FPS, narration word limits, maximum slow-motion, and seed. Saved sessions created by older versions retain their original render settings.

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

## ComfyUI models

The backend uses ComfyUI's **Text to Video (Wan 2.2)** graph with:

- Wan 2.2 T2V high-noise and low-noise 14B FP8 models;
- matching LightX2V four-step LoRAs;
- UMT5 text encoder;
- Wan VAE.

Sampling stays at the validated blueprint values. The application changes only user-facing inputs such as prompt, resolution, frame count, FPS, and seed.

## Installation

```powershell
git clone <your-repository-url> Wan-endless-learning-stream
cd Wan-endless-learning-stream
python -m pip install -r requirements.txt
python app.py
```

On the tested layout, `Start Wan Video UI.cmd` uses ComfyUI's virtual-environment Python. `Launch Wan Video UI.vbs` starts the application hidden, waits for server readiness, and opens `http://127.0.0.1:7868`.

The launchers inherit `WAN_*` environment variables. `.env.example` is a reference file; the application does not silently load `.env` files.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `WAN_AI_ROOT` | `D:\AI` | Base AI application directory |
| `WAN_LOCAL_AI_ROOT` | `D:\LocalAI` | Base local-model directory |
| `WAN_COMFY_ROOT` | `D:\AI\ComfyUI` | ComfyUI installation |
| `WAN_GEMMA4_ROOT` | `D:\LocalAI\Gemma4E4B` | Gemma model and llama.cpp runtime |
| `WAN_LLAMA_RUNTIME_ROOT` | same as `WAN_GEMMA4_ROOT` | Optional separate llama.cpp runtime |
| `WAN_CUDA_LLAMA_RUNTIME_ROOT` | `D:\LocalAI\Bonsai27B` | Optional CUDA-enabled llama.cpp runtime used for bounded Gemma buffer bursts |
| `WAN_SUPERTONIC_ROOT` | `D:\LocalAI\Supertonic3` | Offline neural speech service |
| `WAN_KIWIX_ROOT` | `D:\LocalAI\OfflineWikipedia` | Optional educational sources |
| `WAN_COMFY_URL` | `http://127.0.0.1:8188` | ComfyUI HTTP/WebSocket endpoint |
| `WAN_OUTPUT_ROOT` | `<ComfyUI>\output` | Generated-project root |
| `WAN_HOST` | `127.0.0.1` | Application bind address |
| `WAN_PORT` | `7868` | Application port |
| `WAN_PYTHON_EXE` | ComfyUI venv Python | Visible CMD launcher only |
| `WAN_PYTHONW_EXE` | ComfyUI venv PythonW | Hidden VBS launcher only |

## Output directories

Under `WAN_OUTPUT_ROOT`:

- `wan_theater` contains saved endless-story sessions (playlists, video segments, narration WAVs, and archive JSON).

## Theater synchronization

Gemma planning, Supertonic speech, Wan rendering, and FFmpeg assembly overlap. Playback waits for synchronized media rather than speaking over a missing visual. The synchronizer first stretches unique forward motion; if narration remains longer, it uses forward/backward coverage instead of silently freezing most of the scene.

The live duration controller measures the interval between fully archived scenes, learns the actual seconds per spoken word and bilingual expansion ratio for the selected languages and voice, and targets 1.08× playback coverage inside the configured word budget. Gemma's sentence plan and a broad safe duration envelope are validated without retrying ordinary language-dependent word variation. Supertonic remains within a natural 0.96–1.05 speed range and changes pace only for the residual duration error; pacing is not randomized. The synchronization graph performs interpolation, forward/reverse coverage, encoding, and audio muxing in one FFmpeg pass without console window popups.

### Endless Dream

Choose **Endless dream** and enter a faint cue such as `velvet`, `warm rain`, or `blue door`. The cue is deliberately not treated as a request specification or factual topic. Gemma uses it once as a loose association, invents the people, places, objects, histories and internal rules, and begins inside an unfolding image rather than defining the word or announcing a dream.

### Live world direction

While a Theater session is open, **Direct the world while it runs** accepts either a one-scene event or a persistent world rule. The default **After planned buffer** timing reserves the first scene after Gemma's completed plans and any planning request already in flight. The optional **Next unrendered** timing is for people who prefer responsiveness over preserving speculative work.

### Interactive character show

Choose **Interactive character show** to create a recurring on-screen host, recognizable setting, and ongoing activity. Viewers can send chat messages to the host, inject one-scene events, or set persistent rules.

### Language-learning mode

Choose the language of the story under **Narration language**, then choose a different **Language-learning translation**. Gemma keeps the story itself in the first language and produces a validated one-to-one translation for every sentence. The Theater UI displays the paired sentences and Supertonic speaks each original sentence immediately followed by its translation.

Every UI page includes **Exit and release**. The localhost-only action interrupts queued ComfyUI work, marks active Theater work resumable, stops app-owned Gemma, Supertonic, Kiwix, and ComfyUI process trees, unloads ComfyUI models, and closes the local web server.

## Development

Run the checks:

```powershell
python -m py_compile app.py theater_pipeline.py test_prompt.py
python -m unittest -v test_prompt.py
```

Tests do not download models or require a GPU.

## Security and privacy

All services bind to loopback by default. There is no authentication layer; do not expose these ports directly to an untrusted network. The application forces Hugging Face and Transformers offline mode when it launches ComfyUI.
