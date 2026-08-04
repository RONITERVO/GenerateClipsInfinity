# Performance and Optimization Guide

This document describes how Wan Video Studio currently uses the reference computer, what has actually been measured, where the present bottleneck is, and how to evaluate future tuning work without overstating its value.

It is written for two audiences:

- developers and systems engineers who need enough detail to reproduce and challenge the measurements;
- users and contributors who want to understand why a component can be fully utilized without the whole application being poorly optimized.

This is a baseline, not a promise that every story, model revision, driver, or Windows installation will produce the same numbers. Measurements marked **observed** came from a live session. Proposed changes are marked **experiments** until they have been tested with the procedure in this document.

## Executive summary

The application is already optimized to run its major jobs concurrently:

- Gemma writes and translates on the CPU;
- Supertonic creates speech on the CPU;
- Wan creates video on the GPU;
- FFmpeg stretches, interpolates, encodes, and archives completed video on the CPU.

The current configuration favors long-term stability on a 12 GB GPU over maximum benchmark throughput. It keeps the GPU close to its safe memory limit during generation, deliberately leaves CPU capacity for speech and video assembly, and creates ordinary media and JSON files that can be recovered after interruption.

During the reference bilingual session, the system produced about **67 seconds of playable movie every 60 seconds of wall time**. This is approximately **1.12× real time**. It is enough to maintain an endless stream once the initial buffer exists, but it is not a large safety margin.

The captured steady-state bottleneck was **Gemma story planning plus sentence translation**, not video rendering, speech synthesis, storage, or total available memory. GPU utilization reached 100% during sampling but fell while the application waited for the next structured bilingual scene. The implementation now overlaps source planning and translation through two bounded Gemma slots; the baseline below predates that change and must not be presented as its measured result.

## Reference system

The measurements below were captured at 2026-08-04 14:00 EEST from the actively running application, not from an isolated synthetic benchmark.

| Component | Reference configuration |
| --- | --- |
| Operating system | Windows 11 Pro, build 10.0.26200 |
| CPU | AMD Ryzen 9 7950X, 16 physical cores / 32 logical processors |
| Memory | 64 GB installed |
| GPU | NVIDIA GeForce RTX 5070, 12,227 MiB VRAM |
| NVIDIA driver | 610.74 |
| Storage | Two healthy Corsair MP600 PRO LPX NVMe SSDs |
| Power plan | Best performance |
| PyTorch | 2.12.0+cu130 |
| CUDA runtime reported by PyTorch | 13.0 |
| GPU compute capability | 12.0 |
| ComfyUI revision | `822aca19836cd75c815631db23c3ad742d1f7d5e` (2026-06-12) |

PyTorch flash, memory-efficient, and math scaled-dot-product attention backends were enabled. SageAttention, xFormers, FlashAttention as a separate package, and Triton were not installed. Their absence is not automatically a defect: the native PyTorch path is working and third-party kernels must be proven compatible with this GPU, PyTorch, and CUDA combination.

## Measured workload

The active Theater session used these settings. It was captured before two-slot parallel translation and therefore remains a comparison baseline:

| Setting | Value |
| --- | --- |
| Session id | `20260804-133318-4d98` |
| Seed prompt | `Beaches` |
| World seed | `1900702449` |
| Experience | Pure story, adult audience |
| Story language | English |
| Translation language | Spanish |
| Voice | M2 |
| Resolution | 480 × 272 |
| Source frames | 81 |
| Playback rate | 16 FPS |
| Total spoken-word budget | 160–210 words per scene |
| Maximum slow motion | 8× |
| Wan workflow | Two-stage Wan 2.2 14B FP8 with LightX2V, four total sampling steps |

At the end of the primary capture the session had archived 26 scenes, containing 28.56 minutes of playable video. The planner was ahead of the renderer, so 28 sentence-translation records already existed. All succeeded on their first attempt. No OOM, generation failure, translation retry, or sustained disk queue was observed.

### Timing results

The following values summarize the most recent ten complete scenes unless otherwise noted.

| Measurement | Observed value | Interpretation |
| --- | ---: | --- |
| Playable duration per scene | 67.2–69.4 seconds average, depending on sample window | The amount of finished movie delivered to the viewer |
| Time between completed scenes | 59.8 seconds average | Best direct measure of sustained end-to-end throughput |
| True steady-state coverage | 1.12× real time | Approximately 12% more content produced than consumed |
| Complete planner cycle | 63.2 seconds average | Story planning plus validated translation |
| Translation stage | 18.3 seconds average | One-to-one sentence translation after story planning |
| Wan generation | 35.8 seconds average | ComfyUI submission through completed raw clip |
| Supertonic speech generation | 16.2 seconds average | About 4.3× faster than the resulting audio duration |
| FFmpeg assembly | 38.0 seconds average | Slow motion, interpolation, coverage, encoding, and muxing |
| Source words | 80.4 average | Original-language narration |
| Translation words | 85.8 average | Translated narration |
| Sentence pairs | 4.1 average | Original→translation units shown and spoken in order |

The bilingual word-budget adjustment is behaving as intended: approximately 166 combined spoken words were produced against a configured 160–210 total-word budget. Translation therefore adds learning content without simply doubling a monolingual scene of the same configured size.

## Resource observations

### GPU and VRAM

During a 12-second mixed loading and rendering sample:

- GPU utilization averaged 77.2% and reached 100%;
- VRAM usage averaged 8,986 MiB and peaked at 11,450 MiB;
- power averaged 187 W and peaked at 243 W against a 275 W limit;
- temperature peaked at 61°C;
- graphics clocks remained around 3.0 GHz.

There was no evidence of thermal throttling or a power-limit bottleneck. Utilization drops were associated with model initialization, VRAM movement, and transitions between workflow stages. The GPU is compute-bound during sampling but is not the continuous end-to-end limiter because it must sometimes wait for Gemma.

The ComfyUI log shows that each Wan noise stage prepares approximately 13.6 GB of model data for dynamic VRAM loading, which cannot be fully resident on a 12 GB card. The text encoder stages approximately 6.4 GB. Low-VRAM behavior is therefore necessary unless the workflow, model format, or GPU changes.

### CPU

During a writer-heavy phase, total CPU usage averaged approximately 29%. Gemma consumed roughly 7–8.6 core equivalents with its configured eight threads. During the busiest overlapping sample, Gemma, Supertonic, FFmpeg, and ComfyUI were active together:

- Gemma used approximately 7.4 core equivalents;
- Supertonic averaged approximately 8.2 core equivalents during its active burst;
- individual FFmpeg stages used approximately 1.2–3.9 core equivalents;
- ComfyUI CPU feeding was comparatively small once the GPU work began.

This is intentional headroom, not necessarily wasted CPU. Supertonic and FFmpeg need capacity while Gemma continues planning. Raising every thread count independently can make all components slower through cache contention and oversubscription.

All relevant processes currently run at normal priority with unrestricted processor affinity. No CCD-specific pinning is configured.

### Memory and storage

The writer-heavy sample had approximately 40.3 GB RAM available and 45% committed memory. The Gemma process used about 6.9 GB working set / 7.0 GB private memory. ComfyUI varied as models were staged and offloaded.

Disk active time averaged approximately 0.2%, with no queue. Storage is not a current bottleneck. Both application drives are healthy NVMe SSDs.

The Windows pagefile is only approximately 0.78 GB and was unused during the measurement. This does not reduce current speed, but it leaves limited commit-reserve protection if future configurations hold more models in RAM. Pagefile sizing should be treated as a resilience decision, not advertised as a performance improvement.

## How the pipeline overlaps work

The code deliberately separates four kinds of work:

| Worker | Main resource | Current concurrency behavior |
| --- | --- | --- |
| Story planner and translator | CPU and RAM | Separate bounded workers use up to two Gemma slots: translate scene N while planning N+1 |
| Supertonic | CPU and RAM | Starts speech generation at the same time as Wan |
| Wan/ComfyUI | GPU, VRAM, some CPU | One serialized workflow at a time |
| FFmpeg assembler | CPU | Processes the completed scene while the next raw clip can render |

The source-plan, translated-scene, and assembly queues are each bounded to prevent unlimited memory growth. The application also serializes ComfyUI submissions through a workflow lock, avoiding two Wan jobs competing for 12 GB of VRAM.

Important source locations:

- `build_prompt()` and `ComfyController._start_process()` in [`app.py`](app.py);
- `StoryRuntime`, `SupertonicRuntime`, and `TheaterManager` in [`theater_pipeline.py`](theater_pipeline.py);
- browser-side buffer calculation and telemetry in [`static/theater.html`](static/theater.html).

### Parallel translation groundwork

Cinema preview now defaults to 80–110 total spoken words per clip. In bilingual mode the 2.1× speech reservation converts this to approximately 39–52 source words, leaving room for the translated speech inside the same total budget. Planning prompts are clamped to the configured limits rather than expanding beyond them with the older ±25-word range.

The planner produces a source-language scene into a bounded queue. A second worker validates its sentence-aligned translation and places only render-ready scenes into another bounded queue. Meanwhile the planner can use the other llama.cpp slot for the next scene. Saved source plans and translated plans remain distinguishable, so interruption recovery does not repeat finished translations or send incomplete bilingual narration to TTS.

This is groundwork, not a throughput claim. Two simultaneous CPU generations may reduce each request's individual token rate while improving combined throughput and GPU feed continuity. Benchmark at least ten completed scenes and compare end-to-end completion timestamps, not just tokens per second.

The server allocates 32K context across two slots, preserving the earlier 16K ceiling per request. The planner still carries the story bible, rolling story summary, ten recent scene descriptors, and bounded grounding material because continuity may require them. `planner_prompt_tokens`, `translation_prompt_tokens`, stage elapsed time, and queue-depth metrics are saved so future work can identify context-growth slowdown before shortening continuity state. Prefer measured summary compaction over arbitrary truncation.

## Current deliberate tradeoffs

### CPU-only Gemma

Gemma runs with `-ngl 0`, eight CPU threads, two parallel request slots, a 32K shared context allocation that preserves 16K per slot, 512-token batches, 128-token microbatches, and memory mapping disabled. Keeping it off the GPU allows planning and translation to overlap Wan rendering. Moving it to the GPU would make each LLM request faster in isolation but would force model swapping or serialize it with video generation, so the total pipeline could become slower.

### CPU-only Supertonic

CUDA is explicitly disabled for Supertonic and ONNX Runtime. Speech finishes substantially before video in the measured workload, so spending scarce VRAM on TTS would not improve scene throughput.

### Conservative ComfyUI memory settings

ComfyUI launches with:

```text
--enable-dynamic-vram
--lowvram
--reserve-vram 1.5
--cache-none
--preview-method auto
--fast-disk
```

The first three settings protect desktop and driver stability while using models larger than physical VRAM. `--cache-none` and automatic previews are candidates for controlled experiments, but changing them is not a guaranteed improvement.

### High-quality CPU interpolation

FFmpeg uses motion-compensated interpolation and software H.264 encoding. This is intentionally more expensive than frame duplication or simple blending. It preserves more meaningful motion during long narration but makes assembly comparable in duration to Wan generation.

## Important telemetry limitation

The Theater UI's `coverage_ratio` and `production_ema` are useful short-term indicators, but they do not currently represent full end-to-end scene throughput.

`coverage_ratio` compares playable duration with the render/TTS readiness interval. It does not include all planner waiting and assembly time. During this capture the UI reported values around 1.5–2.5×, while timestamps between completed archived segments showed a true steady-state value of approximately 1.12×.

For optimization decisions, use completed-segment timestamps over at least ten scenes. UI coverage should not be quoted as an end-to-end benchmark until its definition changes.

## Prioritized experiments

The following items are proposals, not completed optimizations. Change one variable at a time and use the validation procedure below.

### Priority 1: Validate Gemma parallel throughput, threads, and CPU affinity

**Why:** Planning and translation were the measured bottleneck, and two-slot overlap changes CPU contention and per-request decoding speed.

Compare the former one-slot baseline with two slots over at least ten scenes, then test eight, ten, and twelve Gemma threads. Separately test leaving affinity unrestricted versus pinning Gemma to one Ryzen CCD. Cross-CCD scheduling can increase cache traffic, but Windows may already make better scheduling decisions than a fixed mask.

**Success condition:** lower complete planner-cycle time without increasing Wan, TTS, or FFmpeg times enough to reduce completed-scene throughput.

**Risk:** moderate. More LLM threads can interfere with Supertonic and FFmpeg during their overlapping burst.

### Priority 2: Bound Supertonic CPU parallelism

**Why:** Supertonic uses a large CPU burst but finishes well before the video. Some of that CPU capacity may be more valuable to Gemma.

First determine whether the installed Supertonic server exposes ONNX Runtime intra-op/inter-op controls. Benchmark a conservative cap rather than setting generic environment variables without confirming that the runtime honors them.

**Success condition:** planner time improves while TTS still completes before Wan.

**Risk:** low to moderate if a supported control exists; high if implemented through unsupported runtime hacks.

### Priority 3: ComfyUI node caching

**Why:** Logs show roughly 6–8 seconds of initialization for each Wan noise stage. The machine has substantial unused system RAM.

Compare `--cache-none` with the normal/classic cache or a small LRU cache while keeping low-VRAM mode and the 1.5 GB reservation unchanged. Watch whether identical loader/LoRA nodes are actually reused and whether initialization time falls. Do not assume a cache flag will keep 14B tensors resident in VRAM.

**Success condition:** lower Wan prompt time, unchanged visual output, no RAM growth across many scenes, and no CUDA OOM.

**Risk:** moderate. Caching can increase persistent RAM or retain objects that alter model-unloading behavior.

### Priority 4: Disable unused ComfyUI previews

**Why:** The application polls job state and does not consume ComfyUI preview images.

Benchmark `--preview-method none` against `auto`.

**Success condition:** repeatable prompt-time or CPU/GPU improvement without losing required progress reporting.

**Risk:** low.

### Lower-priority experiments

- Hardware H.264 encoding may reduce FFmpeg CPU usage, but interpolation remains CPU-bound and the planner is currently slower than assembly.
- Raising FFmpeg above four threads may shorten some stages but can starve Gemma or Supertonic.
- A cheaper interpolation mode can materially reduce assembly time, but this is a quality tradeoff rather than a free optimization.
- Lowering the VRAM reservation can increase the resident working set but has a real OOM and desktop-stability risk; peak usage is already near capacity.

## Deep architectural options

These require code or model-architecture changes and should not be presented as quick tuning:

- run a separate small translation runtime so story planning does not wait for translation;
- request story and aligned translation in one Gemma completion, accepting a larger validation and repair burden;
- dynamically move Gemma between CPU and GPU during Wan-idle periods, accepting model-transfer cost and more complex scheduling;
- replace the two-model Wan workflow with a model or quantization designed to remain resident on 12 GB VRAM;
- adopt third-party attention kernels after verifying Blackwell, CUDA 13, PyTorch 2.12, and model-quality compatibility.

The current separate translation stage exists for a reason: it permits strict sentence-count and id validation. Combining stages may save time but can make silent misalignment more likely. Any such change must retain fail-closed validation and archive compatibility.

## Perceptual performance and motion coverage

Performance is not only tokens or frames per second. A fast system can still feel broken if narration has no matching motion.

At 81 frames and 16 FPS, the raw clip is approximately 5.06 seconds. The configured 8× slow-motion limit provides about 40.5 seconds of unique forward motion. Recent bilingual narration averaged about 69 seconds, so all ten inspected scenes needed some forward/backward coverage. Their average estimated cycle value was below one, meaning they normally used one forward pass and only part of the reverse pass rather than repeating several full loops.

If avoiding reverse motion is more important than narration depth, reduce the total spoken-word budget until audio approaches 40 seconds. Increasing the budget improves language exposure but necessarily requires more motion coverage unless more source frames or additional clips are generated.

## Reproducible evaluation procedure

Do not evaluate a performance change from one clip or from Task Manager alone.

1. Record the Git commit, ComfyUI revision, model filenames and hashes, driver, PyTorch/CUDA versions, power plan, and exact Theater configuration.
2. Use the same prompt, seed, languages, voice, resolution, frames, FPS, word limits, and slow-motion limit.
3. Start from the same warm or cold state. Label it explicitly; model startup and steady state answer different questions.
4. Generate at least twelve complete scenes. Exclude the first two from steady-state averages.
5. Record completed-segment timestamps, playable duration, Wan seconds, TTS seconds, assembly seconds, planner and translation times, retries, VRAM peak, and any OOM or process restart.
6. Inspect several resulting scenes for pronunciation, alignment, continuity, motion quality, and audio/video synchronization. A faster invalid result is a regression.
7. Change only one variable. Repeat at least twice if the difference is smaller than 10%.
8. Revert unsuccessful experiments completely, including environment variables and launch flags.

Useful read-only commands:

```powershell
# Current GPU state
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw,pstate --format=csv

# Current Theater state
Invoke-RestMethod http://127.0.0.1:7868/api/theater

# Active service commands and owners
Get-CimInstance Win32_Process |
  Where-Object { $_.Name -match 'python|llama|supertonic|ffmpeg' } |
  Select-Object ProcessId, ParentProcessId, Name, CommandLine

# Power plan
powercfg /getactivescheme
```

Session-level raw data is stored below:

```text
<WAN_OUTPUT_ROOT>\wan_theater\<session-id>\session.json
<WAN_OUTPUT_ROOT>\wan_theater\<session-id>\archive.json
<WAN_OUTPUT_ROOT>\wan_theater\<session-id>\logs\
```

## Acceptance criteria for a real optimization

An optimization should be accepted only when all applicable conditions hold:

- completed-scene wall interval improves consistently;
- true steady-state coverage does not fall below 1.0×;
- first-scene latency does not regress unacceptably;
- no new CUDA OOM, translation retry, failed scene, or unbounded memory growth appears;
- narration remains sentence-aligned and correctly ordered;
- TTS still finishes before its matching visual or does not delay playback;
- visual quality, character continuity, and motion coverage remain acceptable;
- saved sessions remain resumable and ordinary archived files remain valid.

CPU utilization, GPU utilization, token rate, or one component's isolated benchmark is not sufficient by itself.

## Suggested report template

Future tuning reports should include:

```text
Date and commit:
Hardware / driver / PyTorch / CUDA:
Model files and hashes:
Exact launch flags:
Story configuration:
Warm or cold run:
Scenes measured:

Average completed-scene interval:
Average playable duration:
True steady-state coverage:
Planner time:
Translation time:
Wan time:
TTS time:
Assembly time:
Peak VRAM / RAM:
Retries / failures / OOMs:

Visual and audio observations:
Single variable changed:
Conclusion and rollback instructions:
```

This format keeps performance work auditable. It also prevents a local improvement in one component from being marketed as an application-wide speedup when another stage remains the real limiter.
