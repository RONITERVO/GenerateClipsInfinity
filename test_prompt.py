import asyncio
import json
import os
import subprocess
import tempfile
import time
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from app import (
    _is_local_exit_request, api_theater_live_directive, build_prompt, create_app,
    validate_theater_payload,
)
from theater_pipeline import (
    GpuReleaseError, StoryRuntime, SupertonicRuntime, TheaterError, TheaterManager, split_narration_sentences,
    spoken_word_count,
)


class PromptTests(unittest.TestCase):
    def test_exit_endpoint_requires_loopback_and_explicit_same_origin_header(self):
        class Request:
            def __init__(self, remote, header):
                self.remote = remote
                self.headers = {"X-Wan-Local-Exit": header} if header else {}

        self.assertTrue(_is_local_exit_request(Request("127.0.0.1", "release-owned-resources")))
        self.assertTrue(_is_local_exit_request(Request("::1", "release-owned-resources")))
        self.assertFalse(_is_local_exit_request(Request("127.0.0.1", "")))
        self.assertFalse(_is_local_exit_request(Request("192.168.1.20", "release-owned-resources")))

    def test_live_directive_api_rejects_non_object_json(self):
        async def exercise(payload):
            class Request:
                match_info = {"session_id": "session"}

                async def json(self):
                    return payload

            response = await api_theater_live_directive(Request())
            return response.status, json.loads(response.text)

        for payload in (None, []):
            with self.subTest(payload=payload):
                status, body = asyncio.run(exercise(payload))
                self.assertEqual(status, 400)
                self.assertEqual(body["error"], "Request body must be a JSON object.")

    def test_theater_retries_one_externally_interrupted_ffmpeg_run(self):
        class Process:
            def __init__(self, code):
                self.code = code
            async def wait(self):
                return self.code

        async def exercise(log_path):
            manager = TheaterManager.__new__(TheaterManager)
            processes = [Process(0xC000013A), Process(0)]
            with patch(
                "theater_pipeline.asyncio.create_subprocess_exec", side_effect=processes,
            ) as create:
                await manager._run_ffmpeg(["ffmpeg", "-version"], log_path)
                return create.call_count

        with tempfile.TemporaryDirectory() as directory:
            calls = asyncio.run(exercise(Path(directory) / "ffmpeg.log"))
        self.assertEqual(calls, 2)

    def test_theater_does_not_retry_a_genuine_ffmpeg_error(self):
        class Process:
            async def wait(self):
                return 1

        async def exercise(log_path):
            manager = TheaterManager.__new__(TheaterManager)
            with patch(
                "theater_pipeline.asyncio.create_subprocess_exec", return_value=Process(),
            ) as create:
                with self.assertRaisesRegex(TheaterError, "exit 1"):
                    await manager._run_ffmpeg(["ffmpeg", "-version"], log_path)
                return create.call_count

        with tempfile.TemporaryDirectory() as directory:
            calls = asyncio.run(exercise(Path(directory) / "ffmpeg.log"))
        self.assertEqual(calls, 1)

    def test_theater_ffmpeg_and_duration_pass_no_window_creationflags(self):
        class Process:
            def __init__(self, output=b"1.234\n"):
                self.output = output
                self.returncode = 0
            async def wait(self):
                return 0
            async def communicate(self):
                return self.output, b""

        async def exercise_ffmpeg(log_path):
            manager = TheaterManager.__new__(TheaterManager)
            with patch(
                "theater_pipeline.asyncio.create_subprocess_exec", return_value=Process(),
            ) as create:
                await manager._run_ffmpeg(["ffmpeg", "-version"], log_path)
                return create.call_args

        async def exercise_duration():
            manager = TheaterManager.__new__(TheaterManager)
            with patch(
                "theater_pipeline.asyncio.create_subprocess_exec", return_value=Process(),
            ) as create, patch("theater_pipeline.shutil.which", return_value="ffprobe"):
                duration = await manager._duration(Path("dummy.mp4"))
                return duration, create.call_args

        with tempfile.TemporaryDirectory() as directory:
            call_args = asyncio.run(exercise_ffmpeg(Path(directory) / "ffmpeg.log"))
            expected_flag = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            self.assertEqual(call_args.kwargs.get("creationflags"), expected_flag)

        duration, dur_call_args = asyncio.run(exercise_duration())
        self.assertEqual(duration, 1.234)
        self.assertEqual(dur_call_args.kwargs.get("creationflags"), expected_flag)

    def test_blueprint_graph_and_output_node(self):
        config = {
            "prompt": "A test scene",
            "negative": "blurry",
            "width": 480,
            "height": 272,
            "frames": 17,
            "fps": 16,
            "seed": 42,
            "filename_prefix": "wan_theater_test",
        }
        graph = build_prompt(config)
        self.assertEqual(graph["8"]["inputs"]["end_at_step"], 2)
        self.assertEqual(graph["12"]["inputs"]["start_at_step"], 2)
        self.assertEqual(graph["16"]["class_type"], "SaveVideo")
        self.assertEqual(graph["4"]["inputs"]["width"], 480)
        self.assertEqual(graph["4"]["inputs"]["length"], 17)

    def test_theater_invalid_frame_rule_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "4n\\+1"):
            validate_theater_payload({
                "prompt": "test",
                "quality_settings": {"width": 480, "height": 272, "frames": 18, "fps": 16},
            })

    def test_legacy_routes_redirect_to_root(self):
        app = create_app()
        routes = [r.resource.canonical for r in app.router.routes() if r.method == "GET"]
        self.assertIn("/", routes)
        self.assertIn("/theater", routes)
        self.assertIn("/movie", routes)

    def test_theater_sync_uses_one_encode_graph_for_slow_and_repeated_motion(self):
        quality = {"frames": 81, "fps": 16, "max_slow": 8.0}
        slow_graph, slow_duration, repeated = TheaterManager._visual_sync_filter(5.062, 35.0, quality)
        self.assertFalse(repeated)
        self.assertEqual(slow_duration, 35.0)
        self.assertIn("minterpolate=fps=16", slow_graph)
        self.assertNotIn("reverse", slow_graph)

        loop_graph, slow_duration, repeated = TheaterManager._visual_sync_filter(5.062, 50.0, quality)
        self.assertTrue(repeated)
        self.assertAlmostEqual(slow_duration, 40.496, places=3)
        self.assertIn("reverse", loop_graph)
        self.assertIn("loop=loop=-1", loop_graph)
        self.assertIn("trim=duration=50.000000", loop_graph)

    def test_theater_defaults_to_tested_cinema_preview(self):
        config = validate_theater_payload({"prompt": "Teach astronomy through an adventure."})
        self.assertEqual(config["quality"], "custom")
        self.assertEqual(config["quality_settings"], TheaterManager.CINEMA_DEFAULTS)
        self.assertEqual(config["mode"], "edutainment")
        self.assertEqual(config["audience"], "family")
        self.assertEqual(config["voice"], "M1")
        self.assertEqual(config["language"], "en")
        self.assertEqual(config["translation_language"], "")
        self.assertEqual(config["quality_settings"]["min_words"], 80)
        self.assertEqual(config["quality_settings"]["max_words"], 110)
        self.assertEqual(config["context_compaction_scenes"], 30)
        self.assertGreaterEqual(config["seed"], 0)

    def test_dream_mode_accepts_a_minimal_seed_without_grounding_or_learning_focus(self):
        manager = TheaterManager.__new__(TheaterManager)
        config = validate_theater_payload({
            "prompt": "Velvet", "mode": "dream", "learning_focus": "History of textiles",
            "translation_language": "fi",
        })
        prompt = manager._system_prompt(config)
        self.assertEqual(config["prompt"], "Velvet")
        self.assertEqual(config["learning_focus"], "")
        self.assertEqual(config["translation_language"], "fi")
        self.assertFalse(manager.uses_grounding(config))
        self.assertIn("faint associative spark", prompt)
        self.assertIn("Invent every person, place, object, history, rule and relationship", prompt)
        self.assertNotIn("seed as a binding premise contract", prompt)
        self.assertNotIn("offline encyclopedia excerpts", prompt)

    def test_context_compaction_interval_is_advanced_and_bounded(self):
        config = validate_theater_payload({"prompt": "A story", "context_compaction_scenes": 45})
        self.assertEqual(config["context_compaction_scenes"], 45)
        self.assertEqual(
            validate_theater_payload({"prompt": "A story", "context_compaction_scenes": 0})[
                "context_compaction_scenes"
            ],
            0,
        )
        with self.assertRaisesRegex(ValueError, "0 .* 5 and 200"):
            validate_theater_payload({"prompt": "A story", "context_compaction_scenes": 2})

    def test_theater_accepts_distinct_offline_translation_language(self):
        config = validate_theater_payload({
            "prompt": "A forest mystery", "language": "fi", "translation_language": "en",
        })
        self.assertEqual(config["language"], "fi")
        self.assertEqual(config["translation_language"], "en")
        with self.assertRaisesRegex(ValueError, "must differ"):
            validate_theater_payload({"prompt": "A story", "language": "fi", "translation_language": "fi"})
        with self.assertRaisesRegex(ValueError, "supported translation"):
            validate_theater_payload({"prompt": "A story", "translation_language": "na"})

    def test_bilingual_word_budget_reduces_source_prose(self):
        config = validate_theater_payload({"prompt": "A story", "translation_language": "fi"})
        minimum, maximum = TheaterManager.narration_word_limits(config)
        self.assertEqual((minimum, maximum), (39, 52))
        self.assertLess(minimum, config["quality_settings"]["min_words"])
        self.assertLess(maximum, config["quality_settings"]["max_words"])

    def test_offline_sentence_splitter_keeps_closing_quotes_and_cjk_boundaries(self):
        text = 'She said, "Run now!" Then they crossed the bridge. \u732b\u306f\u8d70\u3063\u305f\u3002\u6708\u304c\u51fa\u305f\uff01'
        self.assertEqual(split_narration_sentences(text), [
            'She said, "Run now!"', "Then they crossed the bridge.", "\u732b\u306f\u8d70\u3063\u305f\u3002", "\u6708\u304c\u51fa\u305f\uff01",
        ])
        self.assertEqual(
            split_narration_sentences("\u300c\u884c\u3053\u3046\u3002\u300d\u6b21\u3078\u3002"),
            ["\u300c\u884c\u3053\u3046\u3002\u300d", "\u6b21\u3078\u3002"],
        )
        self.assertEqual(
            split_narration_sentences("\u300e\u5f85\u3063\u3066\uff01\u300f\u7d42\u308f\u308a\u3002"),
            ["\u300e\u5f85\u3063\u3066\uff01\u300f", "\u7d42\u308f\u308a\u3002"],
        )

    def test_spoken_word_count_handles_unspaced_japanese(self):
        self.assertEqual(spoken_word_count("\u300c\u884c\u3053\u3046\u3002\u300d\u6b21\u3078\u3002", "ja"), 3)
        self.assertEqual(spoken_word_count("Let's follow the moon-lit path.", "en"), 5)

    def test_translation_stage_preserves_one_to_one_sentence_alignment(self):
        async def exercise(root: Path):
            manager = TheaterManager.__new__(TheaterManager)
            manager.root = root
            (root / "session" / "logs").mkdir(parents=True)
            manager._save = lambda _state: None
            requests = []

            class Writer:
                async def complete(self, messages, max_tokens=900):
                    requests.append(messages[-1]["content"])
                    return (
                        '{"title_translation":"Departure","sentences":['
                        '{"id":1,"translation":"Let us go."},'
                        '{"id":2,"translation":"Next."}]}',
                        {"tokens_per_second": 12.5, "elapsed_seconds": 0.4, "prompt_tokens": 96},
                    )

            manager.writer = Writer()
            state = {
                "id": "session", "config": {"language": "ja", "translation_language": "en"},
                "metrics": {},
            }
            scene = {
                "number": 2, "title": "\u51fa\u767a",
                "narration": "\u300c\u884c\u3053\u3046\u3002\u300d\u6b21\u3078\u3002",
            }
            return await manager._prepare_narration(state, scene), requests

        with tempfile.TemporaryDirectory() as directory:
            result, requests = asyncio.run(exercise(Path(directory)))
        originals = ["\u300c\u884c\u3053\u3046\u3002\u300d", "\u6b21\u3078\u3002"]
        self.assertEqual(result["translated_title"], "Departure")
        self.assertEqual(len(result["narration_sentences"]), 2)
        self.assertEqual([pair["original"] for pair in result["narration_sentences"]], originals)
        self.assertEqual(result["source_word_count"], 3)
        self.assertEqual(result["translation_word_count"], 4)
        self.assertEqual(result["total_spoken_words"], 7)
        self.assertEqual(result["translation_metrics"]["prompt_tokens"], 96)
        self.assertIn(originals[0], requests[0])
        self.assertIn(originals[1], requests[0])

    def test_wav_concatenation_preserves_all_sentence_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parts = [root / "one.wav", root / "two.wav"]
            frame_counts = [80, 120]
            for path, count in zip(parts, frame_counts):
                with wave.open(str(path), "wb") as output:
                    output.setnchannels(1)
                    output.setsampwidth(2)
                    output.setframerate(8000)
                    output.writeframes(b"\x00\x00" * count)
            combined = root / "combined.wav"
            SupertonicRuntime._concatenate_wavs(parts, combined)
            with wave.open(str(combined), "rb") as result:
                self.assertEqual(result.getnframes(), sum(frame_counts))
                self.assertEqual(result.getframerate(), 8000)

    def test_archived_scene_keeps_pipeline_metrics(self):
        async def exercise(root: Path):
            manager = TheaterManager.__new__(TheaterManager)
            manager.root = root / "wan_theater"
            manager.output_root = root
            manager._save = lambda _state: None

            async def synchronize(_state, _scene, _video, _audio):
                return root / "wan_theater" / "session" / "segments" / "scene_00002.mp4", {
                    "raw_video_duration": 5.0, "duration": 30.0, "slow_duration": 30.0,
                    "motion_repeated": False, "estimated_motion_cycles": 1.0,
                }

            manager._synchronize = synchronize
            scene = {
                "number": 2, "title": "Path", "beat": "Departure", "narration": "They leave.",
                "visual_action": "They cross a bridge.", "asset_fingerprint": "abc123",
                "source_word_count": 2, "translation_word_count": 3, "total_spoken_words": 5,
                "planner_metrics": {"elapsed_seconds": 10.0, "prompt_tokens": 900},
                "translation_metrics": {"elapsed_seconds": 4.0, "prompt_tokens": 90},
                "gpu_feed_wait_seconds": 0.25,
            }
            state = {
                "id": "session", "config": {"language": "en"}, "segments": [],
                "metrics": {"production_ema": 0.0}, "rendering_scene": 3,
            }
            await manager._assemble_scene(state, {
                "scene": scene, "video_rel": "raw.mp4", "audio_rel": "audio.wav",
                "audio_path": root / "audio.wav", "video_seconds": 20.0, "tts_seconds": 3.0,
                "ready_seconds": 20.0, "cycle_started": time.perf_counter() - 20.0,
            })
            return state["segments"][0]

        with tempfile.TemporaryDirectory() as directory:
            entry = asyncio.run(exercise(Path(directory)))
        self.assertEqual(entry["total_spoken_words"], 5)
        self.assertEqual(entry["planner_metrics"]["prompt_tokens"], 900)
        self.assertEqual(entry["translation_metrics"]["elapsed_seconds"], 4.0)
        self.assertEqual(entry["gpu_feed_wait_seconds"], 0.25)

    def test_durable_archive_keeps_compacted_continuity_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = TheaterManager.__new__(TheaterManager)
            manager.root = root / "wan_theater"
            (manager.root / "session").mkdir(parents=True)
            state = {
                "id": "session", "title": "Story", "status": "running",
                "config": {"prompt": "A durable story", "context_compaction_scenes": 30},
                "bible": {"world": "harbor"}, "grounding": None, "segments": [],
                "story_summary": "Mira guards the gate.",
                "continuity_memory": {"active_threads": ["Find the key"]},
                "context_compacted_through_scene": 30,
                "live_directives": [{"id": "rain", "scope": "persistent", "status": "active", "text": "It rains."}],
                "metrics": {"context_compaction_events": [{"through_scene": 30}]},
            }
            manager._write_playlist(state)
            archive = json.loads((manager.root / "session" / "archive.json").read_text(encoding="utf-8"))
        self.assertEqual(archive["context_compacted_through_scene"], 30)
        self.assertEqual(archive["continuity_memory"]["active_threads"], ["Find the key"])
        self.assertEqual(archive["context_compaction_events"][0]["through_scene"], 30)
        self.assertEqual(archive["live_directives"][0]["id"], "rain")

    def test_coverage_uses_completed_segment_cadence(self):
        async def exercise(root: Path):
            manager = TheaterManager.__new__(TheaterManager)
            manager.root = root / "wan_theater"
            manager.output_root = root
            manager._save = lambda _state: None

            async def synchronize(_state, _scene, _video, _audio):
                return root / "wan_theater" / "session" / "segments" / "scene_00002.mp4", {
                    "raw_video_duration": 5.0, "duration": 40.0, "slow_duration": 40.0,
                    "motion_repeated": False, "estimated_motion_cycles": 1.0,
                }

            manager._synchronize = synchronize
            now = time.time()
            state = {
                "id": "session", "config": {"language": "en"},
                "segments": [{"number": 1, "created": now - 50.0, "duration": 35.0}],
                "metrics": {"production_ema": 0.0}, "rendering_scene": 3,
            }
            scene = {
                "number": 2, "title": "Path", "beat": "Departure", "narration": "They leave.",
                "visual_action": "They cross a bridge.", "asset_fingerprint": "abc123",
                "source_word_count": 80, "translation_word_count": 0, "total_spoken_words": 80,
            }
            await manager._assemble_scene(state, {
                "scene": scene, "video_rel": "raw.mp4", "audio_rel": "audio.wav",
                "audio_path": root / "audio.wav", "video_seconds": 20.0, "tts_seconds": 3.0,
                "ready_seconds": 20.0, "cycle_started": time.perf_counter() - 20.0,
            })
            return state["metrics"]

        with tempfile.TemporaryDirectory() as directory:
            metrics = asyncio.run(exercise(Path(directory)))
        self.assertAlmostEqual(metrics["completion_interval_ema"], 50.0, delta=0.1)
        self.assertAlmostEqual(metrics["coverage_ratio"], 0.8, delta=0.01)
        self.assertEqual(metrics["speech_seconds_per_word_ema"], 0.5)

    def test_resumed_session_excludes_offline_gap_from_cadence(self):
        async def exercise(root: Path):
            manager = TheaterManager.__new__(TheaterManager)
            manager.root = root / "wan_theater"
            manager.output_root = root
            manager._save = lambda _state: None

            async def synchronize(_state, _scene, _video, _audio):
                return root / "wan_theater" / "session" / "segments" / "scene_00002.mp4", {
                    "raw_video_duration": 5.0, "duration": 20.0, "slow_duration": 20.0,
                    "motion_repeated": False, "estimated_motion_cycles": 1.0,
                }

            manager._synchronize = synchronize
            now = time.time()
            state = {
                "id": "session", "config": {"language": "en"},
                "segments": [{"number": 1, "created": now - 3600.0, "duration": 20.0}],
                "metrics": {"production_ema": 0.0, "run_started_at": now - 30.0},
            }
            scene = {
                "number": 2, "title": "Path", "beat": "Departure", "narration": "They leave.",
                "visual_action": "They cross a bridge.", "asset_fingerprint": "abc123",
                "source_word_count": 40, "translation_word_count": 0, "total_spoken_words": 40,
            }
            await manager._assemble_scene(state, {
                "scene": scene, "video_rel": "raw.mp4", "audio_rel": "audio.wav",
                "audio_path": root / "audio.wav", "video_seconds": 20.0, "tts_seconds": 3.0,
                "ready_seconds": 20.0, "cycle_started": time.perf_counter() - 20.0,
            })
            return state["metrics"]

        with tempfile.TemporaryDirectory() as directory:
            metrics = asyncio.run(exercise(Path(directory)))
        self.assertAlmostEqual(metrics["completion_interval_ema"], 20.0, delta=0.1)

    def test_bilingual_tts_alternates_original_then_translation(self):
        async def exercise(root: Path):
            runtime = SupertonicRuntime.__new__(SupertonicRuntime)
            calls = []

            async def synthesize(text, output, *, voice, language, speed=1.05):
                calls.append((text, language, voice, speed))
                with wave.open(str(output), "wb") as audio:
                    audio.setnchannels(1)
                    audio.setsampwidth(2)
                    audio.setframerate(8000)
                    audio.writeframes(b"\x00\x00" * 8)
                return 0.01

            runtime.synthesize = synthesize
            await runtime.synthesize_alternating([
                {"original": "\u300c\u884c\u3053\u3046\u3002\u300d", "translation": "Let us go."},
                {"original": "\u6b21\u3078\u3002", "translation": "Next."},
            ], root / "result.wav", voice="F2", original_language="ja", translation_language="en")
            return calls

        with tempfile.TemporaryDirectory() as directory:
            calls = asyncio.run(exercise(Path(directory)))
        self.assertEqual(calls, [
            ("\u300c\u884c\u3053\u3046\u3002\u300d", "ja", "F2", 1.05),
            ("Let us go.", "en", "F2", 1.05),
            ("\u6b21\u3078\u3002", "ja", "F2", 1.05), ("Next.", "en", "F2", 1.05),
        ])

    def test_theater_accepts_all_advanced_generation_values(self):
        requested = {
            "width": 640, "height": 368, "frames": 65, "fps": 24,
            "min_words": 120, "max_words": 360, "max_slow": 12.5,
        }
        config = validate_theater_payload({"prompt": "A story", "quality_settings": requested})
        self.assertEqual(config["quality_settings"], requested)
        self.assertEqual(TheaterManager.quality_settings(config), requested)

    def test_theater_rejects_invalid_custom_frame_rule(self):
        with self.assertRaisesRegex(ValueError, r"4n\+1"):
            validate_theater_payload({"prompt": "A story", "quality_settings": {"frames": 80}})

    def test_legacy_theater_archives_keep_their_saved_quality(self):
        settings = TheaterManager.quality_settings({"quality": "realtime"})
        self.assertEqual(settings["width"], 192)
        self.assertEqual(settings["frames"], 33)

    def test_first_scene_review_failure_is_sent_to_consumer(self):
        async def exercise():
            manager = TheaterManager.__new__(TheaterManager)

            async def fail_review(_state, _scene):
                raise TheaterError("closed factual gate")

            manager._verify_scene = fail_review
            state = {
                "id": "test", "planned": [],
                "bootstrap_scene": {"number": 1, "beat": "b", "visual_action": "v", "camera": "c"},
            }
            queue = asyncio.Queue(maxsize=3)
            await asyncio.wait_for(manager._planner_loop(state, queue), timeout=1)
            return queue.get_nowait()

        result = asyncio.run(exercise())
        self.assertIn("closed factual gate", result["_error"])

    def test_planner_crash_cannot_leave_consumer_waiting(self):
        async def exercise():
            async def crash():
                await asyncio.sleep(0)
                raise RuntimeError("planner exploded")

            queue = asyncio.Queue(maxsize=3)
            planner = asyncio.create_task(crash())
            await TheaterManager._next_planned_scene(queue, planner)

        with self.assertRaisesRegex(TheaterError, "planner exploded"):
            asyncio.run(exercise())

    def test_timed_scene_wait_does_not_leave_a_hidden_queue_consumer(self):
        async def exercise():
            queue = asyncio.Queue()
            planner = asyncio.create_task(asyncio.sleep(10))
            try:
                with self.assertRaises(asyncio.TimeoutError):
                    await asyncio.wait_for(
                        TheaterManager._next_planned_scene(queue, planner), timeout=0.01,
                    )
                await queue.put({"number": 1})
                await asyncio.sleep(0)
                return queue.qsize()
            finally:
                planner.cancel()
                await asyncio.gather(planner, return_exceptions=True)

        self.assertEqual(asyncio.run(exercise()), 1)

    def test_single_scene_list_from_small_model_is_accepted(self):
        scene = {"number": 1, "title": "A scene"}
        self.assertIs(TheaterManager._scene_object([scene], "test"), scene)
        with self.assertRaisesRegex(TheaterError, "exactly one scene"):
            TheaterManager._scene_object([scene, dict(scene)], "test")

    def test_structured_cast_is_rendered_for_video_prompt(self):
        bible = {
            "protagonists": [
                {"name": "Paju", "role": "young pig", "appearance": "red scarf"},
                {"name": "Susi", "role": "approaching wolf", "appearance": "grey coat"},
            ]
        }
        self.assertEqual(
            TheaterManager._cast_text(bible),
            "Paju, young pig, red scarf; Susi, approaching wolf, grey coat",
        )

    def test_writer_contract_keeps_explicit_seed_events_binding(self):
        manager = TheaterManager.__new__(TheaterManager)
        prompt = manager._system_prompt({"audience": "family", "mode": "story", "language": "fi"})
        self.assertIn("binding premise contract", prompt)
        self.assertIn("Never delay, remove, reverse or contradict", prompt)
        self.assertIn("MANDATORY OUTPUT LANGUAGE: Finnish (suomi) [fi]", prompt)
        self.assertIn("Do not translate the user's story into English", prompt)

    def test_interactive_character_mode_reuses_pure_story_resources_and_translation(self):
        manager = TheaterManager.__new__(TheaterManager)
        config = validate_theater_payload({
            "prompt": "A lighthouse keeper talks with viewers.",
            "mode": "interactive", "language": "fi", "translation_language": "en",
        })
        prompt = manager._system_prompt(config)
        self.assertEqual(config["mode"], "interactive")
        self.assertEqual(config["translation_language"], "en")
        self.assertFalse(manager.uses_grounding(config))
        self.assertIn("one stable primary on-screen host", prompt)
        self.assertIn("host's natural first-person speech", prompt)
        self.assertIn("delayed turns, not real-time perception", prompt)

    def test_interactive_visual_prompt_keeps_host_present_and_recognizable(self):
        manager = TheaterManager.__new__(TheaterManager)
        state = {
            "config": {"mode": "interactive"},
            "bible": {
                "visual_style": "cinematic documentary", "world": "an old lighthouse",
                "protagonists": [{"name": "Mira", "role": "keeper", "appearance": "yellow coat"}],
                "premise_contract": ["Mira repairs the lamp"], "continuity_rules": ["The storm continues"],
            },
        }
        prompt = manager._visual_prompt(state, {
            "camera": "medium shot", "visual_action": "Mira cleans a brass lens.",
        })
        self.assertIn("primary host clearly recognizable and present", prompt)
        self.assertIn("natural near-camera eyeline", prompt)

    def test_dream_bootstrap_replaces_literal_seed_requirements_and_removes_facts(self):
        async def exercise(root: Path):
            requests = []

            class Writer:
                async def complete(self, messages, max_tokens=1300):
                    requests.append(messages[-1]["content"])
                    return json.dumps({
                        "title": "The Soft Staircase",
                        "bible": {
                            "protagonists": [{"name": "Iri", "role": "wanderer", "appearance": "silver coat"}],
                            "world": "a staircase floating through warm rain",
                            "visual_style": "soft nocturnal cinema",
                            "premise_contract": ["Velvet must remain literal"],
                            "continuity_rules": ["Doors remember the last color they touched"],
                        },
                        "story_summary": "Iri follows a staircase through rain.",
                        "scene": {
                            "number": 1, "title": "Warm Steps", "beat": "The stairs unfold",
                            "narration": "Warm rain gathers while a silver staircase unfolds beneath Iri.",
                            "visual_action": "Iri steps onto a stair that opens like a flower.",
                            "camera": "slow tracking shot", "learning_point": "Velvet is a woven textile.",
                            "sources": [{"title": "Textiles", "url": "offline://textiles"}],
                        },
                    }), {"tokens_per_second": 12.0, "elapsed_seconds": 1.0, "prompt_tokens": 100}

            manager = TheaterManager.__new__(TheaterManager)
            manager.root = root
            manager.writer = Writer()
            state = {
                "id": "dream", "metrics": {},
                "config": validate_theater_payload({"prompt": "Velvet", "mode": "dream"}),
            }
            result = await manager._bootstrap(state)
            return requests[0], result

        with tempfile.TemporaryDirectory() as directory:
            request, result = asyncio.run(exercise(Path(directory)))
        self.assertIn("loosely associated with this pre-sleep cue: Velvet", request)
        self.assertIn("no non-negotiable literal requirements", request)
        self.assertNotIn("OFFLINE ENCYCLOPEDIA EXCERPTS", request)
        self.assertEqual(result["bible"]["experience"], "dream")
        self.assertEqual(result["bible"]["seed_role"], "weak_association")
        self.assertNotIn("Velvet must remain literal", result["bible"]["premise_contract"])
        self.assertEqual(result["scene"]["learning_point"], "")
        self.assertNotIn("sources", result["scene"])

    def test_dream_visual_prompt_allows_intentional_metamorphosis(self):
        manager = TheaterManager.__new__(TheaterManager)
        state = {
            "config": {"mode": "dream"},
            "bible": {
                "visual_style": "soft nocturnal cinema", "world": "a floating staircase",
                "protagonists": [{"name": "Iri", "role": "wanderer", "appearance": "silver coat"}],
                "premise_contract": ["Everything is invented"],
                "continuity_rules": ["Doors remember colors"],
            },
        }
        prompt = manager._visual_prompt(state, {
            "camera": "slow tracking shot", "visual_action": "A stair opens into a pale moth.",
        })
        self.assertIn("invented dream-space", prompt)
        self.assertIn("surreal metamorphosis", prompt)
        self.assertNotIn("Same faces, ages, bodies", prompt)

    def test_gemma4_e4b_is_primary_and_uses_measured_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "models" / "gemma-4-E4B-it-Q4_K_M.gguf"
            model.parent.mkdir(parents=True)
            model.touch()
            runtime = StoryRuntime(root, root, root)
            self.assertEqual(runtime.model, model)
            self.assertEqual(runtime.model_alias, StoryRuntime.GEMMA4_E4B_ALIAS)
            self.assertEqual(runtime.threads, 8)
            self.assertEqual(runtime.parallel_slots, 2)
            self.assertEqual(runtime.context_tokens_per_slot, 16384)
            args = runtime._server_args(root / "runtime" / "llama-server.exe")
            self.assertEqual(args[args.index("--parallel") + 1], "2")
            self.assertEqual(args[args.index("-c") + 1], "32768")
            self.assertEqual(runtime.sampling["top_k"], 64)

    def test_cuda_writer_profile_uses_benchmarked_isolated_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "models").mkdir()
            (root / "models" / "gemma-4-E4B-it-Q4_K_M.gguf").touch()
            (root / "runtime").mkdir()
            server = root / "runtime" / "llama-server.exe"
            server.touch()
            (root / "runtime" / "ggml-cuda.dll").touch()
            runtime = StoryRuntime(root, root, root, root)
            args = runtime._server_args(server, "gpu")
            self.assertTrue(runtime.gpu_available)
            self.assertEqual(args[args.index("--port") + 1], "18083")
            self.assertEqual(args[args.index("-ngl") + 1], "99")
            self.assertEqual(args[args.index("--parallel") + 1], "2")
            self.assertEqual(args[args.index("-c") + 1], "32768")
            self.assertEqual(args[args.index("-fa") + 1], "on")
            self.assertEqual(args[args.index("-ctk") + 1], "f16")
            self.assertEqual(args[args.index("-ctv") + 1], "f16")

    def test_gpu_buffer_fill_is_bounded_and_translation_ready(self):
        async def exercise():
            manager = TheaterManager.__new__(TheaterManager)
            manager._save = lambda _state: None

            async def verify(_state, scene):
                return scene

            async def plan(_state, number, _recent):
                return {"number": number, "title": f"Scene {number}"}

            async def prepare(_state, scene):
                scene["narration_sentences"] = [{"original": f"Words {scene['number']}."}]
                return scene

            manager._verify_scene = verify
            manager._plan_next = plan
            manager._prepare_narration = prepare
            state = {
                "id": "bounded", "config": {"language": "en", "translation_language": ""},
                "planned": [], "segments": [], "metrics": {},
                "bootstrap_scene": {
                    "number": 1, "title": "Scene 1", "beat": "begin",
                    "visual_action": "walk", "camera": "wide",
                },
            }
            prepared = await asyncio.wait_for(manager._fill_story_buffer(state, 3), timeout=1)
            return prepared, state

        prepared, state = asyncio.run(exercise())
        self.assertGreaterEqual(prepared, 3)
        self.assertLessEqual(len(state["planned"]), 5)
        self.assertTrue(all(
            TheaterManager._narration_is_prepared(state["config"], scene)
            for scene in state["planned"][:3]
        ))

    def test_gpu_buffer_burst_releases_writer_before_returning(self):
        async def exercise(root: Path):
            events = []
            manager = TheaterManager.__new__(TheaterManager)
            manager.root = root
            manager.GPU_BURST_TARGET = 3
            manager._save = lambda _state: None

            class Controller:
                workflow_lock = asyncio.Lock()

                async def wait_until_idle(self):
                    events.append("idle")

                async def free_models(self):
                    events.append("free")

            class Writer:
                model_label = "Gemma"

                async def start(self, _logs, profile="cpu"):
                    events.append(f"start:{profile}")

                async def stop(self, profile=None):
                    events.append(f"stop:{profile}")

                async def healthy(self, profile=None):
                    return False

                def activate(self, profile):
                    events.append(f"activate:{profile}")

            async def fill(_state, target, _excluded=None):
                events.append(f"fill:{target}")
                return target

            manager.controller = Controller()
            manager.writer = Writer()
            manager._fill_story_buffer = fill
            state = {
                "id": "session", "config": {"language": "en", "translation_language": ""},
                "bible": {"world": "test"}, "planned": [], "segments": [], "metrics": {},
            }
            await manager._prime_gpu_story_buffer(state)
            return events, state

        with tempfile.TemporaryDirectory() as directory:
            events, state = asyncio.run(exercise(Path(directory)))
        self.assertEqual(events, [
            "idle", "free", "start:gpu", "fill:3", "stop:gpu", "activate:cpu",
        ])
        self.assertFalse(state["metrics"]["gpu_burst_active"])
        self.assertTrue(state["metrics"]["gpu_burst_completed"])

    def test_gpu_buffer_burst_fails_closed_if_cuda_server_survives(self):
        async def exercise(root: Path):
            manager = TheaterManager.__new__(TheaterManager)
            manager.root = root
            manager.GPU_BURST_TARGET = 3
            manager._save = lambda _state: None

            class Controller:
                workflow_lock = asyncio.Lock()
                async def wait_until_idle(self): pass
                async def free_models(self): pass

            class Writer:
                model_label = "Gemma"
                async def start(self, _logs, profile="cpu"): pass
                async def stop(self, profile=None): pass
                async def healthy(self, profile=None): return True
                def activate(self, profile): pass

            manager.controller = Controller()
            manager.writer = Writer()
            manager._fill_story_buffer = lambda _state, _target, _excluded=None: asyncio.sleep(0, result=3)
            state = {
                "id": "session", "config": {"language": "en", "translation_language": ""},
                "bible": {"world": "test"}, "planned": [], "segments": [], "metrics": {},
            }
            await manager._prime_gpu_story_buffer(state)

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(GpuReleaseError, "still running"):
                asyncio.run(exercise(Path(directory)))

    def test_every_adaptive_gpu_refill_compacts_before_planning(self):
        async def exercise(root: Path):
            events = []
            manager = TheaterManager.__new__(TheaterManager)
            manager.root = root
            manager.GPU_BURST_TARGET = 3
            manager._save = lambda _state: None

            class Controller:
                workflow_lock = asyncio.Lock()
                async def wait_until_idle(self): events.append("idle")
                async def free_models(self): events.append("free")

            class Writer:
                model_label = "Gemma"
                profile = "cpu"
                async def start(self, _logs, profile="cpu"):
                    self.profile = profile
                    events.append(f"start:{profile}")
                async def stop(self, profile=None): events.append(f"stop:{profile}")
                async def healthy(self, profile=None): return False
                def activate(self, profile):
                    self.profile = profile
                    events.append(f"activate:{profile}")

            async def compact(_state, reason):
                events.append(f"compact:{reason}")
                return True

            async def fill(_state, target, _excluded=None):
                events.append(f"fill:{target}")
                return target

            manager.controller = Controller()
            manager.writer = Writer()
            manager._compact_story_context = compact
            manager._fill_story_buffer = fill
            state = {
                "id": "session", "config": {"language": "en", "translation_language": ""},
                "bible": {"world": "test"}, "planned": [{"number": 30}],
                "segments": [], "metrics": {},
            }
            await manager._prime_gpu_story_buffer(state, reason="adaptive_refill")
            return events

        with tempfile.TemporaryDirectory() as directory:
            events = asyncio.run(exercise(Path(directory)))
        self.assertLess(events.index("compact:adaptive_refill"), events.index("fill:3"))

    def test_adaptive_gpu_refill_requires_an_empty_queue_and_measured_advantage(self):
        manager = TheaterManager.__new__(TheaterManager)

        class Writer:
            gpu_available = True

        manager.writer = Writer()
        state = {
            "config": {"language": "en", "translation_language": "es"},
            "metrics": {
                "translation_elapsed_ema": 40.0,
                "translation_request_started_at": time.time() - 5.0,
                "gpu_burst_total_seconds": 18.6,
            },
        }
        source_queue = asyncio.Queue()
        ready_queue = asyncio.Queue()
        self.assertTrue(manager._should_refill_on_gpu(state, ready_queue, source_queue))
        self.assertGreater(state["metrics"]["gpu_refill_predicted_cpu_wait"], 34)
        ready_queue.put_nowait({"number": 4})
        self.assertFalse(manager._should_refill_on_gpu(state, ready_queue, source_queue))
        ready_queue.get_nowait()
        state["metrics"]["translation_request_started_at"] = time.time() - 35.0
        self.assertFalse(manager._should_refill_on_gpu(state, ready_queue, source_queue))
        state["metrics"].pop("translation_request_started_at")
        state["metrics"].update({
            "planner_cycle_ema": 25.0,
            "context_compaction_elapsed_ema": 30.0,
            "context_compaction_started_at": time.time() - 5.0,
        })
        self.assertTrue(manager._should_refill_on_gpu(state, ready_queue, source_queue))
        self.assertGreater(state["metrics"]["gpu_refill_predicted_cpu_wait"], 80)

    def test_gpu_refill_cycles_do_not_contaminate_cpu_wait_model(self):
        async def exercise():
            manager = TheaterManager.__new__(TheaterManager)
            manager._save = lambda _state: None

            class Writer:
                profile = "gpu"

            async def plan(_state, number, _recent):
                return {"number": number, "title": f"Scene {number}"}

            manager.writer = Writer()
            manager._plan_next = plan
            state = {"id": "ema", "planned": [], "metrics": {"planner_cycle_ema": 50.0}}
            queue = asyncio.Queue(maxsize=3)
            task = asyncio.create_task(manager._planner_loop(state, queue))
            try:
                await asyncio.wait_for(queue.get(), timeout=1)
                return state["metrics"]
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        metrics = asyncio.run(exercise())
        self.assertEqual(metrics["planner_cycle_ema"], 50.0)
        self.assertIn("gpu_planner_cycle_seconds", metrics)

    def test_context_compaction_preserves_structured_continuity(self):
        async def exercise(root: Path):
            manager = TheaterManager.__new__(TheaterManager)
            manager.root = root
            manager._save = lambda _state: None
            requests = []

            class Writer:
                profile = "gpu"
                async def complete(self, messages, max_tokens=900):
                    requests.append(messages[-1]["content"])
                    return (
                        '{"story_summary":"Mira guards the opened gate while Sol searches for the missing key.",'
                        '"continuity_memory":{"character_states":['
                        '{"name":"Mira","state":"alert","location":"gate","possessions":["map"]}],'
                        '"active_threads":["Find the missing key"],'
                        '"continuity_facts":["The northern gate is permanently open"]}}',
                        {"elapsed_seconds": 1.5, "prompt_tokens": 700, "completion_tokens": 120},
                    )

            manager.writer = Writer()
            (root / "session" / "logs").mkdir(parents=True)
            state = {
                "id": "session",
                "config": {"language": "en", "mode": "story", "audience": "family"},
                "bible": {"premise_contract": ["Mira must guard the gate"]},
                "story_summary": "Long prior summary. " * 100,
                "planned": [
                    {"number": number, "title": f"Scene {number}", "beat": "Advance", "narration": "Action.",
                     "visual_action": "Mira moves."}
                    for number in range(1, 31)
                ],
                "metrics": {},
            }
            compacted = await manager._compact_story_context(state, reason="adaptive_refill")
            return compacted, state, requests

        with tempfile.TemporaryDirectory() as directory:
            compacted, state, requests = asyncio.run(exercise(Path(directory)))
        self.assertTrue(compacted)
        self.assertEqual(state["metrics"]["last_context_compaction_scene"], 30)
        self.assertEqual(state["metrics"]["context_compaction_profile"], "gpu")
        self.assertEqual(state["continuity_memory"]["character_states"][0]["name"], "Mira")
        self.assertIn("Find the missing key", state["continuity_memory"]["active_threads"])
        self.assertIn("FIXED BIBLE", requests[0])
        self.assertLess(
            state["metrics"]["context_compaction_after_chars"],
            state["metrics"]["context_compaction_before_chars"],
        )

    def test_context_compaction_interval_uses_last_attempt_not_context_limit(self):
        manager = TheaterManager.__new__(TheaterManager)
        state = {
            "config": {"context_compaction_scenes": 30}, "bible": {"world": "test"},
            "planned": [{"number": number} for number in range(1, 31)], "metrics": {},
        }
        self.assertTrue(manager._context_compaction_due(state))
        state["metrics"]["last_context_compaction_attempt_scene"] = 30
        self.assertFalse(manager._context_compaction_due(state))
        state["planned"].extend({"number": number} for number in range(31, 60))
        self.assertFalse(manager._context_compaction_due(state))
        state["planned"].append({"number": 60})
        self.assertTrue(manager._context_compaction_due(state))
        state["config"]["context_compaction_scenes"] = 0
        self.assertFalse(manager._context_compaction_due(state))

    def test_compaction_replaces_old_recent_scenes_with_three_causal_anchors(self):
        recent = [{"number": number} for number in range(1, 14)]
        compacted = {"context_compacted_through_scene": 10}
        self.assertEqual(
            [item["number"] for item in TheaterManager._recent_scene_context(compacted, recent)],
            [8, 9, 10, 11, 12, 13],
        )
        self.assertEqual(
            [item["number"] for item in TheaterManager._recent_scene_context({}, recent)],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        )

    def test_restored_workers_exclude_rendered_but_unarchived_scene(self):
        async def exercise():
            manager = TheaterManager.__new__(TheaterManager)
            manager._save = lambda _state: None
            manager._plan_next = lambda *_args: asyncio.sleep(10)
            manager._prepare_narration = lambda *_args: asyncio.sleep(10)
            prepared = lambda number: {
                "number": number, "title": f"Scene {number}",
                "narration_sentences": [{"original": "Ready."}],
            }
            state = {
                "id": "restore", "config": {"language": "en", "translation_language": ""},
                "segments": [], "planned": [prepared(1), prepared(2), {"number": 3, "title": "Scene 3"}],
                "metrics": {},
            }
            source, ready, planner, translator = manager._restore_story_workers(state, {1})
            try:
                return [item["number"] for item in list(ready._queue)], [item["number"] for item in list(source._queue)]
            finally:
                await manager._cancel_story_workers(planner, translator)

        ready_numbers, source_numbers = asyncio.run(exercise())
        self.assertEqual(ready_numbers, [2])
        self.assertEqual(source_numbers, [3])

    def test_planning_overlaps_translation_through_bounded_queues(self):
        async def exercise():
            manager = TheaterManager.__new__(TheaterManager)
            manager._save = lambda _state: None
            second_plan_started = asyncio.Event()
            first_translation_started = asyncio.Event()

            async def plan(_state, number, _recent):
                if number == 2:
                    second_plan_started.set()
                    await asyncio.wait_for(first_translation_started.wait(), timeout=1)
                return {"number": number, "title": f"Scene {number}"}

            async def translate(_state, scene):
                if scene["number"] == 1:
                    first_translation_started.set()
                    await asyncio.wait_for(second_plan_started.wait(), timeout=1)
                scene["narration_sentences"] = [{"original": f"Words {scene['number']}."}]
                return scene

            manager._plan_next = plan
            manager._prepare_narration = translate
            state = {"config": {"language": "en", "translation_language": "fi"}, "planned": [], "metrics": {}}
            source_queue = asyncio.Queue(maxsize=3)
            ready_queue = asyncio.Queue(maxsize=3)
            planner = asyncio.create_task(manager._planner_loop(state, source_queue))
            translator = asyncio.create_task(
                manager._translation_loop(state, source_queue, ready_queue, planner)
            )
            ready = await asyncio.wait_for(ready_queue.get(), timeout=1)
            planner.cancel()
            translator.cancel()
            await asyncio.gather(planner, translator, return_exceptions=True)
            return ready, second_plan_started.is_set(), first_translation_started.is_set()

        ready, planned_in_parallel, translated_in_parallel = asyncio.run(exercise())
        self.assertEqual(ready["number"], 1)
        self.assertTrue(planned_in_parallel)
        self.assertTrue(translated_in_parallel)

    def test_resume_only_retranslates_incomplete_saved_scenes(self):
        bilingual = {"language": "fi", "translation_language": "en"}
        source_only = {"narration_sentences": [{"original": "Lähdetään."}]}
        translated = {
            "translated_title": "Departure",
            "narration_sentences": [{"original": "Lähdetään.", "translation": "Let us go."}],
        }
        self.assertFalse(TheaterManager._narration_is_prepared(bilingual, source_only))
        self.assertTrue(TheaterManager._narration_is_prepared(bilingual, translated))
        self.assertTrue(TheaterManager._narration_is_prepared(
            {"language": "fi", "translation_language": ""}, source_only,
        ))

    def test_bilingual_planner_prompt_stays_inside_default_total_budget(self):
        async def exercise(root: Path):
            manager = TheaterManager.__new__(TheaterManager)
            manager.root = root
            (root / "session" / "logs").mkdir(parents=True)
            requests = []

            class Writer:
                async def complete(self, messages, max_tokens=900):
                    requests.append(messages[-1]["content"])
                    return (
                        '{"story_summary":"Moving onward","scene":{"number":2,"title":"Path",'
                        '"beat":"They depart","narration":"At dawn the travelers follow a path. Silver mist curls '
                        'softly between ancient trees. They cross a bridge above rushing water. Their careful footsteps '
                        'echo through the valley. Together they choose the brighter forest trail. The storm gathers '
                        'behind them.",'
                        '"visual_action":"The group crosses a bridge","camera":"wide tracking",'
                        '"learning_point":""}}',
                        {"tokens_per_second": 14.0, "elapsed_seconds": 1.2, "prompt_tokens": 420},
                    )

            async def verify(_state, scene):
                return scene

            manager.writer = Writer()
            manager._verify_scene = verify
            state = {
                "id": "session",
                "config": validate_theater_payload({"prompt": "A path", "translation_language": "fi"}),
                "bible": {}, "story_summary": "They are ready.",
                "continuity_memory": {"active_threads": ["Find the gate"]}, "planned": [],
                "live_directives": [{
                    "id": "storm", "scope": "next_scene", "status": "pending",
                    "text": "A sudden storm forces them into the lighthouse.",
                }],
                "metrics": {"production_ema": 40.0},
            }
            scene = await manager._plan_next(state, 2, [])
            manager._log_live_directive = lambda *_args: None
            manager._mark_directives_applied(state, scene)
            return requests[0], state["metrics"], scene

        with tempfile.TemporaryDirectory() as directory:
            request, metrics, scene = asyncio.run(exercise(Path(directory)))
        self.assertIn("Create scene 2 with 39-42 source-language narration words", request)
        self.assertIn("hard playback-duration budget", request)
        self.assertIn("exactly 6 complete sentences", request)
        self.assertIn("sentence contain 7-7 words", request)
        self.assertIn("Find the gate", request)
        self.assertIn("LIVE WORLD EVENTS AND DIRECTIONS", request)
        self.assertIn("A sudden storm forces them into the lighthouse", request)
        self.assertEqual(metrics["planner_prompt_tokens"], 420)
        self.assertEqual(scene["planner_metrics"]["elapsed_seconds"], 1.2)
        self.assertEqual(scene["_live_directive_ids"], ["storm"])

    def test_live_steering_rolls_back_only_unrendered_plans_and_causal_state(self):
        manager = TheaterManager.__new__(TheaterManager)
        state = {
            "story_summary": "Scene three already happened.",
            "continuity_memory": {"active_threads": ["Wrong future"]},
            "context_compacted_through_scene": 3,
            "planned": [
                {"number": 1, "title": "Archived"},
                {
                    "number": 2, "title": "Speculative", "_planning_context_before": {
                        "story_summary": "Only scene one happened.",
                        "has_continuity_memory": True,
                        "continuity_memory": {"active_threads": ["Original thread"]},
                        "has_compaction_boundary": True,
                        "context_compacted_through_scene": 1,
                        "context_compaction_metrics": {"last_context_compaction_scene": 1},
                    },
                },
                {"number": 3, "title": "More speculation"},
            ],
            "live_directives": [{
                "id": "storm", "scope": "next_scene", "status": "applied", "applied_scene": 2,
            }, {
                "id": "rain", "scope": "persistent", "status": "active", "first_applied_scene": 2,
            }],
            "metrics": {
                "last_context_compaction_scene": 3, "context_compaction_count": 2,
                "planner_repair_reason": "stale output",
            },
        }
        discarded = manager._rollback_speculative_plans(state, {1})
        self.assertEqual(discarded, 2)
        self.assertEqual([item["number"] for item in state["planned"]], [1])
        self.assertEqual(state["story_summary"], "Only scene one happened.")
        self.assertEqual(state["continuity_memory"]["active_threads"], ["Original thread"])
        self.assertEqual(state["context_compacted_through_scene"], 1)
        self.assertEqual(state["metrics"]["last_context_compaction_scene"], 1)
        self.assertNotIn("context_compaction_count", state["metrics"])
        self.assertNotIn("planner_repair_reason", state["metrics"])
        self.assertEqual(state["live_directives"][0]["status"], "pending")
        self.assertNotIn("first_applied_scene", state["live_directives"][1])
        self.assertEqual(state["metrics"]["last_live_steering_scene"], 2)

    def test_run_rebinds_workers_without_dropping_prepared_scenes_on_live_steering(self):
        async def exercise(root: Path):
            manager = TheaterManager.__new__(TheaterManager)
            prepared = [
                {"number": 2, "title": "Prepared two", "narration_sentences": [{"original": "Two."}]},
                {"number": 3, "title": "Prepared three", "narration_sentences": [{"original": "Three."}]},
            ]
            completed = {"number": 1, "title": "Completed", "path": "scene-1.mp4"}
            state = {
                "id": "session",
                "config": validate_theater_payload({"prompt": "A continuous story", "mode": "story"}),
                "bible": {"premise": "Keep going"},
                "story_summary": "Scene one is complete.",
                "segments": [completed],
                "planned": [{"number": 1, "title": "Completed"}, *prepared],
                "metrics": {},
                "live_directives": [],
            }

            class Runtime:
                gpu_available = False

                async def start(self, *_args, **_kwargs):
                    return None

            async def idle(*_args):
                await asyncio.Event().wait()

            steering_event = asyncio.Event()
            steering_event.set()
            manager.root = root
            manager.writer = Runtime()
            manager.supertonic = Runtime()
            manager.steering_events = {"session": steering_event}
            manager._save = lambda _state: None
            manager._planner_loop = idle
            manager._translation_loop = idle
            manager._assembly_loop = idle
            bindings = []
            original_restore = manager._restore_story_workers

            def track_restore(inner_state, excluded_numbers):
                result = original_restore(inner_state, excluded_numbers)
                bindings.append(result)
                return result

            manager._restore_story_workers = track_restore
            rebuilt_ready_numbers = []

            async def inspect_rebuilt_queue(queue, _translation_task):
                rebuilt_ready_numbers.extend(item["number"] for item in list(queue._queue))
                raise asyncio.CancelledError

            manager._next_planned_scene = inspect_rebuilt_queue
            with self.assertRaises(asyncio.CancelledError):
                await manager._run(state)
            return state, steering_event, bindings, rebuilt_ready_numbers, completed

        with tempfile.TemporaryDirectory() as directory:
            state, event, bindings, ready_numbers, completed = asyncio.run(exercise(Path(directory)))
        self.assertEqual(state["segments"], [completed])
        self.assertFalse(event.is_set())
        self.assertEqual(ready_numbers, [2, 3])
        self.assertEqual(len(bindings), 2)
        self.assertIsNot(bindings[0][0], bindings[1][0])
        self.assertIsNot(bindings[0][1], bindings[1][1])
        self.assertTrue(all(task.done() for binding in bindings for task in binding[2:]))

    def test_delayed_directive_preserves_buffer_while_fast_directive_wakes_pipeline(self):
        async def exercise(root: Path):
            manager = TheaterManager.__new__(TheaterManager)
            manager.root = root
            (root / "session" / "logs").mkdir(parents=True)
            state = {
                "id": "session", "segments": [], "live_directives": [],
                "planned": [{"number": 1}, {"number": 2}, {"number": 3}],
                "metrics": {"planner_cycle_started_at": time.time()},
            }
            task = asyncio.create_task(asyncio.sleep(60))
            manager.sessions = {"session": state}
            manager.tasks = {"session": task}
            manager.steering_events = {"session": asyncio.Event()}
            manager._save = lambda _state: None
            try:
                manager.add_live_directive("session", "  Keep the lighthouse visible.  ", "persistent")
                delayed = dict(state["live_directives"][0])
                woke_for_delayed = manager.steering_events["session"].is_set()
                manager.remove_live_directive("session", delayed["id"])
                woke_for_delayed_remove = manager.steering_events["session"].is_set()
                manager.add_live_directive(
                    "session", "Open the red door now.", "next_scene", "next_unrendered",
                )
                fast = dict(state["live_directives"][1])
                return delayed, fast, woke_for_delayed, woke_for_delayed_remove, manager.steering_events["session"].is_set(), state
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        with tempfile.TemporaryDirectory() as directory:
            delayed, fast, woke_for_delayed, woke_for_remove, woke_for_fast, state = asyncio.run(exercise(Path(directory)))
        self.assertEqual(delayed["text"], "Keep the lighthouse visible.")
        self.assertEqual(delayed["status"], "active")
        self.assertEqual(delayed["delivery"], "after_buffer")
        self.assertEqual(delayed["activation_scene"], 5)
        self.assertFalse(woke_for_delayed)
        self.assertFalse(woke_for_remove)
        self.assertEqual(fast["delivery"], "next_unrendered")
        self.assertTrue(woke_for_fast)
        self.assertEqual(state["live_directives"][0]["status"], "removed")

    def test_delayed_direction_is_hidden_from_gemma_until_reserved_scene(self):
        state = {"live_directives": [{
            "id": "later", "scope": "next_scene", "status": "pending", "delivery": "after_buffer",
            "activation_scene": 7, "text": "The comet becomes visible.",
        }]}
        early_prompt, early_ids = TheaterManager._steering_context(state, 6)
        ready_prompt, ready_ids = TheaterManager._steering_context(state, 7)
        self.assertEqual(early_prompt, "")
        self.assertEqual(early_ids, [])
        self.assertIn("The comet becomes visible", ready_prompt)
        self.assertEqual(ready_ids, ["later"])

    def test_audience_chat_is_a_single_durable_turn_only_in_interactive_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = TheaterManager.__new__(TheaterManager)
            manager.root = Path(directory)
            manager.tasks = {}
            manager.steering_events = {}
            manager._save = lambda _state: None
            ordinary = {
                "id": "ordinary", "config": {"mode": "story"}, "planned": [], "segments": [],
                "metrics": {}, "live_directives": [],
            }
            interactive = {
                "id": "interactive", "config": {"mode": "interactive"}, "planned": [], "segments": [],
                "metrics": {}, "live_directives": [],
            }
            manager.sessions = {"ordinary": ordinary, "interactive": interactive}
            with self.assertRaisesRegex(TheaterError, "Interactive character"):
                manager.add_live_directive("ordinary", "How was your day?", "audience_message")
            manager.add_live_directive("interactive", "How was your day?", "audience_message")
            directive = interactive["live_directives"][0]
            prompt, ids = manager._steering_context(interactive, directive["activation_scene"])
            self.assertEqual(directive["status"], "pending")
            self.assertEqual(directive["delivery"], "after_buffer")
            self.assertIn("delayed viewer speech", prompt)
            self.assertIn("answer it naturally", prompt)
            self.assertEqual(ids, [directive["id"]])
            manager._log_live_directive = lambda *_args: None
            manager._mark_directives_applied(
                interactive, {"number": directive["activation_scene"], "_live_directive_ids": ids},
            )
            self.assertEqual(directive["status"], "applied")

    def test_live_word_target_stays_inside_custom_budget(self):
        manager = TheaterManager.__new__(TheaterManager)
        bilingual = {
            "config": validate_theater_payload({"prompt": "A path", "translation_language": "fi"}),
            "metrics": {"production_ema": 40.0},
        }
        monolingual = {
            "config": validate_theater_payload({"prompt": "A path"}),
            "metrics": {"production_ema": 40.0},
        }
        self.assertEqual(manager._narration_request_limits(bilingual), (39, 42))
        self.assertEqual(manager._narration_request_limits(monolingual), (104, 110))

    def test_duration_controller_targets_true_cadence_before_voice_slowdown(self):
        manager = TheaterManager.__new__(TheaterManager)
        state = {
            "config": validate_theater_payload({"prompt": "A path", "translation_language": "fi"}),
            "metrics": {
                "completion_interval_ema": 50.3,
                "speech_seconds_per_word_ema": 0.544,
                "spoken_word_multiplier_ema": 2.43,
                "last_narration_speed": 1.05,
            },
        }
        self.assertEqual(manager._target_total_words(state), 100)
        self.assertEqual(manager._target_words(state), 41)
        self.assertEqual(manager._narration_request_limits(state), (39, 44))
        self.assertEqual(manager._narration_speed(state, {"total_spoken_words": 100}), 1.05)
        self.assertEqual(manager._narration_speed(state, {"total_spoken_words": 80}), 0.96)

    def test_theater_planner_keeps_two_scenes_ahead(self):
        async def exercise():
            manager = TheaterManager.__new__(TheaterManager)

            async def plan(_state, number, _recent):
                return {"number": number, "title": f"Scene {number}"}

            manager._plan_next = plan
            manager._save = lambda _state: None
            state = {"planned": []}
            queue = asyncio.Queue(maxsize=3)
            task = asyncio.create_task(manager._planner_loop(state, queue))
            for _ in range(50):
                if queue.qsize() >= 2:
                    break
                await asyncio.sleep(0.01)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            return queue.qsize(), len(state["planned"])

        queued, planned = asyncio.run(exercise())
        self.assertEqual(queued, 2)
        self.assertEqual(planned, 2)


if __name__ == "__main__":
    unittest.main()
