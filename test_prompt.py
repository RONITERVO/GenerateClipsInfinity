import asyncio
import tempfile
import time
import unittest
import wave
from pathlib import Path

from app import build_prompt, validate_movie_payload, validate_payload, validate_theater_payload
from movie_pipeline import MovieManager
from theater_pipeline import (
    GpuReleaseError, StoryRuntime, SupertonicRuntime, TheaterError, TheaterManager, split_narration_sentences,
    spoken_word_count,
)


class PromptTests(unittest.TestCase):
    def test_blueprint_graph_and_output_node(self):
        config = validate_payload(
            {
                "prompt": "A test scene",
                "negative": "blurry",
                "width": 480,
                "height": 272,
                "frames": 17,
                "fps": 16,
                "seed": 42,
            }
        )
        graph = build_prompt(config)
        self.assertEqual(graph["8"]["inputs"]["end_at_step"], 2)
        self.assertEqual(graph["12"]["inputs"]["start_at_step"], 2)
        self.assertEqual(graph["16"]["class_type"], "SaveVideo")
        self.assertEqual(graph["4"]["inputs"]["width"], 480)
        self.assertEqual(graph["4"]["inputs"]["length"], 17)

    def test_invalid_frame_rule_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "4n\\+1"):
            validate_payload({"prompt": "test", "width": 480, "height": 272, "frames": 18, "fps": 16})

    def test_movie_payload_and_random_seed(self):
        config = validate_movie_payload({
            "sentence": "A keeper hears a storm speak.", "shots": 3,
            "width": 192, "height": 192, "frames": 9, "fps": 12,
            "seed": -1, "narration": True,
        })
        self.assertEqual(config["shots"], 3)
        self.assertGreaterEqual(config["seed"], 0)
        self.assertTrue(config["narration"])

    def test_movie_defaults_use_max_frames_and_edit_policy(self):
        config = validate_movie_payload({"sentence": "A local movie idea.", "shots": 3})
        self.assertEqual(config["frames"], 81)
        self.assertEqual(config["sync_mode"], "fit_video_to_audio")
        self.assertEqual(config["fill_mode"], "freeze")
        self.assertTrue(config["motion_interpolation"])

    def test_audio_retime_chain_supports_extreme_factors(self):
        self.assertEqual(MovieManager._atempo(8.0), "atempo=2.000000,atempo=2.000000,atempo=2.000000")
        self.assertEqual(MovieManager._atempo(0.25), "atempo=0.500000,atempo=0.500000")

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
        self.assertGreaterEqual(config["seed"], 0)

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

            async def fill(_state, target):
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
            manager._fill_story_buffer = lambda _state, _target: asyncio.sleep(0, result=3)
            state = {
                "id": "session", "config": {"language": "en", "translation_language": ""},
                "bible": {"world": "test"}, "planned": [], "segments": [], "metrics": {},
            }
            await manager._prime_gpu_story_buffer(state)

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(GpuReleaseError, "still running"):
                asyncio.run(exercise(Path(directory)))

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
                "bible": {}, "story_summary": "They are ready.", "planned": [],
                "metrics": {"production_ema": 40.0},
            }
            scene = await manager._plan_next(state, 2, [])
            return requests[0], state["metrics"], scene

        with tempfile.TemporaryDirectory() as directory:
            request, metrics, scene = asyncio.run(exercise(Path(directory)))
        self.assertIn("Create scene 2 with 39-42 source-language narration words", request)
        self.assertIn("hard playback-duration budget", request)
        self.assertIn("exactly 6 complete sentences", request)
        self.assertIn("sentence contain 7-7 words", request)
        self.assertEqual(metrics["planner_prompt_tokens"], 420)
        self.assertEqual(scene["planner_metrics"]["elapsed_seconds"], 1.2)

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
