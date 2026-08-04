import asyncio
import tempfile
import unittest
from pathlib import Path

from app import build_prompt, validate_movie_payload, validate_payload, validate_theater_payload
from movie_pipeline import MovieManager
from theater_pipeline import StoryRuntime, TheaterError, TheaterManager


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

    def test_theater_defaults_are_offline_realtime_safe(self):
        config = validate_theater_payload({"prompt": "Teach astronomy through an adventure."})
        self.assertEqual(config["quality"], "realtime")
        self.assertEqual(config["mode"], "edutainment")
        self.assertEqual(config["audience"], "family")
        self.assertEqual(config["voice"], "M1")
        self.assertEqual(config["language"], "en")
        self.assertGreaterEqual(config["seed"], 0)

    def test_theater_rejects_unknown_quality(self):
        with self.assertRaisesRegex(ValueError, "supported theater quality"):
            validate_theater_payload({"prompt": "A story", "quality": "impossible"})

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
            self.assertEqual(runtime.sampling["top_k"], 64)

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
