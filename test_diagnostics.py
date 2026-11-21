"""Tests for MatchDiagnostics telemetry correctness and protocol safety."""
import gzip
import json
import os
import sys
import unittest

# Ensure python directory is in path
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from diagnostics import MatchDiagnostics
from bot import ApexBot


class TestMatchDiagnostics(unittest.TestCase):
    def test_basic_counters(self):
        diag = MatchDiagnostics(enabled=True)
        diag.on_hello({"type": "hello", "slot": 3})

        # Simulate obs 1
        obs1 = {
            "type": "obs", "tick": 1, "t": 0.1, "alive": 16,
            "self": {
                "id": 3, "pos": [0, 0, 0], "aim": [0, 0], "hp": 1400, "in_zone": True,
                "weapons": [{"kind": "gun", "cooldown": 0, "ammo": 12, "reload": 0}]
            },
            "events": []
        }
        act1 = {"tick": 1, "move": [1, 0], "aim": [0, 0], "fire": [True, False]}
        diag.update(obs1, act1)

        # Simulate obs 2: took zone damage, outside zone
        obs2 = {
            "type": "obs", "tick": 2, "t": 0.2, "alive": 16,
            "self": {
                "id": 3, "pos": [1, 0, 0], "aim": [0, 0], "hp": 1380, "in_zone": False,
                "weapons": [{"kind": "gun", "cooldown": 0.24, "ammo": 11, "reload": 0}]
            },
            "events": [{"type": "damaged", "amount": 20, "weapon": "zone", "bearing": None, "by": None}]
        }
        act2 = {"tick": 2, "move": [1, 0], "aim": [0, 0], "fire": [False, False]}
        diag.update(obs2, act2)

        # Simulate obs 3: landed a hit, collected a cube
        obs3 = {
            "type": "obs", "tick": 3, "t": 0.3, "alive": 15,
            "self": {
                "id": 3, "pos": [2, 0, 0], "aim": [0, 0], "hp": 1380, "in_zone": True,
                "weapons": [{"kind": "gun", "cooldown": 0, "ammo": 11, "reload": 0}]
            },
            "events": [
                {"type": "hit", "target": 5, "amount": 45, "weapon": "gun", "rear": True},
                {"type": "pickup", "cubes": 1}
            ]
        }
        act3 = {"tick": 3, "move": [1, 0], "aim": [0, 0], "fire": [False, False]}
        diag.update(obs3, act3)

        # End of match
        end_msg = {"type": "end", "place": 4, "kills": 1}
        diag.on_end(end_msg)

        rep = diag.generate_report()
        self.assertEqual(rep["summary"]["placement"], 4)
        self.assertEqual(rep["summary"]["kills"], 1)
        self.assertEqual(rep["summary"]["score"], 12)
        self.assertAlmostEqual(rep["summary"]["survival_duration_s"], 0.2)
        self.assertEqual(rep["damage"]["received_from_zone"], 20.0)
        self.assertEqual(rep["damage"]["dealt_total"], 45.0)
        self.assertEqual(rep["damage"]["rear_hits_dealt"], 1)
        self.assertEqual(rep["economy_and_loot"]["cubes_collected"], 1)
        self.assertEqual(rep["survival_and_health"]["time_outside_zone_s"], 0.1)

    def test_regen_inference(self):
        diag = MatchDiagnostics(enabled=True)
        diag.on_hello({"type": "hello", "slot": 0})

        # Tick at t=0 taking damage
        obs0 = {
            "type": "obs", "tick": 0, "t": 0.0,
            "self": {"id": 0, "hp": 500, "in_zone": True, "weapons": []},
            "events": [{"type": "damaged", "amount": 100, "weapon": "gun"}]
        }
        diag.update(obs0, {"tick": 0})

        # Tick at t=5.0: 5.0 seconds passed without damage, HP rises to 545
        obs1 = {
            "type": "obs", "tick": 50, "t": 5.0,
            "self": {"id": 0, "hp": 545, "in_zone": True, "weapons": []},
            "events": []
        }
        diag.update(obs1, {"tick": 50})

        rep = diag.generate_report()
        self.assertAlmostEqual(rep["survival_and_health"]["health_recovered_regen"], 45.0)

    def test_disabled_diagnostics_is_noop(self):
        diag = MatchDiagnostics(enabled=False)
        diag.on_hello({"type": "hello", "slot": 0})
        diag.update({"t": 1.0, "self": {"hp": 100}}, {})
        diag.on_end({"type": "end", "place": 1, "kills": 5})
        # Counter should remain 0 when disabled
        self.assertEqual(diag.total_ticks, 0)
        self.assertIsNone(diag.final_place)

    def test_replay_stream_run(self):
        """Run bot with diagnostics against the recorded stream to verify real-world pipeline."""
        stream_path = os.path.join(os.path.dirname(HERE), "check", "stream.jsonl.gz")
        arena_dir = os.path.join(os.path.dirname(HERE), "arena")
        
        with gzip.open(stream_path, "rt") as f:
            hello = json.loads(f.readline())
            hello["map"]["dir"] = arena_dir
            bot = ApexBot(hello)
            
            end_msg = None
            for line in f:
                m = json.loads(line)
                if m.get("type") == "obs":
                    bot.act(m)
                elif m.get("type") == "end":
                    end_msg = m
                    bot.on_end(m)
                    break

        self.assertIsNotNone(end_msg)
        rep = bot.diag.generate_report()
        self.assertEqual(rep["summary"]["placement"], 13)
        self.assertEqual(rep["summary"]["kills"], 0)
        self.assertGreater(rep["damage"]["received_total"], 1000)
        self.assertIsNotNone(rep["survival_and_health"]["death_context"])
        print("\n[TEST] Sample stream diagnostic report generated successfully.")


if __name__ == "__main__":
    unittest.main()
