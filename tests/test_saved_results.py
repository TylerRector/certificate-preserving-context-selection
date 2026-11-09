import json
import unittest

from certificate_selection.core import SEEDS, packet_check
from certificate_selection.storage import ROOT, load_jsonl


class SavedResultsTests(unittest.TestCase):
    def test_main_counts_and_exact_tie(self):
        rows = json.loads((ROOT / "data/final/cpu/cpu_metrics.json").read_text())
        primary = [row for row in rows if row["mode"] == "primary"]
        methods = {
            method: {row["root"]: row for row in primary if row["method"] == method}
            for method in ["multi_candidate", "certificate_greedy", "okapi_bm25"]
        }
        for method in methods.values():
            self.assertEqual(len(method), 2048)
            self.assertEqual(
                {
                    seed: sum((row["seed"] == seed for row in method.values()))
                    for seed in SEEDS
                },
                {seed: 512 for seed in SEEDS},
            )
        for name in ["multi_candidate", "certificate_greedy"]:
            self.assertEqual(
                sum((row["certified"] for row in methods[name].values())), 1966
            )
            self.assertEqual(
                sum((row["condition_kept"] for row in methods[name].values())), 5766
            )
            self.assertEqual(
                sum((row["condition_total"] for row in methods[name].values())), 13560
            )
        self.assertTrue(
            all(
                (
                    row["certified"] == methods["certificate_greedy"][root]["certified"]
                    for root, row in methods["multi_candidate"].items()
                )
            )
        )
        self.assertEqual(
            sum((row["certified"] for row in methods["okapi_bm25"].values())), 0
        )

    def test_preliminary_external_certificates(self):
        episodes = {
            ep["id"]: ep for ep in load_jsonl(ROOT / "data/recomp/input/episodes.jsonl")
        }
        rows = load_jsonl(ROOT / "data/recomp/compression/packets.jsonl")
        self.assertEqual(len(episodes), 8)
        self.assertEqual(
            {
                seed: sum((ep["seed"] == seed for ep in episodes.values()))
                for seed in SEEDS
            },
            {seed: 2 for seed in SEEDS},
        )
        expected = {
            "recomp_nq_anchored": 1,
            "recomp_hotpotqa_anchored": 0,
            "recomp_nq_raw": 0,
            "recomp_hotpotqa_raw": 0,
        }
        for method, count in expected.items():
            packets = [row for row in rows if row["method"] == method]
            self.assertEqual(len(packets), 8)
            self.assertEqual(
                sum(
                    (
                        packet_check(episodes[row["id"]], row["events"])["certified"]
                        for row in packets
                    )
                ),
                count,
            )


if __name__ == "__main__":
    unittest.main()
