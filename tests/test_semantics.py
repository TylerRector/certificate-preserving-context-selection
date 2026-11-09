import copy
import unittest

from transformers import AutoTokenizer

from core import METHODS, SEEDS, Builder, generate, leaves, packet_check, reference
from dataset import integrity, packet, tests
from storage import TOKENIZER


class CertificateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = Builder(
            AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
        )

    def test_all_seeds_and_private_metadata(self):
        self.assertEqual(tests(self.builder), 32)

    def test_budget_and_certificate_soundness(self):
        for seed in SEEDS:
            for index in range(4):
                episode = generate(seed, index, "test", background=0)
                integrity(episode)
                for method in METHODS:
                    row = packet(episode, method, 2048, self.builder)
                    self.assertLessEqual(row["tokens"], 2048)
                    if row["certified"]:
                        self.assertEqual(row["possible"], [reference(episode)])

    def one_condition_world(self):
        episode = generate(SEEDS[0], 0, "test", "edge_allow", 0)
        by_id = {event["id"]: event for event in episode["events"]}
        frontier = by_id[episode["frontier"]]
        policy = by_id[frontier["policy"]]
        atom = copy.deepcopy(leaves(policy["expr"])[0])
        policy["expr"] = {"op": "AND", "args": [atom]}
        head = atom["scope"] + "/" + atom["key"]
        frontier["heads"] = {head: frontier["heads"][head]}
        current = by_id[frontier["heads"][head]]
        current["value"] = atom["equals"]
        episode["arity"] = 1
        return episode, atom, current

    def test_stale_fact_cannot_replace_snapshot_head(self):
        episode, atom, current = self.one_condition_world()
        current["value"] = "amber" if atom["equals"] == "blue" else "blue"
        for event in episode["events"]:
            if (
                event["kind"] == "observation"
                and event["channel"] == "status_tool"
                and event["scope"] == atom["scope"]
                and event["key"] == atom["key"]
                and event["id"] != current["id"]
            ):
                event["value"] = atom["equals"]
        self.assertEqual(reference(episode), "VIOLATION")
        retained = [
            event for event in episode["events"] if event["id"] != current["id"]
        ]
        self.assertFalse(packet_check(episode, retained)["certified"])

    def test_delegation_must_follow_receipt_time(self):
        episode, _, _ = self.one_condition_world()
        by_id = {event["id"]: event for event in episode["events"]}
        action = by_id[episode["action"]]
        frontier = by_id[episode["frontier"]]
        records = [
            copy.deepcopy(by_id[event_id]) for event_id in frontier["receipts"][:2]
        ]
        records[0].update(sender="A0", receiver="relay", at=400)
        records[1].update(sender="relay", receiver=action["actor"], at=380)
        frontier["receipts"] = [event["id"] for event in records]
        episode["events"] = [
            event for event in episode["events"] if event["kind"] != "receipt"
        ] + records
        self.assertEqual(reference(episode), "VIOLATION")
        self.assertEqual(
            packet_check(episode, episode["events"])["possible"], ["VIOLATION"]
        )
        records[1]["at"] = 450
        self.assertEqual(reference(episode), "ALLOW")
        self.assertEqual(
            packet_check(episode, episode["events"])["possible"], ["ALLOW"]
        )

    def test_missing_snapshot_does_not_certify_unknown(self):
        episode = generate(SEEDS[0], 0, "test", background=0)
        retained = [
            event for event in episode["events"] if event["id"] != episode["frontier"]
        ]
        self.assertFalse(packet_check(episode, retained)["certified"])


if __name__ == "__main__":
    unittest.main()
