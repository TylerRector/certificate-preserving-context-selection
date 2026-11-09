import re

from core import generate, packet_check, reference, sha


def audit_reader(responses, jobs, episodes, builder, requests):
    packets = {(row["dataset"], row["id"], row["method"]): row for row in jobs}
    generations = {row["request_hash"]: row for row in requests}
    truths = {}
    for row in responses:
        key = row["dataset"], row["id"], row["method"]
        packet = packets[key]
        episode = episodes.get(row["id"])
        if episode is None:
            assert row["dataset"] == "control"
            episode = generate(row["seed"], row["index"], "qualification", background=0)
        assert episode["id"] == row["id"]
        if row["id"] not in truths:
            truths[row["id"]] = reference(episode)
        assert truths[row["id"]] == row["truth"]
        events = packet["events"]
        source = {event["id"]: event for event in episode["events"]}
        assert all(source[event["id"]] == event for event in events)
        tokens = builder.render(episode, events)
        assert tokens == packet["prompt_token_ids"]
        assert sha(tokens) == row["request_hash"]
        request = generations[row["request_hash"]]
        assert request["prompt_token_ids"] == tokens
        assert request["raw_output"] == row["raw_output"]
        assert request["output_token_ids"] == row["output_token_ids"]
        assert (
            builder.tok.decode(
                row["output_token_ids"], skip_special_tokens=True
            ).strip()
            == row["raw_output"]
        )
        matches = re.findall(
            r"(?m)^FINAL:\s*(ALLOW|VIOLATION|UNKNOWN)\s*$", row["raw_output"]
        )
        prediction = matches[-1] if matches else "INVALID"
        assert prediction == row["prediction"]
        present = {event["id"] for event in events}
        citations = set(re.findall(r"\b[eb][0-9a-f]{6,16}\b", row["raw_output"]))
        fabricated = sorted(citations - present)
        retained = [
            event
            for event in events
            if event["id"] in citations
            or event["id"] in (episode["action"], episode["frontier"])
        ]
        cited_proof = packet_check(episode, retained)
        full_proof = packet_check(episode, events)
        correct = prediction == truths[row["id"]]
        assert row["correct"] == correct
        assert row["certified"] == full_proof["certified"]
        assert row["packet_supported_correct"] == (correct and full_proof["certified"])
        assert row["citation_certificate"] == cited_proof["certified"]
        assert row["fabricated_ids"] == fabricated
        assert row["cited_ids"] == sorted(citations)
        grounded = (
            correct
            and cited_proof["certified"]
            and cited_proof["possible"] == [truths[row["id"]]]
            and not fabricated
        )
        assert row["grounded_correct"] == grounded
    return len(responses)
