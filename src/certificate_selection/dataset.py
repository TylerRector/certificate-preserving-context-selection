import copy
import random

from certificate_selection.core import (
    METHODS,
    SEEDS,
    generate,
    leaves,
    packet_check,
    reference,
    sha,
)


def source(ep, builder, target):
    ep = copy.deepcopy(ep)
    rng = random.Random(int(sha([ep["root"], "background"])[:16], 16))
    position = int(sha([ep["root"], "position"])[:8], 16) % 3
    base = len(builder.render(ep, ep["events"]))
    count = max(0, int((target - base) / 83))
    background = []
    for j in range(count + 8):
        at = (
            20 + j / 1000
            if position == 2
            else 650 + j / 1000
            if position == 0
            else 20 + j / 1000
            if j % 2
            else 650 + j / 1000
        )
        background.append(
            {
                "id": "b" + sha([ep["root"], j])[:10],
                "kind": "observation",
                "channel": rng.choice(["status_tool", "agent"]),
                "actor": f"A{j % 8}",
                "at": at,
                "scope": rng.choice(
                    [
                        next(
                            (
                                e["scope"]
                                for e in ep["events"]
                                if e["id"] == ep["action"]
                            )
                        ),
                        f"other_{j}",
                    ]
                ),
                "key": f"flag_{rng.randrange(10000, 99999)}",
                "value": rng.choice(["blue", "amber", None]),
                "detail": "Routine workspace update; independent source scope and timestamp binding are required.",
            }
        )
    n = count
    for _ in range(5):
        es = ep["events"] + background[:n]
        actual = len(builder.render(ep, es))
        error = target - actual
        if abs(error) < 150:
            break
        per = (actual - base) / max(1, n)
        new = max(0, int(n + error / max(1, per)))
        if new > len(background):
            for j in range(len(background), new + 4):
                x = copy.deepcopy(background[-1])
                x["id"] = "b" + sha([ep["root"], j])[:10]
                x["at"] += j / 100000
                background.append(x)
        n = new
    ep["events"] = sorted(
        ep["events"] + background[:n], key=lambda e: (e["at"], e["id"])
    )
    ep["source_tokens"] = len(builder.render(ep, ep["events"]))
    ep["position"] = ["early", "middle", "late"][position]
    assert abs(ep["source_tokens"] - target) < 300
    return ep


def integrity(ep):
    es = ep["events"]
    by = {e["id"]: e for e in es}
    assert len(es) == len(by)
    a = by[ep["action"]]
    f = by[ep["frontier"]]
    p = by[f["policy"]]
    assert p == max(
        (
            e
            for e in es
            if e["kind"] == "policy"
            and e["channel"] == "operator"
            and (e["scope"] == a["scope"])
            and (e["at"] <= a["at"])
        ),
        key=lambda e: e["at"],
    )
    assert set(f["receipts"]) == {
        e["id"]
        for e in es
        if e["kind"] == "receipt"
        and e["channel"] == "collector"
        and (e["scope"] == a["scope"])
        and (e["grant"] == a["grant"])
        and (e["at"] <= a["at"] < e["expires"])
    }
    for atom in leaves(p["expr"]):
        current = max(
            (
                e
                for e in es
                if e["kind"] == "observation"
                and e["channel"] == "status_tool"
                and (e["scope"] == atom["scope"])
                and (e["key"] == atom["key"])
                and (e["at"] <= a["at"])
            ),
            key=lambda e: e["at"],
        )
        assert f["heads"][atom["scope"] + "/" + atom["key"]] == current["id"]
    r = packet_check(ep, es)
    assert r["certified"] and r["possible"] == [reference(ep)]


def packet(ep, method, budget, builder):
    es, status = builder.select(ep, method, budget)
    ids = builder.render(ep, es)
    assert len(ids) <= budget
    original = {e["id"]: e for e in ep["events"]}
    assert all((original[e["id"]] == e for e in es))
    r = packet_check(ep, es)
    truth = reference(ep)
    assert not r["certified"] or r["possible"] == [truth]
    return {
        "id": ep["id"],
        "root": ep["root"],
        "seed": ep["seed"],
        "index": ep["index"],
        "mode": ep["mode"],
        "shape": ep["shape"],
        "arity": ep["arity"],
        "method": method,
        "budget": budget,
        "truth": truth,
        "status": status,
        "tokens": len(ids),
        "prompt_hash": sha(ids),
        "prompt_token_ids": ids,
        "events": es,
        **r,
    }


def tests(builder):
    count = 0
    for seed in SEEDS:
        for i in range(8):
            ep = generate(seed, i, "dev", background=0)
            integrity(ep)
            for m in METHODS:
                row = packet(ep, m, 2048, builder)
                assert not row["certified"] or row["possible"] == [reference(ep)]
            a = generate(seed, i, "dev", "edge_allow", 0)
            v = generate(seed, i, "dev", "edge_deny", 0)
            diffs = [(x, y) for x, y in zip(a["events"], v["events"]) if x != y]
            assert len(diffs) == 1 and diffs[0][0]["kind"] == "receipt"
            assert reference(a) == "ALLOW" and reference(v) == "VIOLATION"
            assert not packet_check(
                a, [e for e in a["events"] if e["kind"] not in ("receipt", "message")]
            )["certified"]
            masked = copy.deepcopy(ep)
            for key in ["id", "root", "seed", "index", "shape", "agents", "arity"]:
                masked[key] = "PRIVATE_METADATA_CHANGED"
            for m in METHODS:
                original = builder.select(ep, m, 2048)[0]
                changed = builder.select(masked, m, 2048)[0]
                assert builder.render(ep, original) == builder.render(masked, changed)
            assert not packet_check(
                ep, [e for e in ep["events"] if e["id"] != ep["frontier"]]
            )["certified"]
            count += 1
    return count
