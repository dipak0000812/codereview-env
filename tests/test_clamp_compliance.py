"""Phase-2 compliance test: every grader path returns strictly 0 < score < 1."""
import sys
sys.path.insert(0, ".")

from graders import _clamp, risk_score, jaccard_score, reviewer_score, merge_score, compute_reward


def test_clamp_boundaries():
    assert _clamp(0.0) > 0.0
    assert _clamp(1.0) < 1.0
    assert _clamp(-1.0) > 0.0
    assert _clamp(2.0) < 1.0
    assert _clamp(0.5) == 0.5
    assert _clamp(float('nan')) > 0.0
    assert _clamp(None) > 0.0


def test_risk_score_all_paths():
    for p in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
        for t in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
            s = risk_score(p, t)
            assert 0 < s < 1, f"risk_score({p},{t})={s}"
    # Invalid input
    s = risk_score("INVALID", "LOW")
    assert 0 < s < 1, f"risk_score(INVALID,LOW)={s}"


def test_jaccard_all_paths():
    # Both empty
    s = jaccard_score([], [])
    assert 0 < s < 1, f"jaccard([],[])={s}"
    # One empty
    s = jaccard_score([], ["a"])
    assert 0 < s < 1, f"jaccard([],['a'])={s}"
    s = jaccard_score(["a"], [])
    assert 0 < s < 1, f"jaccard(['a'],[])={s}"
    # Perfect match
    s = jaccard_score(["a", "b"], ["a", "b"])
    assert 0 < s < 1, f"jaccard perfect={s}"
    # No overlap
    s = jaccard_score(["a"], ["b"])
    assert 0 < s < 1, f"jaccard no overlap={s}"


def test_reviewer_all_paths():
    s = reviewer_score("alice", "alice")
    assert 0 < s < 1, f"reviewer match={s}"
    s = reviewer_score("alice", "bob")
    assert 0 < s < 1, f"reviewer mismatch={s}"


def test_merge_all_paths():
    for p in ['APPROVE', 'BLOCK', 'REQUEST_CHANGES']:
        for t in ['APPROVE', 'BLOCK', 'REQUEST_CHANGES']:
            for r in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
                s = merge_score(p, t, r)
                assert 0 < s < 1, f"merge({p},{t},{r})={s}"


class FakeAction:
    def __init__(self, **kw):
        self.risk_level = kw.get("risk_level", "LOW")
        self.affected_modules = kw.get("affected_modules", [])
        self.recommended_reviewer = kw.get("recommended_reviewer", "")
        self.merge_decision = kw.get("merge_decision", "")


def test_compute_reward_all_tasks():
    gt = {
        "risk_level": "HIGH",
        "blast_radius": ["auth", "db"],
        "recommended_reviewer": "alice",
        "merge_decision": "BLOCK",
    }
    for task in ["task1", "task2", "task3", "unknown"]:
        action = FakeAction(
            risk_level="LOW",
            affected_modules=["auth"],
            recommended_reviewer="bob",
            merge_decision="APPROVE",
        )
        s = compute_reward(action, gt, task)
        assert 0 < s < 1, f"compute_reward({task})={s}"

    # Perfect match
    action = FakeAction(
        risk_level="HIGH",
        affected_modules=["auth", "db"],
        recommended_reviewer="alice",
        merge_decision="BLOCK",
    )
    for task in ["task1", "task2", "task3"]:
        s = compute_reward(action, gt, task)
        assert 0 < s < 1, f"compute_reward perfect({task})={s}"


if __name__ == "__main__":
    test_clamp_boundaries()
    test_risk_score_all_paths()
    test_jaccard_all_paths()
    test_reviewer_all_paths()
    test_merge_all_paths()
    test_compute_reward_all_tasks()
    print("ALL COMPLIANCE TESTS PASSED: 0 < score < 1 for every path")
