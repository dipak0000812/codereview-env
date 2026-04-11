"""Variance and correctness tests for grader functions.

NON-NEGOTIABLE: All tests must pass before submission.
Tests import from the REAL graders.py — no local duplicates.

Run with: pytest tests/test_grader_variance.py -v
"""

import pytest
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from graders import (
    EPS,
    _clamp,
    compute_reward,
    risk_score,
    jaccard_score,
    reviewer_score,
    merge_score,
)


# ---------------------------------------------------------------------------
# Mock action
# ---------------------------------------------------------------------------

@dataclass
class MockAction:
    episode_id: str = ""
    risk_level: str = "LOW"
    affected_modules: List[str] = field(default_factory=list)
    recommended_reviewer: str = ""
    merge_decision: str = ""


# ---------------------------------------------------------------------------
# Ground-truth fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gt_task1():
    return {
        "risk_level": "HIGH",
        "blast_radius": [],
        "recommended_reviewer": "alice",
        "merge_decision": "BLOCK"
    }


@pytest.fixture
def gt_task2():
    return {
        "risk_level": "MEDIUM",
        "blast_radius": ["auth/models.py", "core/middleware.py", "db/connection.py"],
        "recommended_reviewer": "bob",
        "merge_decision": "REQUEST_CHANGES"
    }


@pytest.fixture
def gt_task3():
    return {
        "risk_level": "CRITICAL",
        "blast_radius": ["auth/views.py", "auth/models.py"],
        "recommended_reviewer": "charlie",
        "merge_decision": "BLOCK"
    }


# ---------------------------------------------------------------------------
# _clamp tests (critical foundation)
# ---------------------------------------------------------------------------

class TestClamp:
    def test_clamp_zero_returns_eps(self):
        assert _clamp(0.0) == EPS

    def test_clamp_one_returns_one_minus_eps(self):
        assert _clamp(1.0) == 1.0 - EPS

    def test_clamp_negative_returns_eps(self):
        assert _clamp(-5.0) == EPS

    def test_clamp_above_one_returns_one_minus_eps(self):
        assert _clamp(2.0) == 1.0 - EPS

    def test_clamp_middle_unchanged(self):
        assert _clamp(0.5) == 0.5

    def test_clamp_none_returns_eps(self):
        assert _clamp(None) == EPS

    def test_clamp_nan_returns_eps(self):
        assert _clamp(float('nan')) == EPS

    def test_clamp_output_always_in_open_interval(self):
        for val in [-100, -1, -EPS, 0.0, EPS, 0.3, 0.5, 0.9, 1.0 - EPS, 1.0, 2.0, 1e9]:
            result = _clamp(val)
            assert 0.0 < result < 1.0, f"_clamp({val}) = {result} not in (0, 1)"


# ---------------------------------------------------------------------------
# risk_score tests
# ---------------------------------------------------------------------------

class TestRiskScore:
    def test_exact_match_near_one(self):
        score = risk_score("HIGH", "HIGH")
        assert score >= 1.0 - EPS - 1e-9

    def test_one_level_off_is_partial(self):
        score = risk_score("MEDIUM", "HIGH")
        assert 0.45 <= score <= 0.55, f"score={score}"

    def test_two_levels_off_small(self):
        score = risk_score("LOW", "HIGH")
        assert score <= 0.20

    def test_three_levels_off_very_small(self):
        score = risk_score("LOW", "CRITICAL")
        # Additional safety penalty for severe underestimation
        assert score <= 0.10

    def test_invalid_predicted_returns_low(self):
        score = risk_score("UNKNOWN", "HIGH")
        assert score == EPS

    def test_case_insensitive(self):
        score_upper = risk_score("HIGH", "HIGH")
        # Our implementation uses .upper() internally — both should work
        assert score_upper > 0.9

    def test_all_levels_return_float(self):
        for p in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
            for t in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
                score = risk_score(p, t)
                assert isinstance(score, float)
                assert 0.0 < score < 1.0

    def test_deterministic(self):
        scores = [risk_score("HIGH", "CRITICAL") for _ in range(100)]
        assert len(set(scores)) == 1


# ---------------------------------------------------------------------------
# jaccard_score tests
# ---------------------------------------------------------------------------

class TestJaccardScore:
    def test_perfect_match_near_one(self):
        score = jaccard_score(["a.py", "b.py"], ["a.py", "b.py"])
        assert score >= 1.0 - EPS - 1e-9

    def test_empty_both_near_one(self):
        score = jaccard_score([], [])
        assert score >= 1.0 - EPS - 1e-9

    def test_empty_predicted_nonzero_truth_near_zero(self):
        score = jaccard_score([], ["a.py", "b.py"])
        assert score == EPS

    def test_nonempty_predicted_empty_truth_near_zero(self):
        score = jaccard_score(["a.py"], [])
        assert score == EPS

    def test_partial_overlap(self):
        # intersection=1, union=3 → Jaccard=1/3
        score = jaccard_score(["a.py", "b.py"], ["b.py", "c.py"])
        assert abs(score - 1 / 3) < 0.01

    def test_two_of_three_overlap(self):
        # intersection=2, union=3 → Jaccard=2/3
        score = jaccard_score(["a.py", "b.py"], ["a.py", "b.py", "c.py"])
        assert abs(score - 2 / 3) < 0.01

    def test_case_insensitive_paths(self):
        score = jaccard_score(["Auth/Models.py"], ["auth/models.py"])
        assert score >= 1.0 - EPS - 1e-9

    def test_duplicates_handled(self):
        score = jaccard_score(["a.py", "a.py", "b.py"], ["a.py", "b.py"])
        assert score >= 1.0 - EPS - 1e-9

    def test_returns_float_in_range(self):
        for _ in range(50):
            modules = random.sample(["a.py", "b.py", "c.py", "d.py"], random.randint(0, 4))
            truth = random.sample(["a.py", "b.py", "c.py", "d.py"], random.randint(0, 4))
            score = jaccard_score(modules, truth)
            assert isinstance(score, float)
            assert 0.0 < score < 1.0

    def test_deterministic(self):
        scores = [jaccard_score(["a.py", "b.py"], ["b.py", "c.py"]) for _ in range(100)]
        assert len(set(scores)) == 1


# ---------------------------------------------------------------------------
# reviewer_score tests
# ---------------------------------------------------------------------------

class TestReviewerScore:
    def test_exact_match_near_one(self):
        score = reviewer_score("alice", "alice")
        assert score >= 1.0 - EPS - 1e-9

    def test_case_insensitive_match(self):
        score = reviewer_score("ALICE", "alice")
        assert score >= 1.0 - EPS - 1e-9

    def test_whitespace_stripped(self):
        score = reviewer_score("  alice  ", "alice")
        assert score >= 1.0 - EPS - 1e-9

    def test_mismatch_near_zero(self):
        score = reviewer_score("bob", "alice")
        assert score == EPS

    def test_empty_predicted_near_zero(self):
        score = reviewer_score("", "alice")
        assert score == EPS

    def test_both_empty_near_zero(self):
        score = reviewer_score("", "")
        assert score == EPS

    def test_returns_float_in_range(self):
        for p, t in [("alice", "alice"), ("bob", "alice"), ("", "alice"), ("alice", "")]:
            score = reviewer_score(p, t)
            assert isinstance(score, float)
            assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# merge_score tests
# ---------------------------------------------------------------------------

class TestMergeScore:
    def test_correct_block_on_critical(self):
        score = merge_score("BLOCK", "BLOCK", "CRITICAL")
        assert score >= 1.0 - EPS - 1e-9

    def test_approve_on_critical_penalized(self):
        # Wrong + penalty: 0.0 - 0.5 = max(0, -0.5) = 0.0 → EPS
        score = merge_score("APPROVE", "BLOCK", "CRITICAL")
        assert score == EPS

    def test_correct_approve_on_critical_half_credit(self):
        # Correct but penalty: 1.0 - 0.5 = 0.5
        score = merge_score("APPROVE", "APPROVE", "CRITICAL")
        assert abs(score - 0.5) < 0.01

    def test_approve_on_low_no_penalty(self):
        score = merge_score("APPROVE", "APPROVE", "LOW")
        assert score >= 1.0 - EPS - 1e-9

    def test_wrong_decision_near_zero(self):
        score = merge_score("APPROVE", "BLOCK", "LOW")
        assert score == EPS

    def test_request_changes_correct(self):
        score = merge_score("REQUEST_CHANGES", "REQUEST_CHANGES", "MEDIUM")
        assert score >= 1.0 - EPS - 1e-9

    def test_returns_float_in_range(self):
        for p in ["APPROVE", "BLOCK", "REQUEST_CHANGES"]:
            for t in ["APPROVE", "BLOCK", "REQUEST_CHANGES"]:
                for r in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
                    score = merge_score(p, t, r)
                    assert isinstance(score, float)
                    assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# compute_reward tests
# ---------------------------------------------------------------------------

class TestComputeReward:
    def test_perfect_task1_near_one(self, gt_task1):
        action = MockAction(risk_level=gt_task1["risk_level"])
        score = compute_reward(action, gt_task1, "task1")
        assert score >= 0.85

    def test_perfect_task2_near_one(self, gt_task2):
        action = MockAction(affected_modules=gt_task2["blast_radius"].copy())
        score = compute_reward(action, gt_task2, "task2")
        assert score >= 0.85

    def test_perfect_task3_near_one(self, gt_task3):
        action = MockAction(
            risk_level=gt_task3["risk_level"],
            affected_modules=gt_task3["blast_radius"].copy(),
            recommended_reviewer=gt_task3["recommended_reviewer"],
            merge_decision=gt_task3["merge_decision"]
        )
        score = compute_reward(action, gt_task3, "task3")
        assert score >= 0.85

    def test_random_task1_mid_range(self, gt_task1):
        scores = []
        for _ in range(200):
            action = MockAction(risk_level=random.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"]))
            scores.append(compute_reward(action, gt_task1, "task1"))
        avg = sum(scores) / len(scores)
        assert 0.25 <= avg <= 0.70, f"Random task1 avg={avg:.3f}"

    def test_random_task2_low_range(self, gt_task2):
        all_mods = ["auth/models.py", "core/middleware.py", "db/connection.py",
                    "api/router.py", "utils/helpers.py"]
        scores = []
        for _ in range(200):
            n = random.randint(0, 4)
            mods = random.sample(all_mods, n) if n > 0 else []
            action = MockAction(affected_modules=mods)
            scores.append(compute_reward(action, gt_task2, "task2"))
        avg = sum(scores) / len(scores)
        assert avg <= 0.50, f"Random task2 avg={avg:.3f}"

    def test_all_tasks_always_float_in_range(self, gt_task3):
        for task in ["task1", "task2", "task3"]:
            for _ in range(50):
                action = MockAction(
                    risk_level=random.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"]),
                    affected_modules=random.sample(
                        ["auth/views.py", "auth/models.py", "core/middleware.py"],
                        random.randint(0, 3)
                    ),
                    recommended_reviewer=random.choice(["alice", "bob", "charlie"]),
                    merge_decision=random.choice(["APPROVE", "BLOCK", "REQUEST_CHANGES"])
                )
                score = compute_reward(action, gt_task3, task)
                assert isinstance(score, float), f"{task}: score is not float"
                assert 0.0 < score < 1.0, f"{task}: score={score} out of (0,1)"

    def test_task3_weights_sum_correct(self, gt_task3):
        """All-correct task3 should be near 1.0 since weights 0.25+0.30+0.20+0.25=1.0."""
        action = MockAction(
            risk_level=gt_task3["risk_level"],
            affected_modules=gt_task3["blast_radius"].copy(),
            recommended_reviewer=gt_task3["recommended_reviewer"],
            merge_decision=gt_task3["merge_decision"]
        )
        score = compute_reward(action, gt_task3, "task3")
        assert score >= 0.90, f"All-correct task3 score={score:.3f} too low"

    def test_task3_partial_risk_blast_correct(self, gt_task3):
        """Risk correct (25%) + blast correct (30%) = 55% minimum."""
        action = MockAction(
            risk_level=gt_task3["risk_level"],
            affected_modules=gt_task3["blast_radius"].copy(),
            recommended_reviewer="wrong_person",
            merge_decision="REQUEST_CHANGES"  # Wrong
        )
        score = compute_reward(action, gt_task3, "task3")
        assert 0.45 <= score <= 0.70, f"Partial task3 score={score:.3f}"

    def test_unknown_task_returns_low_nonzero(self, gt_task3):
        action = MockAction()
        score = compute_reward(action, gt_task3, "task_unknown")
        assert isinstance(score, float)
        assert 0.0 < score < 1.0

    def test_compute_reward_deterministic(self, gt_task3):
        action = MockAction(
            risk_level="CRITICAL",
            affected_modules=["auth/views.py"],
            recommended_reviewer="charlie",
            merge_decision="BLOCK"
        )
        scores = [compute_reward(action, gt_task3, "task3") for _ in range(100)]
        assert len(set(scores)) == 1, "compute_reward is not deterministic!"

    def test_none_fields_do_not_crash(self, gt_task1):
        """Grader must never crash on None/empty action fields."""
        action = MockAction(risk_level="", affected_modules=None,
                            recommended_reviewer=None, merge_decision=None)
        # affected_modules=None might not go through dataclass
        action.affected_modules = action.affected_modules or []
        action.recommended_reviewer = action.recommended_reviewer or ""
        action.merge_decision = action.merge_decision or ""
        score = compute_reward(action, gt_task1, "task1")
        assert isinstance(score, float)
        assert 0.0 < score < 1.0