"""Variance tests for grader functions.

NON-NEGOTIABLE: These tests import from server.graders (the REAL graders).
If they fail, we do not proceed. This is the #1 disqualification prevention.

Run with: pytest tests/test_grader_variance.py -v
All tests must be GREEN before any other code is written.

FIXED: No longer duplicates grader functions locally.
Imports from server.graders to test the ACTUAL production code.
"""

import pytest
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import REAL graders  not local copies
from graders import (
    compute_reward,
    risk_score,
    jaccard_score,
    reviewer_score,
    merge_score
)


#  Mock action for testing (avoids OpenEnv model overhead) 

@dataclass
class MockAction:
    """Lightweight mock action for variance testing."""
    episode_id: str = ""
    risk_level: str = "LOW"
    affected_modules: List[str] = None
    recommended_reviewer: str = ""
    merge_decision: str = ""

    def __post_init__(self):
        if self.affected_modules is None:
            self.affected_modules = []


#  Ground truth fixtures 

@pytest.fixture
def task1_ground_truth():
    return {
        "risk_level": "HIGH",
        "blast_radius": [],
        "recommended_reviewer": "alice",
        "merge_decision": "BLOCK"
    }


@pytest.fixture
def task2_ground_truth():
    return {
        "risk_level": "MEDIUM",
        "blast_radius": ["auth/models.py", "core/middleware.py", "db/connection.py"],
        "recommended_reviewer": "bob",
        "merge_decision": "REQUEST_CHANGES"
    }


@pytest.fixture
def task3_ground_truth():
    return {
        "risk_level": "CRITICAL",
        "blast_radius": ["auth/views.py", "auth/models.py"],
        "recommended_reviewer": "charlie",
        "merge_decision": "BLOCK"
    }


#  Test classes 

class TestRandomAgentScoresLow:
    """Random agent should score in expected low range.
    Proves grader is not trivially constant.
    """

    def test_random_agent_task1(self, task1_ground_truth):
        scores = []
        risk_levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        for _ in range(200):
            action = MockAction(risk_level=random.choice(risk_levels))
            score = compute_reward(action, task1_ground_truth, "task1")
            scores.append(score)

        avg = sum(scores) / len(scores)
        # With 4 levels uniform random: ~25% exact(1.0) + 50% 1(0.5) + 25% 2+(0.0-0.2)
        assert 0.30 <= avg <= 0.70, f"Random task1 avg={avg:.3f} out of range"

    def test_random_agent_task2(self, task2_ground_truth):
        all_modules = [
            "auth/views.py", "auth/models.py", "core/middleware.py",
            "db/connection.py", "api/router.py", "utils/helpers.py"
        ]
        scores = []
        for _ in range(200):
            n = random.randint(0, 4)
            mods = random.sample(all_modules, n) if n > 0 else []
            action = MockAction(affected_modules=mods)
            score = compute_reward(action, task2_ground_truth, "task2")
            scores.append(score)

        avg = sum(scores) / len(scores)
        assert avg <= 0.45, f"Random task2 avg={avg:.3f} too high"

    def test_random_agent_task3(self, task3_ground_truth):
        risk_levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        decisions = ["APPROVE", "BLOCK", "REQUEST_CHANGES"]
        reviewers = ["alice", "bob", "charlie", "david"]
        all_modules = ["auth/views.py", "auth/models.py", "core/middleware.py"]

        scores = []
        for _ in range(200):
            action = MockAction(
                risk_level=random.choice(risk_levels),
                affected_modules=random.sample(
                    all_modules, random.randint(0, len(all_modules))
                ),
                recommended_reviewer=random.choice(reviewers),
                merge_decision=random.choice(decisions)
            )
            score = compute_reward(action, task3_ground_truth, "task3")
            scores.append(score)

        avg = sum(scores) / len(scores)
        assert avg <= 0.50, f"Random task3 avg={avg:.3f} too high"


class TestPerfectAgentScoresHigh:
    """Perfect agent (uses ground truth) should score >= 0.85."""

    def test_perfect_agent_task1(self, task1_ground_truth):
        action = MockAction(risk_level=task1_ground_truth["risk_level"])
        score = compute_reward(action, task1_ground_truth, "task1")
        assert score >= 0.85, f"Perfect task1 score={score:.3f} too low"

    def test_perfect_agent_task2(self, task2_ground_truth):
        action = MockAction(
            affected_modules=task2_ground_truth["blast_radius"].copy()
        )
        score = compute_reward(action, task2_ground_truth, "task2")
        assert score >= 0.85, f"Perfect task2 score={score:.3f} too low"

    def test_perfect_agent_task3(self, task3_ground_truth):
        action = MockAction(
            risk_level=task3_ground_truth["risk_level"],
            affected_modules=task3_ground_truth["blast_radius"].copy(),
            recommended_reviewer=task3_ground_truth["recommended_reviewer"],
            merge_decision=task3_ground_truth["merge_decision"]
        )
        score = compute_reward(action, task3_ground_truth, "task3")
        assert score >= 0.85, f"Perfect task3 score={score:.3f} too low"


class TestPartialAgentScoresMiddle:
    """Partial correctness should give partial credit."""

    def test_one_level_off_task1(self, task1_ground_truth):
        # Ground truth is HIGH  MEDIUM is one off
        action = MockAction(risk_level="MEDIUM")
        score = compute_reward(action, task1_ground_truth, "task1")
        assert 0.45 <= score <= 0.55, f"One-off task1 score={score:.3f}"

    def test_partial_blast_radius_task2(self, task2_ground_truth):
        # Ground truth has 3 modules  predict 2 of them
        action = MockAction(
            affected_modules=["auth/models.py", "core/middleware.py"]
        )
        score = compute_reward(action, task2_ground_truth, "task2")
        # Jaccard = 2/3 = 0.667
        assert 0.55 <= score <= 0.75, f"Partial task2 score={score:.3f}"

    def test_two_correct_fields_task3(self, task3_ground_truth):
        # Risk + blast correct, reviewer + merge wrong
        action = MockAction(
            risk_level=task3_ground_truth["risk_level"],
            affected_modules=task3_ground_truth["blast_radius"].copy(),
            recommended_reviewer="wrong_person",
            merge_decision="APPROVE"   # Wrong AND triggers CRITICAL penalty
        )
        score = compute_reward(action, task3_ground_truth, "task3")
        # risk=0.25*1.0 + blast=0.30*1.0 + reviewer=0 + merge=0 = 0.55
        # But merge=APPROVE on CRITICAL = penalty  merge_score = max(0, 0-0.5)=0
        assert 0.40 <= score <= 0.70, f"Partial task3 score={score:.3f}"


class TestGraderNeverCrashes:
    """Grader must never crash or return None/non-float."""

    @pytest.mark.parametrize("risk", ["LOW", "MEDIUM", "HIGH", "CRITICAL", "INVALID", "", None])
    def test_risk_score_never_crashes(self, risk, task1_ground_truth):
        if risk is None:
            return  # Skip None  Pydantic would reject before grader
        action = MockAction(risk_level=risk)
        score = compute_reward(action, task1_ground_truth, "task1")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_empty_modules_task2(self, task2_ground_truth):
        action = MockAction(affected_modules=[])
        score = compute_reward(action, task2_ground_truth, "task2")
        assert isinstance(score, float)
        assert score == 0.0  # Empty vs non-empty = no match

    def test_duplicate_modules_handled(self, task2_ground_truth):
        action = MockAction(
            affected_modules=["auth/models.py", "auth/models.py", "db/connection.py"]
        )
        score = compute_reward(action, task2_ground_truth, "task2")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize("task", ["task1", "task2", "task3"])
    def test_all_tasks_return_float_in_range(self, task, task3_ground_truth):
        action = MockAction(
            risk_level="HIGH",
            affected_modules=["auth/models.py"],
            recommended_reviewer="alice",
            merge_decision="BLOCK"
        )
        score = compute_reward(action, task3_ground_truth, task)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestCriticalSafetyRule:
    """APPROVE on CRITICAL must apply -0.5 penalty."""

    def test_correct_block_on_critical_no_penalty(self):
        score = merge_score("BLOCK", "BLOCK", "CRITICAL")
        assert score == 1.0

    def test_wrong_approve_on_critical_is_zero(self):
        # Wrong decision + penalty = 0.0 + (-0.5) = max(0, -0.5) = 0.0
        score = merge_score("APPROVE", "BLOCK", "CRITICAL")
        assert score == 0.0

    def test_correct_approve_on_critical_gets_penalty(self):
        # Correct decision + penalty = 1.0 + (-0.5) = 0.5
        score = merge_score("APPROVE", "APPROVE", "CRITICAL")
        assert score == 0.5

    def test_approve_on_non_critical_no_penalty(self):
        score = merge_score("APPROVE", "APPROVE", "HIGH")
        assert score == 1.0

    def test_approve_on_low_risk_no_penalty(self):
        score = merge_score("APPROVE", "APPROVE", "LOW")
        assert score == 1.0


class TestGradersDeterministic:
    """Same inputs must always produce same output (pure functions)."""

    def test_risk_score_deterministic(self):
        scores = [risk_score("HIGH", "CRITICAL") for _ in range(100)]
        assert len(set(scores)) == 1

    def test_jaccard_deterministic(self):
        scores = [
            jaccard_score(["a.py", "b.py"], ["b.py", "c.py"])
            for _ in range(100)
        ]
        assert len(set(scores)) == 1

    def test_compute_reward_deterministic(self):
        gt = {
            "risk_level": "HIGH",
            "blast_radius": ["auth.py"],
            "recommended_reviewer": "alice",
            "merge_decision": "BLOCK"
        }
        action = MockAction(
            risk_level="HIGH",
            affected_modules=["auth.py"],
            recommended_reviewer="alice",
            merge_decision="BLOCK"
        )
        scores = [compute_reward(action, gt, "task3") for _ in range(100)]
        assert len(set(scores)) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])