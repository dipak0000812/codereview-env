"""Unit tests for CodeReview environment, sessions, and lifecycle.

Tests are written against the ACTUAL implementation:
- environment.reset() returns a single CodeReviewObservation (not a tuple)
- close_session() actually DELETES the session (not just sets done=True)
- grader() returns float in (EPS, 1-EPS) — never crashes

Run with: pytest tests/test_environment.py -v
"""

import sys
import os
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from environment import CodeReviewEnvironment
from sessions import create_session, get_session, close_session, session_count
from dataset import DatasetLoader
from models import CodeReviewAction, CodeReviewObservation, CodeReviewState
from graders import EPS

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    return CodeReviewEnvironment()


@pytest.fixture
def mock_scenario():
    return {
        "scenario_id": "test_001",
        "task": "task1",
        "diff": "--- a/test.py\n+++ b/test.py\n@@ -1 +1 @@\n-old\n+new",
        "dependency_map": {"test.py": ["dep.py"]},
        "file_history": {"test.py": {"commits_last_30d": 1, "test_coverage": 90}},
        "available_reviewers": ["alice", "bob"],
        "ground_truth": {
            "risk_level": "LOW",
            "blast_radius": [],
            "recommended_reviewer": "alice",
            "merge_decision": "APPROVE"
        }
    }


@pytest.fixture
def mock_dataset(mock_scenario):
    with patch.object(DatasetLoader, 'sample', return_value=mock_scenario):
        yield


# ---------------------------------------------------------------------------
# Reset tests
# ---------------------------------------------------------------------------

class TestEnvironmentReset:
    def test_reset_returns_observation_object(self, env, mock_dataset):
        """reset() must return a single CodeReviewObservation (not a tuple)."""
        result = env.reset('task1')
        assert isinstance(result, CodeReviewObservation), (
            f"Expected CodeReviewObservation, got {type(result)}"
        )

    def test_reset_not_done(self, env, mock_dataset):
        obs = env.reset('task1')
        assert obs.done is False

    def test_reset_reward_zero(self, env, mock_dataset):
        obs = env.reset('task1')
        assert obs.reward == 0.0

    def test_reset_has_episode_id(self, env, mock_dataset):
        obs = env.reset('task1')
        assert obs.episode_id and len(obs.episode_id) > 0

    def test_reset_has_diff(self, env, mock_dataset):
        obs = env.reset('task1')
        assert obs.diff and len(obs.diff) > 0

    def test_reset_task1_has_dependency_map(self, env, mock_dataset):
        obs = env.reset('task1')
        assert isinstance(obs.dependency_map, dict)

    def test_reset_task2_has_dependency_map(self, env, mock_dataset):
        obs = env.reset('task2')
        assert isinstance(obs.dependency_map, dict)

    def test_reset_task3_hides_reviewers_initially(self, env, mock_dataset):
        """Task3 step 1 should not reveal reviewers yet (progressive reveal)."""
        obs = env.reset('task3')
        assert obs.available_reviewers == []

    def test_reset_task3_hides_file_history_initially(self, env, mock_dataset):
        """Task3 step 1 should not reveal file_history yet."""
        obs = env.reset('task3')
        assert obs.file_history == {}

    def test_reset_creates_session(self, env, mock_dataset):
        obs = env.reset('task1')
        session = get_session(obs.episode_id)
        assert session is not None
        assert session["task"] == "task1"

    def test_reset_unique_episode_ids(self, env, mock_dataset):
        obs1 = env.reset('task1')
        obs2 = env.reset('task1')
        assert obs1.episode_id != obs2.episode_id

    def test_reset_sets_state(self, env, mock_dataset):
        obs = env.reset('task1')
        assert env.state.episode_id == obs.episode_id
        assert env.state.step_count == 0
        assert env.state.task == 'task1'


# ---------------------------------------------------------------------------
# Step tests (single-step tasks)
# ---------------------------------------------------------------------------

class TestEnvironmentStepSingleTask:
    def test_step_task1_returns_observation(self, env, mock_dataset):
        obs = env.reset('task1')
        action = CodeReviewAction(
            episode_id=obs.episode_id,
            risk_level="LOW",
            affected_modules=[],
            recommended_reviewer="alice",
            merge_decision="APPROVE"
        )
        result = env.step(action)
        assert isinstance(result, CodeReviewObservation)

    def test_step_task1_done_true(self, env, mock_dataset):
        obs = env.reset('task1')
        action = CodeReviewAction(
            episode_id=obs.episode_id, risk_level="LOW",
            affected_modules=[], recommended_reviewer="alice", merge_decision="APPROVE"
        )
        result = env.step(action)
        assert result.done is True

    def test_step_reward_in_range(self, env, mock_dataset):
        obs = env.reset('task1')
        action = CodeReviewAction(
            episode_id=obs.episode_id, risk_level="MEDIUM",
            affected_modules=[], recommended_reviewer="bob", merge_decision="BLOCK"
        )
        result = env.step(action)
        assert 0.0 < result.reward < 1.0

    def test_step_invalid_episode_raises(self, env, mock_dataset):
        env.reset('task1')
        action = CodeReviewAction(
            episode_id="nonexistent-id",
            risk_level="LOW", affected_modules=[],
            recommended_reviewer="alice", merge_decision="APPROVE"
        )
        with pytest.raises(ValueError):
            env.step(action)

    def test_step_closes_session(self, env, mock_dataset):
        """After single-step task completes, session should be gone."""
        obs = env.reset('task1')
        eid = obs.episode_id
        action = CodeReviewAction(
            episode_id=eid, risk_level="LOW",
            affected_modules=[], recommended_reviewer="alice", merge_decision="APPROVE"
        )
        env.step(action)
        with pytest.raises(ValueError):
            get_session(eid)

    def test_step_task2_done_true(self, env, mock_dataset):
        obs = env.reset('task2')
        action = CodeReviewAction(
            episode_id=obs.episode_id, risk_level="LOW",
            affected_modules=["dep.py"], recommended_reviewer="alice", merge_decision="APPROVE"
        )
        result = env.step(action)
        assert result.done is True


# ---------------------------------------------------------------------------
# Task3 multi-step lifecycle
# ---------------------------------------------------------------------------

class TestTask3MultiStep:
    def test_task3_step1_not_done(self, env, mock_dataset):
        obs = env.reset('task3')
        action = CodeReviewAction(
            episode_id=obs.episode_id, risk_level="LOW",
            affected_modules=[], recommended_reviewer="", merge_decision=""
        )
        result = env.step(action)
        assert result.done is False

    def test_task3_step2_not_done(self, env, mock_dataset):
        obs = env.reset('task3')
        eid = obs.episode_id
        env.step(CodeReviewAction(episode_id=eid, risk_level="LOW",
                                  affected_modules=[], recommended_reviewer="", merge_decision=""))
        result = env.step(CodeReviewAction(episode_id=eid, risk_level="LOW",
                                           affected_modules=[], recommended_reviewer="", merge_decision=""))
        assert result.done is False

    def test_task3_step3_done(self, env, mock_dataset):
        obs = env.reset('task3')
        eid = obs.episode_id
        env.step(CodeReviewAction(episode_id=eid, risk_level="LOW",
                                  affected_modules=[], recommended_reviewer="", merge_decision=""))
        env.step(CodeReviewAction(episode_id=eid, risk_level="LOW",
                                  affected_modules=[], recommended_reviewer="", merge_decision=""))
        result = env.step(CodeReviewAction(episode_id=eid, risk_level="LOW",
                                           affected_modules=[], recommended_reviewer="alice",
                                           merge_decision="APPROVE"))
        assert result.done is True

    def test_task3_step2_reveals_dependency_map(self, env, mock_dataset):
        obs = env.reset('task3')
        eid = obs.episode_id
        result = env.step(CodeReviewAction(episode_id=eid, risk_level="LOW",
                                           affected_modules=[], recommended_reviewer="", merge_decision=""))
        assert isinstance(result.dependency_map, dict)

    def test_task3_step3_reveals_reviewers(self, env, mock_dataset):
        obs = env.reset('task3')
        eid = obs.episode_id
        env.step(CodeReviewAction(episode_id=eid, risk_level="LOW",
                                  affected_modules=[], recommended_reviewer="", merge_decision=""))
        result = env.step(CodeReviewAction(episode_id=eid, risk_level="LOW",
                                           affected_modules=[], recommended_reviewer="", merge_decision=""))
        assert isinstance(result.available_reviewers, list)
        assert len(result.available_reviewers) > 0

    def test_task3_all_rewards_in_range(self, env, mock_dataset):
        obs = env.reset('task3')
        eid = obs.episode_id
        r1 = env.step(CodeReviewAction(episode_id=eid, risk_level="LOW",
                                        affected_modules=[], recommended_reviewer="", merge_decision=""))
        r2 = env.step(CodeReviewAction(episode_id=eid, risk_level="LOW",
                                        affected_modules=[], recommended_reviewer="", merge_decision=""))
        r3 = env.step(CodeReviewAction(episode_id=eid, risk_level="LOW",
                                        affected_modules=[], recommended_reviewer="alice",
                                        merge_decision="APPROVE"))
        for r in [r1, r2, r3]:
            assert 0.0 < r.reward < 1.0


# ---------------------------------------------------------------------------
# Session store tests
# ---------------------------------------------------------------------------

class TestSessionStore:
    def test_create_and_get(self):
        eid = create_session("task1", {"diff": "test-diff"})
        session = get_session(eid)
        assert session["task"] == "task1"
        assert session["scenario"]["diff"] == "test-diff"

    def test_get_missing_raises(self):
        with pytest.raises(ValueError, match="not found"):
            get_session("does-not-exist-at-all")

    def test_close_deletes_session(self):
        """close_session() must REMOVE the session, not just set done=True."""
        eid = create_session("task1", {})
        close_session(eid)
        with pytest.raises(ValueError):
            get_session(eid)

    def test_close_nonexistent_no_error(self):
        """Closing a non-existent session should not raise."""
        close_session("ghost-session-12345")  # Should not raise

    def test_multiple_sessions_isolated(self):
        id1 = create_session("task1", {"x": 1})
        id2 = create_session("task2", {"y": 2})
        assert get_session(id1)["scenario"]["x"] == 1
        assert get_session(id2)["scenario"]["y"] == 2
        close_session(id1)
        with pytest.raises(ValueError):
            get_session(id1)
        assert get_session(id2)["scenario"]["y"] == 2
        close_session(id2)


# ---------------------------------------------------------------------------
# Grader tests
# ---------------------------------------------------------------------------

class TestGrader:
    def test_grader_returns_float(self, env, mock_dataset):
        obs = env.reset('task1')
        eid = obs.episode_id
        actions = [{"risk_level": "LOW", "affected_modules": [],
                    "recommended_reviewer": "alice", "merge_decision": "APPROVE"}]
        score = env.grader("task1", eid, actions)
        assert isinstance(score, float)

    def test_grader_in_open_interval(self, env, mock_dataset):
        obs = env.reset('task1')
        eid = obs.episode_id
        actions = [{"risk_level": "LOW", "affected_modules": [],
                    "recommended_reviewer": "alice", "merge_decision": "APPROVE"}]
        score = env.grader("task1", eid, actions)
        assert 0.0 < score < 1.0

    def test_grader_empty_actions(self, env, mock_dataset):
        score = env.grader("task1", "any-id", [])
        assert 0.0 < score < 1.0

    def test_grader_no_crash_on_expired_session(self, env, mock_dataset):
        """Grader must not crash when session has expired."""
        score = env.grader("task1", "expired-session-xyz", [
            {"risk_level": "HIGH", "affected_modules": [], "merge_decision": "BLOCK",
             "recommended_reviewer": "alice"}
        ])
        assert isinstance(score, float)
        assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# Metadata / close tests
# ---------------------------------------------------------------------------

class TestMetadataAndClose:
    def test_get_metadata_returns_dict(self, env):
        meta = env.get_metadata()
        assert isinstance(meta, dict)
        assert "name" in meta
        assert "version" in meta

    def test_close_does_not_crash(self, env):
        env.close()  # Should not raise


# ---------------------------------------------------------------------------
# Full lifecycle
# ---------------------------------------------------------------------------

class TestFullLifecycle:
    def test_multiple_independent_episodes(self, env, mock_dataset):
        for _ in range(3):
            obs = env.reset('task1')
            assert not obs.done
            action = CodeReviewAction(
                episode_id=obs.episode_id, risk_level="LOW",
                affected_modules=[], recommended_reviewer="alice", merge_decision="APPROVE"
            )
            result = env.step(action)
            assert result.done is True

    def test_task3_full_episode(self, env, mock_dataset):
        obs = env.reset('task3')
        eid = obs.episode_id
        # Step 1
        r1 = env.step(CodeReviewAction(episode_id=eid, risk_level="LOW",
                                        affected_modules=[], recommended_reviewer="", merge_decision=""))
        assert not r1.done
        # Step 2
        r2 = env.step(CodeReviewAction(episode_id=eid, risk_level="LOW",
                                        affected_modules=[], recommended_reviewer="", merge_decision=""))
        assert not r2.done
        # Step 3
        r3 = env.step(CodeReviewAction(episode_id=eid, risk_level="LOW",
                                        affected_modules=[], recommended_reviewer="alice",
                                        merge_decision="APPROVE"))
        assert r3.done