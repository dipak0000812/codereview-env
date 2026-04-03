"""Unit tests for the CodeReview environment (stateless, session store)."""

import sys
import os
import pytest
from unittest.mock import patch

# Add project root to path so that modules are found
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from environment import CodeReviewEnvironment
from sessions import create_session, get_session, close_session
from dataset import DatasetLoader
from models import CodeReviewAction, CodeReviewObservation, CodeReviewState


@pytest.fixture
def env():
    """Create a fresh environment instance for each test."""
    return CodeReviewEnvironment()


@pytest.fixture
def mock_dataset():
    """Mock dataset to return predictable scenarios."""
    with patch.object(DatasetLoader, 'sample') as mock_sample:
        mock_sample.return_value = {
            "scenario_id": "test_001",
            "task": "task1",
            "diff": "--- a/test.py\n+++ b/test.py\n@@ -1 +1 @@\n-old\n+new",
            "dependency_map": {"test.py": []},
            "file_history": {},
            "available_reviewers": ["alice", "bob"],
            "ground_truth": {
                "risk_level": "LOW",
                "blast_radius": [],
                "recommended_reviewer": "alice",
                "merge_decision": "APPROVE"
            }
        }
        yield mock_sample


class TestEnvironmentReset:
    def test_reset_returns_tuple(self, env, mock_dataset):
        result = env.reset('task1')
        assert isinstance(result, tuple), "reset() should return a tuple"
        assert len(result) == 2

    def test_reset_creates_session(self, env, mock_dataset):
        obs, episode_id = env.reset('task1')
        session = get_session(episode_id)
        assert session is not None
        assert session["task"] == "task1"

    def test_reset_returns_valid_observation(self, env, mock_dataset):
        obs, episode_id = env.reset('task1')
        assert isinstance(obs, CodeReviewObservation)
        assert obs.episode_id == episode_id
        assert obs.task == "task1"

    def test_reset_task1(self, env, mock_dataset):
        obs, episode_id = env.reset('task1')
        assert obs.task == "task1"
        assert obs.dependency_map is not None

    def test_reset_task2(self, env, mock_dataset):
        obs, episode_id = env.reset('task2')
        assert obs.task == "task2"
        assert obs.dependency_map is not None

    def test_reset_task3(self, env, mock_dataset):
        obs, episode_id = env.reset('task3')
        assert obs.task == "task3"
        # Step 1 of task3: available_reviewers is empty (revealed later)
        assert obs.available_reviewers == []


class TestEnvironmentStep:
    def test_step_requires_episode_id(self, env, mock_dataset):
        obs, episode_id = env.reset('task1')
        action = CodeReviewAction(
            episode_id=episode_id,
            risk_level="LOW",
            affected_modules=[],
            recommended_reviewer="alice",
            merge_decision="APPROVE"
        )
        # step() only takes action (episode_id is inside action)
        result = env.step(action)
        assert result.done is True

    def test_step_returns_done_true(self, env, mock_dataset):
        obs, episode_id = env.reset('task1')
        action = CodeReviewAction(
            episode_id=episode_id,
            risk_level="LOW",
            affected_modules=[],
            recommended_reviewer="alice",
            merge_decision="APPROVE"
        )
        result = env.step(action)
        assert result.done is True

    def test_step_closes_session(self, env, mock_dataset):
        obs, episode_id = env.reset('task1')
        action = CodeReviewAction(
            episode_id=episode_id,
            risk_level="LOW",
            affected_modules=[],
            recommended_reviewer="alice",
            merge_decision="APPROVE"
        )
        env.step(action)
        with pytest.raises(ValueError):
            get_session(episode_id)

    def test_step_episode_id_mismatch(self, env, mock_dataset):
        obs, episode_id = env.reset('task1')
        # Use a different episode_id inside action
        action = CodeReviewAction(
            episode_id="wrong-id",
            risk_level="LOW",
            affected_modules=[],
            recommended_reviewer="alice",
            merge_decision="APPROVE"
        )
        with pytest.raises(ValueError, match="Session wrong-id not found"):
            env.step(action)


class TestEnvironmentState:
    def test_state_returns_state_object(self, env):
        state = env.state
        # State should have episode_id and step_count (as per your CodeReviewState)
        assert hasattr(state, 'episode_id')
        assert hasattr(state, 'step_count')


class TestSessionStore:
    def test_create_session(self):
        episode_id = create_session("task1", {"diff": "test"})
        assert episode_id is not None
        session = get_session(episode_id)
        assert session["task"] == "task1"
        assert session["scenario"]["diff"] == "test"

    def test_get_session(self):
        episode_id = create_session("task2", {"foo": "bar"})
        session = get_session(episode_id)
        # Data is stored under "scenario" key
        assert session["scenario"]["foo"] == "bar"

    def test_get_session_not_found(self):
        with pytest.raises(ValueError):
            get_session("nonexistent")

    def test_close_session(self):
        episode_id = create_session("task3", {})
        close_session(episode_id)
        with pytest.raises(ValueError):
            get_session(episode_id)

    def test_multiple_sessions_isolated(self):
        id1 = create_session("task1", {"x": 1})
        id2 = create_session("task2", {"y": 2})
        assert get_session(id1)["scenario"]["x"] == 1
        assert get_session(id2)["scenario"]["y"] == 2
        close_session(id1)
        with pytest.raises(ValueError):
            get_session(id1)
        assert get_session(id2)["scenario"]["y"] == 2


class TestEpisodeLifecycle:
    def test_single_episode_lifecycle(self, env, mock_dataset):
        obs, episode_id = env.reset('task1')
        assert not obs.done
        action = CodeReviewAction(
            episode_id=episode_id,
            risk_level="LOW",
            affected_modules=[],
            recommended_reviewer="alice",
            merge_decision="APPROVE"
        )
        result = env.step(action)
        assert result.done is True

    def test_multiple_episodes(self, env, mock_dataset):
        # First episode
        obs1, ep1 = env.reset('task1')
        action1 = CodeReviewAction(
            episode_id=ep1,
            risk_level="LOW",
            affected_modules=[],
            recommended_reviewer="alice",
            merge_decision="APPROVE"
        )
        result1 = env.step(action1)
        assert result1.done is True

        # Second episode (should be fresh)
        obs2, ep2 = env.reset('task1')
        assert ep2 != ep1
        action2 = CodeReviewAction(
            episode_id=ep2,
            risk_level="MEDIUM",
            affected_modules=[],
            recommended_reviewer="bob",
            merge_decision="BLOCK"
        )
        result2 = env.step(action2)
        assert result2.done is True