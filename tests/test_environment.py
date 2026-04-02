"""Environment tests for CodeReview environment.

Tests the stateless environment implementation.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.environment import CodeReviewEnvironment
from server.sessions import create_session, get_session, close_session, _sessions
from models import CodeReviewAction


@pytest.fixture
def env():
    """Create a fresh environment instance."""
    return CodeReviewEnvironment()


@pytest.fixture(autouse=True)
def clear_sessions():
    """Clear all sessions before each test."""
    _sessions.clear()
    yield
    _sessions.clear()


class TestEnvironmentReset:
    """Tests for environment reset() method."""

    def test_reset_returns_tuple(self, env):
        """reset() should return a tuple of (observation, episode_id)."""
        result = env.reset('task1')
        assert isinstance(result, tuple), "reset() should return a tuple"
        assert len(result) == 2, "reset() should return (observation, episode_id)"
        obs, episode_id = result
        assert isinstance(episode_id, str), "episode_id should be a string"
        assert len(episode_id) > 0, "episode_id should not be empty"

    def test_reset_creates_session(self, env):
        """reset() should create a session in the store."""
        obs, episode_id = env.reset('task1')
        session = get_session(episode_id)
        assert session is not None
        assert session['task'] == 'task1'
        assert 'scenario' in session

    def test_reset_returns_valid_observation(self, env):
        """reset() should return a valid observation."""
        obs, episode_id = env.reset('task1')
        assert obs.episode_id == episode_id
        assert obs.task == 'task1'
        assert obs.done is False
        assert obs.reward == 0.0
        assert isinstance(obs.diff, str)
        assert isinstance(obs.dependency_map, dict)

    def test_reset_task1(self, env):
        """reset() should work for task1."""
        obs, episode_id = env.reset('task1')
        assert obs.task == 'task1'
        assert len(obs.diff) > 0

    def test_reset_task2(self, env):
        """reset() should work for task2."""
        obs, episode_id = env.reset('task2')
        assert obs.task == 'task2'
        assert len(obs.dependency_map) > 0

    def test_reset_task3(self, env):
        """reset() should work for task3."""
        obs, episode_id = env.reset('task3')
        assert obs.task == 'task3'
        assert len(obs.available_reviewers) > 0


class TestEnvironmentStep:
    """Tests for environment step() method."""

    def test_step_requires_episode_id(self, env):
        """step() requires episode_id as second argument."""
        obs, episode_id = env.reset('task1')
        action = CodeReviewAction(
            episode_id=episode_id,
            risk_level='HIGH'
        )
        result = env.step(action, episode_id)
        assert isinstance(result.reward, float)
        assert 0.0 <= result.reward <= 1.0

    def test_step_returns_done_true(self, env):
        """step() should return done=True."""
        obs, episode_id = env.reset('task1')
        action = CodeReviewAction(
            episode_id=episode_id,
            risk_level='HIGH'
        )
        result = env.step(action, episode_id)
        assert result.done is True

    def test_step_closes_session(self, env):
        """step() should close the session after execution."""
        obs, episode_id = env.reset('task1')
        action = CodeReviewAction(
            episode_id=episode_id,
            risk_level='HIGH'
        )
        env.step(action, episode_id)
        with pytest.raises(ValueError):
            get_session(episode_id)

    def test_step_episode_id_mismatch(self, env):
        """step() should fail with mismatched episode_id."""
        obs, episode_id = env.reset('task1')
        action = CodeReviewAction(
            episode_id='wrong-id',
            risk_level='HIGH'
        )
        with pytest.raises(ValueError):
            env.step(action, episode_id)


class TestEnvironmentState:
    """Tests for environment state property."""

    def test_state_returns_state_object(self, env):
        """state property should return CodeReviewState."""
        state = env.state
        assert state is not None
        assert hasattr(state, 'episode_id')
        assert hasattr(state, 'step_count')
        assert hasattr(state, 'task')


class TestSessionStore:
    """Tests for session store functionality."""

    def test_create_session(self):
        """create_session should create and return episode_id."""
        scenario = {'diff': 'test', 'ground_truth': {'risk_level': 'LOW'}}
        episode_id = create_session('task1', scenario)
        assert isinstance(episode_id, str)
        assert len(episode_id) > 0

    def test_get_session(self):
        """get_session should retrieve session by episode_id."""
        scenario = {'diff': 'test', 'ground_truth': {'risk_level': 'LOW'}}
        episode_id = create_session('task1', scenario)
        session = get_session(episode_id)
        assert session['task'] == 'task1'
        assert session['scenario'] == scenario

    def test_get_session_not_found(self):
        """get_session should raise error for unknown episode_id."""
        with pytest.raises(ValueError):
            get_session('nonexistent-id')

    def test_close_session(self):
        """close_session should remove session."""
        scenario = {'diff': 'test', 'ground_truth': {'risk_level': 'LOW'}}
        episode_id = create_session('task1', scenario)
        close_session(episode_id)
        with pytest.raises(ValueError):
            get_session(episode_id)

    def test_multiple_sessions_isolated(self):
        """Multiple sessions should be isolated."""
        scenario = {'diff': 'test', 'ground_truth': {'risk_level': 'LOW'}}
        ep1 = create_session('task1', scenario)
        ep2 = create_session('task2', scenario)
        ep3 = create_session('task3', scenario)

        assert get_session(ep1)['task'] == 'task1'
        assert get_session(ep2)['task'] == 'task2'
        assert get_session(ep3)['task'] == 'task3'


class TestEpisodeLifecycle:
    """Tests for complete episode lifecycle."""

    def test_single_episode_lifecycle(self, env):
        """Test complete reset -> step -> done cycle."""
        # Reset
        obs, episode_id = env.reset('task1')
        assert obs.done is False
        assert obs.reward == 0.0

        # Step
        action = CodeReviewAction(
            episode_id=episode_id,
            risk_level='HIGH'
        )
        result = env.step(action, episode_id)
        assert result.done is True
        assert result.episode_id == episode_id

    def test_multiple_episodes(self, env):
        """Test multiple sequential episodes."""
        for i in range(3):
            obs, episode_id = env.reset('task1')
            assert obs.done is False

            action = CodeReviewAction(
                episode_id=episode_id,
                risk_level='HIGH'
            )
            result = env.step(action, episode_id)
            assert result.done is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
