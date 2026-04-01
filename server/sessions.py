"""Episode session store - stateless session management.

CRITICAL: This module holds a global dictionary _sessions keyed by episode_id (UUID).
The environment object NEVER holds episode state — only reads/writes to this store.
This ensures concurrent requests never contaminate each other's episodes.
"""

import uuid
from typing import Dict, Any

# Global session store - keyed by episode_id (UUID)
_sessions: Dict[str, Dict[str, Any]] = {}


def create_session(task: str, scenario: dict) -> str:
    """Create a new episode session.

    Args:
        task: Task identifier (task1, task2, task3)
        scenario: Scenario data dictionary

    Returns:
        episode_id: UUID string for this episode
    """
    episode_id = str(uuid.uuid4())
    _sessions[episode_id] = {
        'task': task,
        'scenario': scenario,
        'step_count': 0,
        'done': False,
        'cumulative_reward': 0.0,
    }
    return episode_id


def get_session(episode_id: str) -> Dict[str, Any]:
    """Retrieve an existing session.

    Args:
        episode_id: UUID string for the episode

    Returns:
        Session dictionary with task, scenario, step_count, etc.

    Raises:
        ValueError: If episode_id not found in session store
    """
    if episode_id not in _sessions:
        raise ValueError(f"Session {episode_id} not found")
    return _sessions[episode_id]


def close_session(episode_id: str) -> None:
    """Close and delete a session.

    Called immediately after step() completes to ensure single-step episodes.

    Args:
        episode_id: UUID string for the episode
    """
    _sessions.pop(episode_id, None)


def list_sessions() -> list:
    """List all active session IDs (for debugging)."""
    return list(_sessions.keys())
