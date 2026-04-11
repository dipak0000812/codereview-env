"""Episode session store - stateless session management.

CRITICAL: This module holds a global dictionary _sessions keyed by episode_id (UUID).
The environment object NEVER holds episode state — only reads/writes to this store.
This ensures concurrent requests never contaminate each other's episodes.

Fix: close_session() now actually DELETES the session from the store to prevent
memory leaks and stale session bugs.
"""

import time
import uuid
from threading import Lock
from typing import Dict, Any, Optional

# Global session store - keyed by episode_id (UUID)
_sessions: Dict[str, Dict[str, Any]] = {}
_lock: Lock = Lock()

# Sessions older than this are purged automatically (30 minutes)
SESSION_TTL_SECONDS = 1800


def create_session(task: str, scenario: dict, episode_id: Optional[str] = None) -> str:
    """Create a new episode session.

    Args:
        task: Task identifier (task1, task2, task3)
        scenario: Scenario data dictionary
        episode_id: Optional UUID string (if not provided, one is generated)

    Returns:
        episode_id: UUID string for this episode
    """
    if episode_id is None:
        episode_id = str(uuid.uuid4())

    with _lock:
        _sessions[episode_id] = {
            'task': task,
            'scenario': scenario,
            'step_count': 0,
            'done': False,
            'cumulative_reward': 0.0,
            'created_at': time.time(),
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
    with _lock:
        if episode_id not in _sessions:
            raise ValueError(f"Session {episode_id} not found")
        return _sessions[episode_id]


def close_session(episode_id: str) -> None:
    """Close and DELETE a session from the store.

    Called immediately after step() completes to ensure session cleanup.
    Previously this only set done=True — now it actually removes the session
    to prevent memory leaks.

    Args:
        episode_id: UUID string for the episode
    """
    with _lock:
        _sessions.pop(episode_id, None)


def list_sessions() -> list:
    """List all active session IDs (for debugging)."""
    with _lock:
        return list(_sessions.keys())


def purge_stale_sessions() -> int:
    """Remove sessions older than SESSION_TTL_SECONDS.

    Returns:
        Number of sessions purged
    """
    now = time.time()
    purged = 0
    with _lock:
        stale = [
            eid for eid, s in _sessions.items()
            if now - s.get('created_at', now) > SESSION_TTL_SECONDS
        ]
        for eid in stale:
            del _sessions[eid]
            purged += 1
    return purged


def session_count() -> int:
    """Return the number of active sessions."""
    with _lock:
        return len(_sessions)
