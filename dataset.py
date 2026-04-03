"""Dataset loader for code review scenarios.

Loads JSON scenario files from the data/ directory.
Each task has 10 scenarios in data/task{1,2,3}/scenario_*.json format.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class Scenario:
    """Represents a code review scenario."""
    scenario_id: str
    task: str
    diff: str
    dependency_map: Dict[str, List[str]]
    file_history: Dict[str, Dict]
    available_reviewers: List[str]
    ground_truth: Dict

    @classmethod
    def from_dict(cls, data: dict) -> 'Scenario':
        """Create Scenario from dictionary."""
        return cls(
            scenario_id=data.get('scenario_id', ''),
            task=data.get('task', ''),
            diff=data.get('diff', ''),
            dependency_map=data.get('dependency_map', {}),
            file_history=data.get('file_history', {}),
            available_reviewers=data.get('available_reviewers', []),
            ground_truth=data.get('ground_truth', {})
        )


class DatasetLoader:
    """Loads and samples scenarios from JSON files."""

    def __init__(self, data_dir: Optional[str] = None, seed: Optional[int] = None):
        """Initialize dataset loader.

        Args:
            data_dir: Path to data directory. Defaults to project root data/.
            seed: Random seed for reproducibility.
        """
        if data_dir is None:
            # Use pathlib to find data dir relative to this file
            self.data_dir = Path(__file__).parent / 'data'
        else:
            self.data_dir = Path(data_dir)

        if seed is not None:
            random.seed(seed)

        self._scenarios: Dict[str, List[Scenario]] = {}
        self._load_all_scenarios()

    def _load_all_scenarios(self) -> None:
        """Load all scenarios from data directory."""
        for task in ['task1', 'task2', 'task3']:
            task_dir = self.data_dir / task
            if not task_dir.exists():
                # Create empty list if directory doesn't exist
                self._scenarios[task] = []
                continue

            scenarios = []
            # Load all JSON files in task directory
            for json_file in sorted(task_dir.glob('scenario_*.json')):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    scenarios.append(Scenario.from_dict(data))
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Warning: Failed to load {json_file}: {e}")

            self._scenarios[task] = scenarios

    def get_scenarios(self, task: str) -> List[Scenario]:
        """Get all scenarios for a task.

        Args:
            task: Task identifier (task1, task2, task3)

        Returns:
            List of scenarios for the task
        """
        return self._scenarios.get(task, [])

    def sample(self, task: str) -> Dict:
        """Sample a random scenario for a task.

        Args:
            task: Task identifier (task1, task2, task3)

        Returns:
            Scenario dictionary
        """
        scenarios = self._scenarios.get(task, [])

        if not scenarios:
            raise ValueError(f"No scenarios found for task: {task}")

        scenario = random.choice(scenarios)
        return {
            'scenario_id': scenario.scenario_id,
            'task': scenario.task,
            'diff': scenario.diff,
            'dependency_map': scenario.dependency_map,
            'file_history': scenario.file_history,
            'available_reviewers': scenario.available_reviewers,
            'ground_truth': scenario.ground_truth
        }

    def get_scenario_by_id(self, scenario_id: str) -> Optional[Dict]:
        """Get a specific scenario by ID.

        Args:
            scenario_id: Scenario ID (e.g., 'task1_001')

        Returns:
            Scenario dictionary or None if not found
        """
        # Parse task from scenario_id (e.g., 'task1_001' -> 'task1')
        task = '_'.join(scenario_id.split('_')[:-1])

        scenarios = self._scenarios.get(task, [])
        for scenario in scenarios:
            if scenario.scenario_id == scenario_id:
                return {
                    'scenario_id': scenario.scenario_id,
                    'task': scenario.task,
                    'diff': scenario.diff,
                    'dependency_map': scenario.dependency_map,
                    'file_history': scenario.file_history,
                    'available_reviewers': scenario.available_reviewers,
                    'ground_truth': scenario.ground_truth
                }
        return None

    def get_task_count(self, task: str) -> int:
        """Get number of scenarios for a task."""
        return len(self._scenarios.get(task, []))

    def get_total_count(self) -> int:
        """Get total number of scenarios across all tasks."""
        return sum(len(scenarios) for scenarios in self._scenarios.values())
