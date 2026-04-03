"""Pure grading functions for CodeReview environment.

All functions are:
- Pure: same input always produces same output
- No external dependencies: no models, no API calls
- Always return float in [0.0, 1.0]
- Never return None

These MUST pass variance tests before being used in production.
"""

from typing import List

# Valid risk levels in order
RISK_LEVELS = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']

# Valid merge decisions
MERGE_DECISIONS = ['APPROVE', 'BLOCK', 'REQUEST_CHANGES']


def risk_score(predicted: str, truth: str) -> float:
    """Score risk level prediction.

    Args:
        predicted: Agent's predicted risk level
        truth: Ground truth risk level

    Returns:
        1.0 if exact match, 0.5 if 1 level, 0.2 if 2 levels, 0.0 otherwise
    """
    if predicted not in RISK_LEVELS or truth not in RISK_LEVELS:
        return 0.0

    diff = abs(RISK_LEVELS.index(predicted) - RISK_LEVELS.index(truth))
    return {0: 1.0, 1: 0.5, 2: 0.2, 3: 0.0}[diff]


def jaccard_score(predicted: List[str], truth: List[str]) -> float:
    """Score blast radius prediction using Jaccard similarity.

    Args:
        predicted: Agent's predicted affected modules
        truth: Ground truth blast radius

    Returns:
        |predicted  truth| / |predicted  truth|
    """
    a, b = set(predicted), set(truth)

    # Both empty = perfect match
    if not a and not b:
        return 1.0

    # One empty, other not = no match
    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def reviewer_score(predicted: str, truth: str) -> float:
    """Score reviewer recommendation.

    Args:
        predicted: Agent's recommended reviewer
        truth: Ground truth reviewer

    Returns:
        1.0 if exact match (ignoring whitespace), 0.0 otherwise
    """
    return 1.0 if predicted.strip().lower() == truth.strip().lower() else 0.0


def merge_score(predicted: str, truth: str, risk_level: str) -> float:
    """Score merge decision with CRITICAL safety rule.

    Args:
        predicted: Agent's merge decision
        truth: Ground truth merge decision
        risk_level: Ground truth risk level (for safety rule)

    Returns:
        1.0 if correct, 0.0 if wrong
        CRITICAL safety rule: APPROVE on CRITICAL = -0.5 penalty
    """
    base = 1.0 if predicted == truth else 0.0

    # CRITICAL safety rule: APPROVE on CRITICAL risk = -0.5 penalty
    # This is ADDITIVE, NOT multiplicative (keeps gradients stable)
    if predicted == 'APPROVE' and risk_level == 'CRITICAL':
        base -= 0.5

    return max(0.0, base)


def compute_reward(action, ground_truth: dict, task: str) -> float:
    """Compute reward based on task type and action.

    Args:
        action: CodeReviewAction from agent
        ground_truth: Ground truth dictionary
        task: Task identifier (task1, task2, task3)

    Returns:
        Reward score in [0.0, 1.0]
    """
    if task == 'task1':
        # Task 1: Risk Classification (Easy)
        return risk_score(action.risk_level, ground_truth['risk_level'])

    elif task == 'task2':
        # Task 2: Blast Radius Identification (Medium)
        return jaccard_score(
            action.affected_modules,
            ground_truth['blast_radius']
        )

    elif task == 'task3':
        # Task 3: Full Review Decision (Hard) - Composite weighted score
        r = 0.0

        # Risk level: 25%
        r += risk_score(
            action.risk_level,
            ground_truth['risk_level']
        ) * 0.25

        # Blast radius: 30%
        r += jaccard_score(
            action.affected_modules,
            ground_truth['blast_radius']
        ) * 0.30

        # Reviewer: 20%
        r += reviewer_score(
            action.recommended_reviewer,
            ground_truth['recommended_reviewer']
        ) * 0.20

        # Merge decision: 25%
        r += merge_score(
            action.merge_decision,
            ground_truth['merge_decision'],
            ground_truth['risk_level']
        ) * 0.25

        return min(max(r, 0.0), 1.0)

    return 0.0


def build_feedback(action, ground_truth: dict, task: str) -> str:
    """Build human-readable feedback for grading breakdown.

    Args:
        action: Agent's action
        ground_truth: Ground truth data
        task: Task identifier

    Returns:
        Human-readable feedback string
    """
    feedback_parts = []

    if task in ['task1', 'task3']:
        predicted = action.risk_level
        truth = ground_truth['risk_level']
        score = risk_score(predicted, truth)
        feedback_parts.append(
            f"Risk level: predicted={predicted}, truth={truth}, score={score}"
        )

    if task in ['task2', 'task3']:
        predicted = action.affected_modules
        truth = ground_truth['blast_radius']
        score = jaccard_score(predicted, truth)
        feedback_parts.append(
            f"Blast radius: predicted={predicted}, truth={truth}, score={score:.2f}"
        )

    if task == 'task3':
        pred_rev = action.recommended_reviewer
        truth_rev = ground_truth['recommended_reviewer']
        rev_score = reviewer_score(pred_rev, truth_rev)
        feedback_parts.append(
            f"Reviewer: predicted={pred_rev}, truth={truth_rev}, score={rev_score}"
        )

        pred_merge = action.merge_decision
        truth_merge = ground_truth['merge_decision']
        merge_sc = merge_score(pred_merge, truth_merge, ground_truth['risk_level'])
        feedback_parts.append(
            f"Merge: predicted={pred_merge}, truth={truth_merge}, score={merge_sc}"
        )

    return " | ".join(feedback_parts)
