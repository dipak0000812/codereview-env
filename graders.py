"""Pure grading functions for CodeReview environment.

All functions are:
- Pure: same input always produces same output (deterministic)
- Side-effect free: no external dependencies, no API calls, no I/O
- Bounded: always return float strictly in open interval (0, 1)
- Crash-safe: never raise exceptions on malformed input

Grading philosophy:
- Exact match = near-perfect score (1.0 - EPS)
- Proportional partial credit for near-misses
- Safety penalties for high-severity classification errors
"""

from typing import List, Optional, Set

# Valid risk levels in severity order (lowest → highest)
RISK_LEVELS = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']

# Valid merge decisions
MERGE_DECISIONS = ['APPROVE', 'BLOCK', 'REQUEST_CHANGES']

# Epsilon — prevents grader from returning exact 0.0 or 1.0
EPS = 1e-6

# Adjacency score table: how much partial credit for off-by-N risk classification
_RISK_ADJACENCY = {0: 1.0, 1: 0.5, 2: 0.15, 3: 0.0}


# ---------------------------------------------------------------------------
# Core utility
# ---------------------------------------------------------------------------

def _clamp(score: float) -> float:
    """Clamp to strict open interval (0, 1). Never returns 0.0 or 1.0 exactly.

    Args:
        score: Raw score value (may be None, NaN, <0, or >1)

    Returns:
        Float in (EPS, 1 - EPS)
    """
    if score is None or score != score:  # NaN check (NaN != NaN is True)
        return EPS
    score = float(score)
    if score <= 0.0:
        return EPS
    if score >= 1.0:
        return 1.0 - EPS
    return score


def _normalize_str(s: str) -> str:
    """Lowercase and strip for case-insensitive comparison."""
    return s.strip().lower()


# ---------------------------------------------------------------------------
# Individual scoring functions
# ---------------------------------------------------------------------------

def risk_score(predicted: str, truth: str) -> float:
    """Score risk level prediction with ordinal partial credit.

    Scoring:
      - Exact match: 1.0
      - Adjacent (off by 1): 0.5
      - Off by 2: 0.15
      - Off by 3 or invalid: 0.0

    Plus a safety multiplier: predicting LOW on CRITICAL gets an extra 0.5x
    penalty (severe underestimation is more dangerous than overestimation).

    Args:
        predicted: Agent's predicted risk level (case-insensitive)
        truth: Ground truth risk level

    Returns:
        Clamped float in (0, 1)
    """
    predicted_upper = (predicted or "").strip().upper()
    truth_upper = (truth or "").strip().upper()

    if predicted_upper not in RISK_LEVELS:
        return _clamp(0.0)
    if truth_upper not in RISK_LEVELS:
        return _clamp(0.0)  # malformed ground truth

    diff = abs(RISK_LEVELS.index(predicted_upper) - RISK_LEVELS.index(truth_upper))
    raw = _RISK_ADJACENCY.get(diff, 0.0)

    # Safety: severe underestimation (predicting LOW when truth is CRITICAL) halves score
    if truth_upper == 'CRITICAL' and predicted_upper == 'LOW':
        raw *= 0.5

    return _clamp(raw)


def jaccard_score(predicted: List[str], truth: List[str]) -> float:
    """Score blast-radius prediction using Jaccard similarity.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|

    Edge cases:
      - Both empty: near-perfect (the agent correctly identified no blast radius)
      - One empty, other not: near-zero

    Args:
        predicted: Agent's list of affected module paths
        truth: Ground truth affected modules

    Returns:
        Clamped Jaccard similarity in (0, 1)
    """
    # Normalize to lowercase sets to allow case-insensitive path matching
    a: Set[str] = {_normalize_str(x) for x in (predicted or []) if x}
    b: Set[str] = {_normalize_str(x) for x in (truth or []) if x}

    if not a and not b:
        return _clamp(1.0)   # Both empty = correct prediction of empty blast radius
    if not a or not b:
        return _clamp(0.0)   # One side empty = miss

    intersection = len(a & b)
    union = len(a | b)
    if union == 0:
        return _clamp(0.5)

    raw = intersection / union
    return _clamp(raw)


def reviewer_score(predicted: str, truth: str) -> float:
    """Score reviewer recommendation.

    Exact match (case-insensitive, whitespace-stripped) → near 1.0.
    Mismatch → near 0.0.

    Future extension: handle reviewer aliases or team membership.

    Args:
        predicted: Agent's recommended reviewer name
        truth: Ground truth reviewer name

    Returns:
        Clamped score
    """
    p = _normalize_str(predicted or "")
    t = _normalize_str(truth or "")

    if not p or not t:
        return _clamp(0.0)

    raw = 1.0 if p == t else 0.0
    return _clamp(raw)


def merge_score(predicted: str, truth: str, risk_level: str) -> float:
    """Score merge decision with a safety rule for critical risk.

    Safety rule: APPROVE on CRITICAL risk applies a -0.5 penalty to the base
    score (encouraging agents to never rubber-stamp critical-risk changes).

    Scoring:
      - Correct decision: 1.0 base
      - Wrong decision: 0.0 base
      - APPROVE on CRITICAL: base - 0.5 (minimum 0.0)

    Args:
        predicted: Agent's merge decision
        truth: Ground truth merge decision
        risk_level: Ground truth risk level (triggers safety rule if CRITICAL)

    Returns:
        Clamped score in (0, 1)
    """
    p = (predicted or "").strip().upper()
    t = (truth or "").strip().upper()
    r = (risk_level or "").strip().upper()

    base = 1.0 if p == t else 0.0

    # Safety penalty: APPROVE on CRITICAL = reckless — penalize regardless of correctness
    if p == 'APPROVE' and r == 'CRITICAL':
        base -= 0.5

    return _clamp(max(0.0, base))


# ---------------------------------------------------------------------------
# Composite reward
# ---------------------------------------------------------------------------

def compute_reward(action, ground_truth: dict, task: str) -> float:
    """Compute composite reward for a completed action.

    Weights (task3 only, must sum to 1.0):
      - Risk classification:  25%
      - Blast radius:         30%
      - Reviewer assignment:  20%
      - Merge decision:       25%

    Args:
        action: Object with risk_level, affected_modules, recommended_reviewer, merge_decision
        ground_truth: Dict with the correct values
        task: 'task1', 'task2', or 'task3'

    Returns:
        Clamped composite score in (0, 1)
    """
    gt = ground_truth or {}

    if task == 'task1':
        score = risk_score(
            getattr(action, 'risk_level', '') or '',
            gt.get('risk_level', '')
        )

    elif task == 'task2':
        score = jaccard_score(
            getattr(action, 'affected_modules', []) or [],
            gt.get('blast_radius', [])
        )

    elif task == 'task3':
        r = 0.0
        r += risk_score(
            getattr(action, 'risk_level', '') or '',
            gt.get('risk_level', '')
        ) * 0.25
        r += jaccard_score(
            getattr(action, 'affected_modules', []) or [],
            gt.get('blast_radius', [])
        ) * 0.30
        r += reviewer_score(
            getattr(action, 'recommended_reviewer', '') or '',
            gt.get('recommended_reviewer', '')
        ) * 0.20
        r += merge_score(
            getattr(action, 'merge_decision', '') or '',
            gt.get('merge_decision', ''),
            gt.get('risk_level', '')
        ) * 0.25
        score = r

    else:
        score = 0.01  # Unknown task — return near-zero but not zero

    return _clamp(score)


# ---------------------------------------------------------------------------
# Human-readable feedback
# ---------------------------------------------------------------------------

def build_feedback(action, ground_truth: dict, task: str) -> str:
    """Build a detailed diagnostic feedback string for an action.

    Args:
        action: Action object with review fields
        ground_truth: Dict with correct values
        task: Task identifier

    Returns:
        Pipe-separated feedback with per-component scores
    """
    gt = ground_truth or {}
    parts = []

    if task in ('task1', 'task3'):
        predicted = getattr(action, 'risk_level', '') or ''
        truth = gt.get('risk_level', '')
        sc = risk_score(predicted, truth)
        correct = predicted.upper() == truth.upper()
        parts.append(
            f"risk_level: predicted={predicted!r} truth={truth!r} "
            f"score={sc:.3f} {'✓' if correct else '✗'}"
        )

    if task in ('task2', 'task3'):
        predicted = getattr(action, 'affected_modules', []) or []
        truth = gt.get('blast_radius', [])
        sc = jaccard_score(predicted, truth)
        parts.append(
            f"blast_radius: predicted={predicted} truth={truth} "
            f"jaccard={sc:.3f}"
        )

    if task == 'task3':
        pred_rev = getattr(action, 'recommended_reviewer', '') or ''
        truth_rev = gt.get('recommended_reviewer', '')
        rev_sc = reviewer_score(pred_rev, truth_rev)
        correct = pred_rev.strip().lower() == truth_rev.strip().lower()
        parts.append(
            f"reviewer: predicted={pred_rev!r} truth={truth_rev!r} "
            f"score={rev_sc:.3f} {'✓' if correct else '✗'}"
        )

        pred_merge = getattr(action, 'merge_decision', '') or ''
        truth_merge = gt.get('merge_decision', '')
        mrg_sc = merge_score(pred_merge, truth_merge, gt.get('risk_level', ''))
        correct = pred_merge.upper() == truth_merge.upper()
        parts.append(
            f"merge: predicted={pred_merge!r} truth={truth_merge!r} "
            f"score={mrg_sc:.3f} {'✓' if correct else '✗'}"
        )

        total = compute_reward(action, gt, task)
        parts.append(f"total_reward={total:.3f}")

    return " | ".join(parts)
