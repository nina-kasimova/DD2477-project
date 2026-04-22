"""
Contextual Bandit (LinUCB) for personalised search re-ranking.

Algorithm
---------
LinUCB (Li et al., 2010) treats each search result as an "arm" of a bandit.
The **context** is a feature vector built from:
  - The document's BM25 score (normalised)
  - The document's semantic score (normalised)
  - The user's personalisation score from history
  - Click-count of the document in user history
  - Whether the document was previously clicked for this query
  - Recency of last click (decayed)

For each arm (document), LinUCB maintains a ridge-regression model that
predicts the expected reward (click probability).  The upper confidence
bound adds exploration, so that uncertain documents get a chance to be
shown higher.

Persistence
-----------
Model parameters (A, b per arm, and a shared A_0, b_0) are stored in a
JSON file so the model survives server restarts.
"""

import json
from typing import Optional
import math
import time
from pathlib import Path

import numpy as np

MODEL_PATH = Path("bandit_model.json")

# ── hyper-parameters ─────────────────────────────────────────────────────────
ALPHA_UCB = 3.0       # exploration factor (higher → more exploration)
D = 6                # feature dimension (see _build_features)
LAMBDA_REG = 1.0       # ridge regression regularisation


class LinUCBModel:
    """
    LinUCB with disjoint linear models — one (A, b) pair per arm_id.

    Arms are identified by Elasticsearch document IDs.
    If an arm has never been seen, it starts with A = λI, b = 0.
    """

    def __init__(self):
        # arm_id → {"A": np.ndarray (d×d), "b": np.ndarray (d,)}
        self._arms: dict[str, dict] = {}
        self._load()

    # ── persistence ──────────────────────────────────────────────────────────
    def _load(self):
        """Load model from disk if it exists."""
        if MODEL_PATH.exists():
            try:
                raw = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
                for arm_id, data in raw.items():
                    self._arms[arm_id] = {
                        "A": np.array(data["A"]),
                        "b": np.array(data["b"]),
                    }
            except Exception:
                self._arms = {}

    def save(self):
        """Persist model to disk."""
        raw = {}
        for arm_id, data in self._arms.items():
            raw[arm_id] = {
                "A": data["A"].tolist(),
                "b": data["b"].tolist(),
            }
        MODEL_PATH.write_text(json.dumps(raw), encoding="utf-8")

    # ── arm management ───────────────────────────────────────────────────────
    def _get_arm(self, arm_id: str) -> dict:
        """Return (or create) model parameters for an arm."""
        if arm_id not in self._arms:
            self._arms[arm_id] = {
                "A": np.eye(D) * LAMBDA_REG,
                "b": np.zeros(D),
            }
        return self._arms[arm_id]

    # ── scoring ──────────────────────────────────────────────────────────────
    def score(self, arm_id: str, x: np.ndarray, alpha: float = ALPHA_UCB) -> tuple[float, float]:
        """
        Compute the UCB score for an arm given context vector x.

        Returns (ucb_score, predicted_reward).
        """
        arm = self._get_arm(arm_id)
        A = arm["A"]
        b = arm["b"]

        A_inv = np.linalg.inv(A)
        theta = A_inv @ b                         # learned weights
        pred = float(x @ theta)                     # predicted reward
        confidence = float(alpha * np.sqrt(x @ A_inv @ x))  # exploration bonus
        ucb = pred + confidence

        return ucb, pred

    # ── update ───────────────────────────────────────────────────────────────
    def update(self, arm_id: str, x: np.ndarray, reward: float):
        """
        Update the model for an arm after observing a reward.

        reward = 1.0 if the user clicked the document, 0.0 otherwise.
        """
        arm = self._get_arm(arm_id)
        arm["A"] = arm["A"] + np.outer(x, x)
        arm["b"] = arm["b"] + reward * x
        self.save()

    # ── stats ────────────────────────────────────────────────────────────────
    @property
    def num_arms(self) -> int:
        return len(self._arms)

    def arm_stats(self, arm_id: str) -> dict:
        """Return diagnostic info for one arm."""
        arm = self._get_arm(arm_id)
        A_inv = np.linalg.inv(arm["A"])
        theta = A_inv @ arm["b"]
        return {
            "arm_id": arm_id,
            "theta": theta.tolist(),
            "n_updates": int(np.trace(arm["A"]) / LAMBDA_REG) - D,
        }


# ── singleton model ──────────────────────────────────────────────────────────
_model = LinUCBModel()


def get_model() -> LinUCBModel:
    return _model


# ── feature engineering ──────────────────────────────────────────────────────
def _build_features(
    bm25_score_norm: float,
    semantic_score_norm: float,
    personal_score_norm: float,
    click_count: int,
    was_clicked_for_query: bool,
    recency_decay: float,
) -> np.ndarray:
    """
    Build a D-dimensional context feature vector for a search result.

    Features (all in [0, 1] range for stable learning):
      0: normalised BM25 score
      1: normalised semantic score
      2: normalised personalisation score from history
      3: log(1 + click_count) / 5  (capped at 1)
      4: 1 if previously clicked for this query, else 0
      5: recency decay factor (1 = just clicked, 0 = never)
    """
    return np.array([
        min(bm25_score_norm, 1.0),
        min(semantic_score_norm, 1.0),
        min(personal_score_norm, 1.0),
        min(math.log1p(click_count) / 5.0, 1.0),
        1.0 if was_clicked_for_query else 0.0,
        min(recency_decay, 1.0),
    ], dtype=np.float64)


def compute_recency_decay(last_clicked_iso: Optional[str], half_life_days: float = 7.0) -> float:
    """Exponential decay based on how recently the doc was clicked."""
    if not last_clicked_iso:
        return 0.0
    try:
        from datetime import datetime, timezone
        # Parse ISO timestamp
        if last_clicked_iso.endswith("Z"):
            last_clicked_iso = last_clicked_iso[:-1] + "+00:00"
        last_clicked = datetime.fromisoformat(last_clicked_iso)
        now = datetime.now(timezone.utc)
        days_ago = (now - last_clicked).total_seconds() / 86400.0
        return math.exp(-math.log(2) * days_ago / half_life_days)
    except Exception:
        return 0.0


def bandit_rerank(
    es,
    query: str,
    results: list[dict],
    history_scores: dict[str, tuple[float, int]],
    click_log_entries: list[dict],
    alpha_ucb: float = ALPHA_UCB,
) -> list[dict]:
    """
    Re-rank search results using LinUCB contextual bandit.

    Parameters
    ----------
    es : Elasticsearch client
    query : the user's search query
    results : list of ES hit dicts (must already have _score)
    history_scores : {doc_id: (history_bm25_score, rank)} from user_history
    click_log_entries : raw click log entries
    alpha_ucb : exploration parameter

    Returns
    -------
    Re-ranked results list with added bandit metadata keys.
    """
    model = get_model()

    if not results:
        return results

    # ── normalise scores ────────────────────────────────────────────────────
    max_score = max((h["_score"] for h in results), default=1.0) or 1.0
    max_hist = max((s for s, _ in history_scores.values()), default=1.0) or 1.0

    # Build a set of (query, doc_id) pairs from past clicks
    past_query_docs = set()
    for entry in click_log_entries:
        past_query_docs.add((entry.get("query", "").lower().strip(),
                             entry.get("doc_id", "")))

    # ── per-doc click counts & recency from history index ───────────────────
    doc_meta: dict[str, dict] = {}
    for entry in click_log_entries:
        did = entry.get("doc_id", "")
        if did not in doc_meta:
            doc_meta[did] = {"click_count": 0, "last_ts": ""}
        doc_meta[did]["click_count"] += 1
        ts = entry.get("timestamp", "")
        if ts > doc_meta[did]["last_ts"]:
            doc_meta[did]["last_ts"] = ts

    # ── score each arm ──────────────────────────────────────────────────────
    scored_results = []
    for hit in results:
        doc_id = hit["_id"]

        # BM25 score (normalised)
        bm25_norm = hit["_score"] / max_score

        # Semantic score — approximate as part of the combined _score
        # (Since we can't easily separate it, we use the overall score)
        semantic_norm = bm25_norm  # they contribute to the same _score

        # History score
        if doc_id in history_scores:
            hist_score, _ = history_scores[doc_id]
            personal_norm = hist_score / max_hist
        else:
            personal_norm = 0.0

        # Click metadata
        meta = doc_meta.get(doc_id, {"click_count": 0, "last_ts": ""})
        click_count = meta["click_count"]
        was_clicked = (query.lower().strip(), doc_id) in past_query_docs
        recency = compute_recency_decay(meta["last_ts"] or None)

        # Build feature vector
        x = _build_features(
            bm25_score_norm=bm25_norm,
            semantic_score_norm=semantic_norm,
            personal_score_norm=personal_norm,
            click_count=click_count,
            was_clicked_for_query=was_clicked,
            recency_decay=recency,
        )

        ucb, pred = model.score(doc_id, x, alpha=alpha_ucb)

        hit["_bandit_ucb"] = round(ucb, 6)
        hit["_bandit_pred"] = round(pred, 6)
        hit["_bandit_features"] = x.tolist()
        hit["_bandit_exploration"] = round(ucb - pred, 6)

        scored_results.append((ucb, hit))

    # ── re-sort by UCB score ────────────────────────────────────────────────
    scored_results.sort(key=lambda pair: pair[0], reverse=True)
    return [hit for _, hit in scored_results]


def record_impressions_and_click(
    query: str,
    shown_results: list[dict],
    clicked_doc_id: Optional[str],
    history_scores: dict[str, tuple[float, int]],
    click_log_entries: list[dict],
):
    """
    After a user clicks (or doesn't click) on a result, update the bandit.

    - The clicked document gets reward = 1.0
    - All other shown documents get reward = 0.0

    This should be called from the click handler.
    """
    model = get_model()

    if not shown_results:
        return

    max_score = max((h.get("_score", 1.0) for h in shown_results), default=1.0) or 1.0
    max_hist = max((s for s, _ in history_scores.values()), default=1.0) or 1.0

    past_query_docs = set()
    for entry in click_log_entries:
        past_query_docs.add((entry.get("query", "").lower().strip(),
                             entry.get("doc_id", "")))

    doc_meta: dict[str, dict] = {}
    for entry in click_log_entries:
        did = entry.get("doc_id", "")
        if did not in doc_meta:
            doc_meta[did] = {"click_count": 0, "last_ts": ""}
        doc_meta[did]["click_count"] += 1
        ts = entry.get("timestamp", "")
        if ts > doc_meta[did]["last_ts"]:
            doc_meta[did]["last_ts"] = ts

    for hit in shown_results:
        doc_id = hit["_id"]
        bm25_norm = hit.get("_score", 0.0) / max_score
        semantic_norm = bm25_norm

        if doc_id in history_scores:
            hist_score, _ = history_scores[doc_id]
            personal_norm = hist_score / max_hist
        else:
            personal_norm = 0.0

        meta = doc_meta.get(doc_id, {"click_count": 0, "last_ts": ""})
        click_count = meta["click_count"]
        was_clicked = (query.lower().strip(), doc_id) in past_query_docs
        recency = compute_recency_decay(meta["last_ts"] or None)

        x = _build_features(
            bm25_score_norm=bm25_norm,
            semantic_score_norm=semantic_norm,
            personal_score_norm=personal_norm,
            click_count=click_count,
            was_clicked_for_query=was_clicked,
            recency_decay=recency,
        )

        reward = 1.0 if doc_id == clicked_doc_id else 0.0
        model.update(doc_id, x, reward)
