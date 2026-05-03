"""
JSON-based quiz results persistence.

Stores quiz answers and scored results in a single JSON file.
Each entry is a self-contained snapshot: the full quiz_answers dict is stored
as-is so the schema flexes automatically when questions are added or changed.
"""
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

_RESULTS_PATH = Path(__file__).parent.parent / "data" / "results.json"


def _read_all() -> list[dict]:
    """Read the results file, returning an empty list if missing or corrupt."""
    if not _RESULTS_PATH.exists():
        return []
    try:
        data = json.loads(_RESULTS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        log.warning("Could not read %s; starting fresh", _RESULTS_PATH)
        return []


def _write_all(results: list[dict]) -> None:
    """Write the full results list back to disk."""
    _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RESULTS_PATH.write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )


def save_result(
    quiz_answers: dict,
    top_cities: list,
    questions_version: int,
    title: str = "",
) -> str:
    """Persist a quiz result and return its ID.

    Args:
        quiz_answers: The raw quiz_answers dict from session state.
        top_cities: The scored/ranked city list from calculate_city_scores().
        questions_version: Number of questions in the quiz at save time.
        title: User-provided label for this result set.

    Returns:
        The UUID string assigned to this result.
    """
    result_id = uuid.uuid4().hex[:12]
    entry = {
        "id": result_id,
        "title": title,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "questions_version": questions_version,
        "quiz_answers": quiz_answers,
        "top_cities": top_cities,
    }

    results = _read_all()
    results.append(entry)
    _write_all(results)
    log.info("Saved result %s (%d cities)", result_id, len(top_cities))
    return result_id


def load_results() -> list[dict]:
    """Return all saved results, newest first."""
    results = _read_all()
    results.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return results


def load_result(result_id: str) -> dict | None:
    """Return a single result by ID, or None if not found."""
    for r in _read_all():
        if r.get("id") == result_id:
            return r
    return None


def delete_result(result_id: str) -> bool:
    """Remove a result by ID. Returns True if found and removed."""
    results = _read_all()
    filtered = [r for r in results if r.get("id") != result_id]
    if len(filtered) == len(results):
        return False
    _write_all(filtered)
    log.info("Deleted result %s", result_id)
    return True
