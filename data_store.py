import json
from datetime import datetime, timezone
from pathlib import Path

from config import CLICK_LOG, SEARCH_LOG, USER_PROFILES


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []

    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def load_profiles() -> dict:
    if not USER_PROFILES.exists():
        return {}

    with USER_PROFILES.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_profiles(profiles: dict) -> None:
    with USER_PROFILES.open("w", encoding="utf-8") as handle:
        json.dump(profiles, handle, indent=2)


def all_users() -> list[dict]:
    profiles = load_profiles()
    users = []
    for user_id in sorted(profiles):
        users.append(
            {
                "user_id": user_id,
                "bio": profiles[user_id].get("bio", ""),
            }
        )
    return users


def current_user(session_obj) -> dict | None:
    user_id = session_obj.get("user_id")
    if not user_id:
        return None

    profiles = load_profiles()
    profile = profiles.get(user_id, {})
    return {
        "user_id": user_id,
        "bio": profile.get("bio", ""),
    }


def set_user_profile(session_obj, user_id: str, bio: str) -> None:
    if user_id:
        session_obj["user_id"] = user_id
        profiles = load_profiles()
        existing_bio = profiles.get(user_id, {}).get("bio", "")
        profiles[user_id] = {"bio": bio if bio else existing_bio}
        save_profiles(profiles)
    else:
        session_obj.pop("user_id", None)


def update_bio(user_id: str, bio: str) -> None:
    profiles = load_profiles()
    profiles[user_id] = {"bio": bio}
    save_profiles(profiles)


def log_search(user_id: str, query_text: str, mode: str) -> None:
    append_jsonl(
        SEARCH_LOG,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
            "query": query_text,
            "mode": mode,
        },
    )


def log_click(user_id: str | None, doc_id: str, query_text: str, rank: int | None, mode: str) -> None:
    append_jsonl(
        CLICK_LOG,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
            "doc_id": doc_id,
            "query": query_text,
            "rank": rank,
            "mode": mode,
        },
    )


def user_activity(user_id: str) -> tuple[list[dict], list[dict]]:
    searches = [row for row in read_jsonl(SEARCH_LOG) if row.get("user_id") == user_id]
    clicks = [row for row in read_jsonl(CLICK_LOG) if row.get("user_id") == user_id]
    searches.reverse()
    clicks.reverse()
    return searches, clicks


def clear_clicks_for_user(user_id: str) -> None:
    rows = [row for row in read_jsonl(CLICK_LOG) if row.get("user_id") != user_id]
    write_jsonl(CLICK_LOG, rows)
