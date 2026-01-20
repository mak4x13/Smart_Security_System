import csv
import json
import os
import re
import difflib
from datetime import datetime, timedelta

from groq import Groq

from backend.config import PERSONS_CSV, ATTENDANCE_CSV

DATE_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M")

RANGE_LABELS = {
    "last_hour": "in the last hour",
    "last_24_hours": "in the last 24 hours",
    "today": "today",
    "last_week": "in the last week",
    "all_time": "overall",
}


def _parse_timestamp(value: str):
    if not value:
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _load_csv(path: str):
    if not os.path.exists(path):
        return [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return reader.fieldnames or [], rows


def _normalize(text: str):
    if not text:
        return ""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _normalize_key(text: str):
    if not text:
        return ""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _has_pronoun(question: str):
    return bool(
        re.search(
            r"\b(his|her|their|them|him|that person|this person|that user|this user|he|she)\b",
            question.lower(),
        )
    )


def _has_status_word(question: str):
    tokens = re.findall(r"[a-zA-Z]+", question.lower())
    for token in tokens:
        if difflib.SequenceMatcher(None, token, "status").ratio() >= 0.8:
            return True
    return False


def _detect_range_key(question: str):
    q = question.lower()
    if "last hour" in q or "last hr" in q or "past hour" in q:
        return "last_hour"
    if "last 24" in q or "past 24" in q or "last day" in q:
        return "last_24_hours"
    if "today" in q:
        return "today"
    if "last week" in q or "past week" in q:
        return "last_week"
    if "overall" in q or "all time" in q or "total" in q:
        return "all_time"
    return "all_time"


def _range_label(range_key: str):
    return RANGE_LABELS.get(range_key, "overall")


def _in_range(ts: datetime, range_key: str, now: datetime):
    if ts is None:
        return False
    if range_key == "last_hour":
        return ts >= now - timedelta(hours=1)
    if range_key == "last_24_hours":
        return ts >= now - timedelta(hours=24)
    if range_key == "today":
        return ts.date() == now.date()
    if range_key == "last_week":
        return ts >= now - timedelta(days=7)
    return True


def _extract_name_candidate(question: str):
    if not question:
        return ""
    patterns = [
        r"(?:named|name is|person|user)\s+([a-zA-Z0-9][a-zA-Z0-9\s'-]{1,60})",
        r"(?:status of|status for)\s+([a-zA-Z0-9][a-zA-Z0-9\s'-]{1,60})",
        r"([a-zA-Z0-9][a-zA-Z0-9\s'-]{1,60})\s+status",
        r"is\s+([a-zA-Z0-9][a-zA-Z0-9\s'-]{1,60})\s+(?:present|enrolled|registered|in|here)",
    ]
    match = None
    for pattern in patterns:
        match = re.search(pattern, question, re.I)
        if match:
            break
    if not match:
        return ""
    candidate = match.group(1).strip()
    candidate = re.sub(
        r"\b(present|enrolled|registered|in|here|data|database|status)\b.*",
        "",
        candidate,
        flags=re.I,
    )
    return candidate.strip().strip("?")


def _score_match(candidate_key: str, target_key: str):
    if not candidate_key or not target_key:
        return 0.0
    if candidate_key == target_key:
        return 1.0
    if candidate_key in target_key or target_key in candidate_key:
        return 0.92
    return difflib.SequenceMatcher(None, candidate_key, target_key).ratio()


def _match_persons(question: str, persons, memory):
    normalized_question = _normalize_key(question)
    candidate_raw = _extract_name_candidate(question)
    candidate_key = _normalize_key(candidate_raw)

    if not candidate_key and memory and _has_pronoun(question):
        candidate_raw = memory.get("last_person_name") or memory.get("last_person_id") or ""
        candidate_key = _normalize_key(candidate_raw)

    matches = []
    for person in persons:
        name = person.get("display_name", "")
        pid = person.get("person_id", "")
        name_key = _normalize_key(name)
        pid_key = _normalize_key(pid)

        if candidate_key:
            score = max(_score_match(candidate_key, name_key), _score_match(candidate_key, pid_key))
            if score >= 0.78:
                matches.append((score, person))
                continue
        else:
            if name_key and name_key in normalized_question:
                matches.append((0.85, person))
            elif pid_key and pid_key in normalized_question:
                matches.append((0.85, person))

    matches.sort(key=lambda item: item[0], reverse=True)
    return candidate_raw, [p for _, p in matches]


def _detect_intent(question: str, name_candidate: str, has_match: bool):
    q = question.lower()
    if _has_status_word(question):
        if name_candidate or has_match or _has_pronoun(question):
            return "status_lookup"
    if "date" in q and "today" in q:
        return "get_date"
    if name_candidate or has_match:
        if "in your data" in q or "in the data" in q or "in database" in q:
            return "person_lookup"
        if "checked in" in q or "check in" in q or "attendance" in q or "present" in q:
            return "attendance_lookup"
        return "person_lookup"
    if "attendance" in q or "checked in" in q or "check in" in q or "present" in q:
        return "attendance_count"
    if ("how many" in q or "count" in q) and ("enroll" in q or "register" in q):
        return "enrollment_count"
    if "enroll" in q or "registered" in q or "registration" in q:
        return "list_persons"
    if ("how many" in q or "count" in q) and ("person" in q or "people" in q):
        return "person_count"
    if "who" in q or "list" in q or "show" in q:
        if "attendance" in q or "checked in" in q or "check in" in q:
            return "list_attendance"
        return "list_persons"
    return "general"


def _summarize_attendance(rows, range_key: str, now: datetime):
    filtered = []
    last_seen = {}
    for row in rows:
        ts = _parse_timestamp(row.get("timestamp", ""))
        if ts is None:
            continue
        pid = row.get("person_id", "")
        if pid:
            if pid not in last_seen or ts > last_seen[pid]:
                last_seen[pid] = ts
        if _in_range(ts, range_key, now):
            filtered.append(row)

    unique_people = {row.get("person_id") for row in filtered if row.get("person_id")}
    return filtered, len(unique_people), last_seen


def _summarize_enrollments(header, persons, range_key: str, now: datetime):
    if not header:
        return {"available": False, "count": 0, "missing": 0}
    if "enrolled_at" not in header:
        return {"available": False, "count": None, "missing": len(persons)}

    count = 0
    missing = 0
    for row in persons:
        enrolled_at = row.get("enrolled_at", "")
        ts = _parse_timestamp(enrolled_at)
        if not enrolled_at or ts is None:
            missing += 1
            continue
        if _in_range(ts, range_key, now):
            count += 1
    return {"available": True, "count": count, "missing": missing}


def _person_attendance_details(person, attendance_rows, range_key: str, now: datetime):
    pid = person.get("person_id", "")
    logs_in_range = []
    last_seen = None
    for row in attendance_rows:
        if row.get("person_id") != pid:
            continue
        ts = _parse_timestamp(row.get("timestamp", ""))
        if ts is None:
            continue
        if last_seen is None or ts > last_seen:
            last_seen = ts
        if _in_range(ts, range_key, now):
            logs_in_range.append(ts)
    return {
        "person_id": pid,
        "display_name": person.get("display_name", ""),
        "check_ins_in_range": len(logs_in_range),
        "last_seen": last_seen.strftime("%Y-%m-%d %H:%M:%S") if last_seen else None,
    }


def _build_context(question: str, memory):
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    today_str = now.strftime("%Y-%m-%d")

    persons_header, persons = _load_csv(PERSONS_CSV)
    _, attendance = _load_csv(ATTENDANCE_CSV)

    range_key = _detect_range_key(question)
    range_label = _range_label(range_key)

    name_candidate, matches = _match_persons(question, persons, memory or {})

    context = {
        "now": now_str,
        "today": today_str,
        "range": {"key": range_key, "label": range_label},
        "name_candidate": name_candidate,
        "memory": {
            "last_person_id": (memory or {}).get("last_person_id", ""),
            "last_person_name": (memory or {}).get("last_person_name", ""),
        },
        "persons": {
            "total": len(persons),
            "rows": persons,
            "fuzzy_matches": [
                {
                    "person_id": p.get("person_id", ""),
                    "display_name": p.get("display_name", ""),
                    "role": p.get("role", ""),
                    "department": p.get("department", ""),
                    "access_status": p.get("access_status", ""),
                    "enrolled_at": p.get("enrolled_at", ""),
                }
                for p in matches[:6]
            ],
            "has_enrollment_timestamps": "enrolled_at" in (persons_header or []),
        },
        "attendance": {
            "rows": attendance,
        },
    }
    return context


def _public_context(context):
    return {k: v for k, v in context.items() if not k.startswith("_")}


def _ask_groq(question: str, context):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None, "missing_api_key"

    client = Groq(api_key=api_key)
    system_prompt = (
        "You are a helpful security assistant. Answer using ONLY the provided context data. "
        "Respond naturally like a GPT chat (friendly, direct, and concise). "
        "If the user greets you, greet back and ask how you can help. "
        "The context includes persons.rows (enrollment records) and attendance.rows (check-in logs). "
        "Interpret 'present' or 'enrolled' as being in persons.csv unless the question mentions "
        "attendance or 'checked in'. "
        "For status questions, use access_status from persons.csv. "
        "For time ranges (last hour, today, last week), compare timestamps to the provided 'now'. "
        "If enrollment timestamps are missing and a time-based enrollment question is asked, "
        "say you only have total enrolled. "
        "If the question uses pronouns (his/her/their), resolve them using memory.last_person_name "
        "or memory.last_person_id when available. "
        "If asked about names starting with a letter, count using persons.rows display_name. "
        "Prefer display_name; if missing, use person_id. "
        "Use fuzzy_matches to resolve typos if available. "
        "Do not mention JSON, internal keys, or raw data structures."
    )
    user_prompt = (
        "Question:\n"
        f"{question}\n\n"
        "Context (JSON):\n"
        f"{json.dumps(_public_context(context), ensure_ascii=True)}"
    )

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    text = completion.choices[0].message.content.strip()
    return text, None


def _label_for_person_id(person_id: str, persons_lookup):
    if not person_id:
        return "Unknown"
    display_name = persons_lookup.get(person_id, "")
    if display_name and display_name != person_id:
        return f"{display_name} ({person_id})"
    return display_name or person_id


def _fallback_answer(context):
    intent = context.get("intent", "general")
    range_label = context.get("range", {}).get("label", "overall")
    today = context.get("today")
    name_candidate = context.get("name_candidate") or "that person"
    persons = context.get("persons", {})
    attendance = context.get("attendance", {})
    persons_lookup = context.get("_persons_lookup", {})

    if intent == "get_date":
        return f"Today is {today}."

    if intent == "person_lookup":
        matches = persons.get("matches", [])
        if not matches:
            return f"No, I could not find {name_candidate} in the enrollment data."
        names = ", ".join([_label_for_person_id(m.get("person_id", ""), persons_lookup) for m in matches])
        return f"Yes, {name_candidate} is enrolled. Match: {names}."

    if intent == "status_lookup":
        matches = persons.get("matches", [])
        if not matches:
            return f"I could not find {name_candidate} in the enrollment data."
        person = matches[0]
        status = person.get("access_status") or "unknown"
        display = person.get("display_name") or person.get("person_id") or name_candidate
        return f"{display} is {status}."

    if intent == "attendance_lookup":
        matches = attendance.get("matches", [])
        if not matches:
            return f"I could not find attendance logs for {name_candidate} {range_label}."
        detail = matches[0]
        count = detail.get("check_ins_in_range", 0)
        last_seen = detail.get("last_seen")
        if count > 0:
            return (
                f"{detail.get('display_name') or name_candidate} checked in {range_label} "
                f"{count} time(s). Last seen: {last_seen or 'unknown'}."
            )
        return (
            f"No check-ins for {detail.get('display_name') or name_candidate} {range_label}. "
            f"Last seen: {last_seen or 'unknown'}."
        )

    if intent == "attendance_count":
        unique_people = attendance.get("unique_people_in_range", 0)
        total_logs = attendance.get("total_logs_in_range", 0)
        return f"{unique_people} unique people checked in {range_label}. Total logs: {total_logs}."

    if intent == "enrollment_count":
        if not persons.get("enrollment_timestamps_available"):
            total = persons.get("total", 0)
            return (
                "Enrollment timestamps are missing for older records. "
                f"Total enrolled persons: {total}."
            )
        count = persons.get("enrollments_in_range", 0)
        missing = persons.get("missing_enrolled_at", 0)
        if missing:
            return (
                f"{count} people enrolled {range_label}. "
                f"Note: {missing} records are missing enrollment timestamps."
            )
        return f"{count} people enrolled {range_label}."

    if intent == "person_count":
        total = persons.get("total", 0)
        return f"There are {total} enrolled persons overall."

    if intent == "list_persons":
        sample = persons.get("sample", [])
        if not sample:
            return "No enrolled persons found yet."
        names = ", ".join([p.get("display_name") or p.get("person_id") for p in sample])
        return f"Enrolled persons ({persons.get('total', 0)}): {names}."

    if intent == "list_attendance":
        unique_ids = attendance.get("unique_ids_in_range", [])
        if not unique_ids:
            return f"No attendance logs {range_label}."
        labels = [_label_for_person_id(pid, persons_lookup) for pid in unique_ids]
        if len(labels) <= 8:
            return f"Checked in {range_label}: {', '.join(labels)}."
        preview = ", ".join(labels[:8])
        remaining = len(labels) - 8
        return f"Checked in {range_label}: {preview}, and {remaining} other(s)."

    return (
        "I can answer questions about enrollments and attendance. "
        "Try: 'How many people enrolled today?' or 'How many checked in last week?'"
    )


def answer_chat(message: str, memory=None):
    context = _build_context(message, memory or {})
    meta = {}

    matches = context.get("persons", {}).get("fuzzy_matches", [])
    if matches:
        meta["matched_person_id"] = matches[0].get("person_id")
        meta["matched_display_name"] = matches[0].get("display_name")

    try:
        answer, error = _ask_groq(message, context)
    except Exception:
        answer, error = None, "groq_error"

    if error == "missing_api_key":
        meta["error"] = "missing_api_key"
        answer = "GROQ_API_KEY is missing. Please set it to use the chatbot."
    elif error:
        answer = "I am having trouble reaching the model right now. Please try again."

    return {"answer": answer, "meta": meta}
