#!/usr/bin/env python3
"""survey_to_csv.py — flatten run_demo's responses.jsonl into one CSV row per session.

    python survey_to_csv.py ~/scratch/survey_responses/responses.jsonl > sessions.csv
    python survey_to_csv.py a.jsonl b.jsonl c.jsonl > sessions.csv   # merge machines

Join with the Google Form results ("B - Cozmo" CSV export) on the form's
"Participant ID" column:
    form row Participant ID == pin_q1  → that row rates THIS session's System 1
                                          (real model = system1_model column)
    form row Participant ID == pin_q2  → rates System 2 (= system2_model)
"""
from __future__ import annotations

import csv
import json
import sys
import time

COLUMNS = [
    "session_id", "started_at", "mode", "completed",
    "order", "system1_model", "system2_model",
    "pin_q1", "pin_q2",
    "consent_name", "agree_system1", "agree_system2",
    "sys1_connected", "sys1_timer_s", "sys1_talk_s", "sys1_n_conversations", "sys1_n_blocks",
    "sys2_connected", "sys2_timer_s", "sys2_talk_s", "sys2_n_conversations", "sys2_n_blocks",
    "q1_opened_form", "q2_opened_form",
    "debrief_name", "gift_choice", "gift_email",
    "ua",
]


def load_sessions(paths: list[str]) -> dict[str, dict]:
    sessions: dict[str, dict] = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = rec.get("session_id")
                if not sid or sid == "preview":
                    continue
                kind = rec.get("kind")
                if kind == "session_start":
                    order = rec.get("order") or ["?", "?"]
                    sessions[sid] = {
                        "session_id": sid,
                        "started_at": time.strftime("%Y-%m-%d %H:%M:%S",
                                                    time.localtime(rec.get("ts", 0))),
                        "mode": "online",
                        "completed": False,
                        "order": ",".join(order),
                        "system1_model": order[0],
                        "system2_model": order[1] if len(order) > 1 else "?",
                        "pin_q1": rec.get("pin_q1", ""),
                        "pin_q2": rec.get("pin_q2", ""),
                        "ua": rec.get("ua", ""),
                    }
                    continue
                row = sessions.get(sid)
                if row is None:
                    continue  # checkpoint without a session_start (foreign file order?)
                if kind == "consent":
                    row["consent_name"] = rec.get("name", "")
                    row["agree_system1"] = rec.get("agree_system1", "")
                    row["agree_system2"] = rec.get("agree_system2", "")
                elif kind == "interact":
                    n = rec.get("which")
                    if rec.get("inperson"):
                        row["mode"] = "inperson"
                    if n in (1, 2):
                        p = f"sys{n}_"
                        row[p + "connected"] = rec.get("connected", "")
                        row[p + "timer_s"] = rec.get("timer_s", "")
                        row[p + "talk_s"] = rec.get("talk_s", "")
                        row[p + "n_conversations"] = rec.get("n_conversations", "")
                        row[p + "n_blocks"] = rec.get("n_blocks", "")
                elif kind == "questionnaire":
                    n = rec.get("which")
                    if n in (1, 2):
                        row[f"q{n}_opened_form"] = rec.get("opened_form", "")
                elif kind == "debrief":
                    row["debrief_name"] = rec.get("name", "")
                elif kind == "gift":
                    row["gift_choice"] = rec.get("gift_choice", rec.get("wants_gift", ""))
                    row["gift_email"] = rec.get("email", "")
                    row["completed"] = True  # gift is the last step
    return sessions


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    sessions = load_sessions(paths)
    writer = csv.DictWriter(sys.stdout, fieldnames=COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in sorted(sessions.values(), key=lambda r: r["started_at"]):
        writer.writerow({c: row.get(c, "") for c in COLUMNS})
    print(f"[survey_to_csv] {len(sessions)} session(s)", file=sys.stderr)


if __name__ == "__main__":
    main()
