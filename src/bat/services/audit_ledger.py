from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


class AuditLedger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._previous_hash = self._last_record_hash()

    def append(self, event_type, payload) -> dict:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "payload": payload,
            "previous_hash": self._previous_hash,
        }
        record["record_hash"] = self._hash_record(record)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        self._previous_hash = record["record_hash"]
        return record

    def _last_record_hash(self) -> str:
        if not self.path.exists():
            return ""
        last_line = ""
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    last_line = line
        if not last_line:
            return ""
        return str(json.loads(last_line).get("record_hash", ""))

    @staticmethod
    def _hash_record(record: dict) -> str:
        unsigned = dict(record)
        unsigned.pop("record_hash", None)
        canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
