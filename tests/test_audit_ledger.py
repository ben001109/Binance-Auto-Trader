import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from bat.services.audit_ledger import AuditLedger


class AuditLedgerTest(unittest.TestCase):
    def test_append_writes_hash_chained_jsonl_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            ledger = AuditLedger(path)

            first = ledger.append("order_blocked", {"symbol": "BTCUSDT", "reason": "trading_disabled"})
            second = ledger.append("order_sent", {"symbol": "BTCUSDT", "order_id": "123"})

            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(rows), 2)
        self.assertEqual(first["previous_hash"], "")
        self.assertEqual(second["previous_hash"], first["record_hash"])
        for row in rows:
            without_hash = dict(row)
            record_hash = without_hash.pop("record_hash")
            canonical = json.dumps(without_hash, sort_keys=True, separators=(",", ":"))
            self.assertEqual(record_hash, hashlib.sha256(canonical.encode("utf-8")).hexdigest())

    def test_existing_file_last_record_hash_is_used_as_previous_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            first = AuditLedger(path).append("order_blocked", {"symbol": "BTCUSDT"})

            next_record = AuditLedger(path).append("order_sent", {"symbol": "BTCUSDT"})

        self.assertEqual(next_record["previous_hash"], first["record_hash"])


if __name__ == "__main__":
    unittest.main()
