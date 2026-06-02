import logging
import sys
import tempfile
import unittest
from pathlib import Path

from bat import logger as bat_logger
from bat.risk.live_guard import OrderIntent
from bat.security.redaction import RedactingFormatter, SensitiveDataFilter, redact_sensitive


class LogRedactionTest(unittest.TestCase):
    def test_redacts_api_keys_signatures_headers_and_order_intent_amounts(self):
        text = (
            "GET /api?signature=abcdef&api_key=abc123 "
            "apiSecret=supersecret X-MBX-APIKEY: headersecret "
            "OrderIntent(symbol='BTCUSDT', side='BUY', quote_qty=500.0, "
            "quantity=None, confidence=0.81, data_age_seconds=60.0)"
        )

        redacted = redact_sensitive(text)

        for secret in ("abcdef", "abc123", "supersecret", "headersecret", "500.0", "0.81", "60.0"):
            self.assertNotIn(secret, redacted)
        self.assertIn("signature=<redacted>", redacted)
        self.assertIn("api_key=<redacted>", redacted)
        self.assertIn("apiSecret=<redacted>", redacted)
        self.assertIn("X-MBX-APIKEY: <redacted>", redacted)
        self.assertIn("quote_qty=<redacted>", redacted)
        self.assertIn("confidence=<redacted>", redacted)

    def test_redacts_common_secret_carriers(self):
        text = (
            "Authorization: Bearer bearer-token apiKey=api-key "
            "secret_key=sec-value access_token=tok-value "
            '{"api_key": "json-key"}'
        )

        redacted = redact_sensitive(text)

        for secret in ("bearer-token", "api-key", "sec-value", "tok-value", "json-key"):
            self.assertNotIn(secret, redacted)
        self.assertIn("Authorization: Bearer <redacted>", redacted)
        self.assertIn("apiKey=<redacted>", redacted)
        self.assertIn("secret_key=<redacted>", redacted)
        self.assertIn("access_token=<redacted>", redacted)
        self.assertIn('"api_key": "<redacted>"', redacted)

    def test_redacts_dict_and_json_style_secret_carriers(self):
        text = (
            "{'api_key': 'single-json-key'} "
            "{'Authorization': 'Bearer single-bearer'} "
            "{'X-MBX-APIKEY': 'single-header'} "
            '{"X-MBX-APIKEY": "json-header"} '
            "api_secret=api-secret generic secret=plain-secret"
        )

        redacted = redact_sensitive(text)

        for secret in (
            "single-json-key",
            "single-bearer",
            "single-header",
            "json-header",
            "api-secret",
            "plain-secret",
        ):
            self.assertNotIn(secret, redacted)
        self.assertIn("'api_key': '<redacted>'", redacted)
        self.assertIn("'Authorization': 'Bearer <redacted>'", redacted)
        self.assertIn("'X-MBX-APIKEY': '<redacted>'", redacted)
        self.assertIn('"X-MBX-APIKEY": "<redacted>"', redacted)
        self.assertIn("api_secret=<redacted>", redacted)
        self.assertIn("secret=<redacted>", redacted)

    def test_sensitive_data_filter_mutates_message_and_string_args(self):
        record = logging.LogRecord(
            "bat.test",
            logging.INFO,
            __file__,
            1,
            "key=%s sig=%s",
            ("api_key=abc", "signature=abcdef"),
            None,
        )

        self.assertTrue(SensitiveDataFilter().filter(record))
        message = record.getMessage()

        self.assertNotIn("abc", message)
        self.assertNotIn("abcdef", message)
        self.assertIn("api_key=<redacted>", message)
        self.assertIn("signature=<redacted>", message)

    def test_sensitive_data_filter_redacts_order_intent_object_args(self):
        intent = OrderIntent(
            symbol="BTCUSDT",
            side="BUY",
            quote_qty=500.0,
            quantity=None,
            confidence=0.81,
            data_age_seconds=60.0,
        )
        record = logging.LogRecord(
            "bat.test",
            logging.INFO,
            __file__,
            1,
            "intent=%s",
            (intent,),
            None,
        )

        self.assertTrue(SensitiveDataFilter().filter(record))
        message = record.getMessage()

        self.assertNotIn("500.0", message)
        self.assertNotIn("0.81", message)
        self.assertNotIn("60.0", message)
        self.assertIn("quote_qty=<redacted>", message)

    def test_sensitive_data_filter_preserves_non_string_message_object(self):
        message_object = {"event": "order"}
        record = logging.LogRecord(
            "bat.test",
            logging.INFO,
            __file__,
            1,
            message_object,
            (),
            None,
        )

        self.assertTrue(SensitiveDataFilter().filter(record))

        self.assertIs(record.msg, message_object)

    def test_redacting_formatter_redacts_exception_traceback_output(self):
        formatter = RedactingFormatter("%(levelname)s:%(message)s")
        try:
            raise RuntimeError("request failed signature=abcdef Authorization: Bearer token")
        except RuntimeError:
            exc_info = sys.exc_info()
            record = logging.getLogger("bat.test").makeRecord(
                "bat.test",
                logging.ERROR,
                __file__,
                1,
                "failed",
                (),
                exc_info=exc_info,
            )

        formatted = formatter.format(record)

        self.assertNotIn("abcdef", formatted)
        self.assertNotIn("Bearer token", formatted)
        self.assertIn("signature=<redacted>", formatted)
        self.assertIn("Authorization: Bearer <redacted>", formatted)

    def test_crash_report_redacts_exception_traceback_output(self):
        original_log_dir = bat_logger.LOG_DIR
        with tempfile.TemporaryDirectory() as tmpdir:
            bat_logger.LOG_DIR = Path(tmpdir)
            try:
                try:
                    raise RuntimeError("crash signature=abcdef api_key=abc123")
                except RuntimeError:
                    bat_logger.write_crash_report(*sys.exc_info())

                reports = list(Path(tmpdir).glob("crash_*.log"))
                self.assertEqual(len(reports), 1)
                contents = reports[0].read_text(encoding="utf-8")
            finally:
                bat_logger.LOG_DIR = original_log_dir

        self.assertNotIn("abcdef", contents)
        self.assertNotIn("abc123", contents)
        self.assertIn("signature=<redacted>", contents)
        self.assertIn("api_key=<redacted>", contents)


if __name__ == "__main__":
    unittest.main()
