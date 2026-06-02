from __future__ import annotations

import logging
import re


_QUERY_SECRET_RE = re.compile(
    r"\b(api_key|apiKey|apiSecret|api_secret|signature|secret_key|access_token|secret|X-MBX-APIKEY)=([^\s&]+)",
    re.IGNORECASE,
)
_HEADER_SECRET_RE = re.compile(r"(X-MBX-APIKEY\s*:\s*)([^\s,;]+)", re.IGNORECASE)
_AUTH_BEARER_RE = re.compile(r"(Authorization\s*:\s*Bearer\s+)([^\s,;]+)", re.IGNORECASE)
_STRUCTURED_AUTH_RE = re.compile(
    r"([\"']Authorization[\"']\s*:\s*[\"']Bearer\s+)([^\"']+)([\"'])",
    re.IGNORECASE,
)
_STRUCTURED_SECRET_RE = re.compile(
    r"([\"'](?:api_key|apiKey|apiSecret|api_secret|signature|secret_key|access_token|secret|X-MBX-APIKEY)[\"']\s*:\s*[\"'])([^\"']+)([\"'])",
    re.IGNORECASE,
)
_ORDER_FIELD_RE = re.compile(
    r"\b(quote_qty|quantity|confidence|data_age_seconds)=([^,)]+)"
)


def redact_sensitive(value) -> str:
    text = str(value)
    text = _QUERY_SECRET_RE.sub(lambda match: f"{match.group(1)}=<redacted>", text)
    text = _HEADER_SECRET_RE.sub(lambda match: f"{match.group(1)}<redacted>", text)
    text = _AUTH_BEARER_RE.sub(lambda match: f"{match.group(1)}<redacted>", text)
    text = _STRUCTURED_AUTH_RE.sub(
        lambda match: f"{match.group(1)}<redacted>{match.group(3)}", text
    )
    text = _STRUCTURED_SECRET_RE.sub(
        lambda match: f"{match.group(1)}<redacted>{match.group(3)}", text
    )
    if "OrderIntent(" in text:
        text = _ORDER_FIELD_RE.sub(lambda match: f"{match.group(1)}=<redacted>", text)
    return text


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact_sensitive(super().format(record))


class SensitiveDataFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_sensitive(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(_redact_arg(arg) for arg in record.args)
        elif isinstance(record.args, dict):
            record.args = {key: _redact_arg(arg) for key, arg in record.args.items()}
        return True


def _redact_arg(arg):
    if isinstance(arg, str):
        return redact_sensitive(arg)
    text = str(arg)
    redacted = redact_sensitive(text)
    return redacted if redacted != text else arg
