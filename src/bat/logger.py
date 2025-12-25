import gzip
import logging
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "bat.log"
ERROR_FILE = LOG_DIR / "bat.error.log"
DEBUG_FILE = LOG_DIR / "bat.debug.log"
DEFAULT_MAX_MB = 100


class DailySizeRotatingFileHandler(TimedRotatingFileHandler):
    def __init__(self, filename, max_bytes=0, **kwargs):
        super().__init__(filename, **kwargs)
        self.maxBytes = max_bytes
        self._part_index = 0

    def shouldRollover(self, record):
        if self.stream is None:
            self.stream = self._open()

        current_time = int(time.time())
        if current_time >= self.rolloverAt:
            return 1

        if self.maxBytes > 0:
            msg = f"{self.format(record)}\n"
            self.stream.seek(0, os.SEEK_END)
            if self.stream.tell() + len(msg.encode("utf-8")) >= self.maxBytes:
                return 1

        return 0

    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None

        current_time = int(time.time())
        if current_time >= self.rolloverAt:
            date_str = time.strftime("%Y-%m-%d", time.localtime(self.rolloverAt - self.interval))
            self._part_index = 0
        else:
            date_str = time.strftime("%Y-%m-%d", time.localtime(current_time))
            self._part_index += 1

        rotated = f"{self.baseFilename}.{date_str}.{self._part_index}"
        if os.path.exists(self.baseFilename):
            os.rename(self.baseFilename, rotated)
            gz_path = f"{rotated}.gz"
            with open(rotated, "rb") as src, gzip.open(gz_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            os.remove(rotated)

        if self.backupCount > 0:
            self._cleanup_old()

        if current_time >= self.rolloverAt:
            new_rollover = self.rolloverAt + self.interval
            while new_rollover <= current_time:
                new_rollover += self.interval
            self.rolloverAt = new_rollover

        self.stream = self._open()

    def _cleanup_old(self):
        log_dir = os.path.dirname(self.baseFilename) or "."
        prefix = os.path.basename(self.baseFilename) + "."
        files = [
            os.path.join(log_dir, name)
            for name in os.listdir(log_dir)
            if name.startswith(prefix) and name.endswith(".gz")
        ]
        files.sort(key=lambda p: os.path.getmtime(p))
        excess = len(files) - self.backupCount
        for path in files[:excess]:
            os.remove(path)


def get_logger(name: str = "bat") -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    max_mb = int(os.getenv("BAT_LOG_MAX_MB", DEFAULT_MAX_MB))
    max_bytes = max_mb * 1024 * 1024
    info_handler = DailySizeRotatingFileHandler(
        LOG_FILE,
        max_bytes=max_bytes,
        when="midnight",
        interval=1,
        backupCount=14,
        encoding="utf-8",
    )
    error_handler = DailySizeRotatingFileHandler(
        ERROR_FILE,
        max_bytes=max_bytes,
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    debug_handler = DailySizeRotatingFileHandler(
        DEBUG_FILE,
        max_bytes=max_bytes,
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8",
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    info_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)
    debug_handler.setFormatter(formatter)

    info_handler.setLevel(logging.INFO)
    error_handler.setLevel(logging.ERROR)
    debug_handler.setLevel(logging.DEBUG)

    class InfoFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return record.levelno < logging.ERROR

    info_handler.addFilter(InfoFilter())

    logger.addHandler(info_handler)
    logger.addHandler(error_handler)
    logger.addHandler(debug_handler)
    return logger


def log_to_file(level: str, message: str) -> None:
    logger = get_logger()
    level = level.upper()
    if level == "ERROR":
        logger.error(message)
    elif level == "WARNING":
        logger.warning(message)
    else:
        logger.info(message)


def write_crash_report(exc_type, exc, tb) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    crash_file = LOG_DIR / f"crash_{timestamp}.log"
    with crash_file.open("w", encoding="utf-8") as handle:
        handle.write("=== BAT Crash Report ===\n")
        handle.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
        handle.write("".join(traceback.format_exception(exc_type, exc, tb)))


def install_crash_handler() -> None:
    def _handle_exception(exc_type, exc, tb):
        logger = get_logger("bat.crash")
        logger.exception("Unhandled exception", exc_info=(exc_type, exc, tb))
        write_crash_report(exc_type, exc, tb)
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _handle_exception
