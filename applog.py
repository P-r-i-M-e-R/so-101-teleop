"""Central error tracing: everything goes to logs/so101.log (rotating) and the console, including
uncaught exceptions in any thread, with full tracebacks."""
import logging
import logging.handlers
import sys
import threading
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / 'logs'
LOG_FILE = LOG_DIR / 'so101.log'
log = logging.getLogger('so101')


def setup():
    if log.handlers:
        return
    LOG_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter('%(asctime)s %(levelname)-7s [%(threadName)s] %(message)s', '%Y-%m-%d %H:%M:%S')
    fh = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=5, encoding='utf-8')
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.WARNING)
    log.setLevel(logging.DEBUG)
    log.addHandler(fh)
    log.addHandler(sh)
    log.propagate = False
    sys.excepthook = lambda t, v, tb: log.critical('UNCAUGHT in main thread', exc_info=(t, v, tb))
    threading.excepthook = lambda a: log.critical(
        'UNCAUGHT in thread %s', a.thread.name if a.thread else '?', exc_info=(a.exc_type, a.exc_value, a.exc_traceback))
    log.info('=== so101 recorder started ===')


def tail(n=200):
    try:
        return ''.join(LOG_FILE.read_text(encoding='utf-8', errors='replace').splitlines(True)[-n:])
    except OSError:
        return ''
