#!/usr/bin/env python3
"""Small timing/logging utilities shared across modules."""

from __future__ import annotations

from functools import wraps
import logging
import time


def timed_info_decorator(func):
    """Decorate one function with a fixed elapsed-time INFO log line."""
    logger = logging.getLogger(func.__module__)

    @wraps(func)
    def wrapped(*args, **kwargs):
        logger.debug("%s...", func.__name__)
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        logger.info("%s complete in %.2fs", func.__name__, float(time.perf_counter() - t0))
        return result

    return wrapped
