"""Canonical per-case status strings shared by tasks, runner, and reporting."""

from __future__ import annotations


class Status:
    PASSED = "PASSED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    API_ERROR = "API_ERROR"
    API_ERROR_FATAL = "API_ERROR_FATAL"
    INTERNAL_ERROR = "INTERNAL_ERROR"


# Statuses that count toward the pass-rate denominator; API and internal
# errors are reported separately instead of being graded as failures.
GRADED_STATUSES = (Status.PASSED, Status.FAILED, Status.TIMEOUT)
