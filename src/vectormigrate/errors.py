from __future__ import annotations


class VectorMigrateError(Exception):
    """Base exception for vectormigrate."""


class ValidationError(VectorMigrateError):
    """Raised when a user-supplied configuration is invalid."""


class CLIUsageError(VectorMigrateError):
    """Raised for CLI usage failures that should produce a structured error."""
