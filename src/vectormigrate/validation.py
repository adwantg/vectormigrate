from __future__ import annotations

import re

from vectormigrate.errors import ValidationError

_ALIAS_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,127}$")


def validate_alias_name(alias_name: str) -> str:
    if not _ALIAS_PATTERN.fullmatch(alias_name):
        raise ValidationError(
            "alias_name must start with an alphanumeric character and contain only "
            "letters, digits, '.', '_', or '-'"
        )
    return alias_name


def validate_vector_field_name(field_name: str) -> str:
    if not field_name or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", field_name):
        raise ValidationError("vector field name must be a valid identifier-like token")
    return field_name


def validate_benchmark_iterations(iterations: int) -> int:
    if iterations <= 0:
        raise ValidationError("iterations must be positive")
    return iterations
