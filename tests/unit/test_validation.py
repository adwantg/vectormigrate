from __future__ import annotations

import pytest

from vectormigrate.errors import ValidationError
from vectormigrate.validation import (
    validate_alias_name,
    validate_benchmark_iterations,
    validate_vector_field_name,
)


def test_validate_alias_name_accepts_valid_value() -> None:
    assert validate_alias_name("retrieval_active") == "retrieval_active"


def test_validate_alias_name_rejects_invalid_value() -> None:
    with pytest.raises(ValidationError):
        validate_alias_name(" bad alias ")


def test_validate_vector_field_name_rejects_invalid_identifier() -> None:
    with pytest.raises(ValidationError):
        validate_vector_field_name("1bad-field")


def test_validate_benchmark_iterations_rejects_non_positive() -> None:
    with pytest.raises(ValidationError):
        validate_benchmark_iterations(0)
