"""Small fallback validator for the live-schema subset used by this project.

The normal runtime uses ``jsonschema`` when installed.  This fallback keeps the
experiment self-contained in minimal Python environments and intentionally
supports only the keywords emitted by :mod:`sar_alloc.schemas`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


@dataclass(slots=True)
class ValidationError:
    message: str
    absolute_path: tuple[Any, ...] = ()
    context: list["ValidationError"] = field(default_factory=list)


def _type_ok(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def _errors(value: Any, schema: Mapping[str, Any], path: tuple[Any, ...]) -> list[ValidationError]:
    if "oneOf" in schema:
        branch_results = [_errors(value, branch, path) for branch in schema.get("oneOf", [])]
        valid = [errors for errors in branch_results if not errors]
        if len(valid) == 1:
            return []
        contexts = [error for errors in branch_results for error in errors]
        return [
            ValidationError(
                f"value matches {len(valid)} oneOf branches; exactly one is required",
                path,
                contexts,
            )
        ]
    expected = schema.get("type")
    if isinstance(expected, str) and not _type_ok(value, expected):
        return [ValidationError(f"{value!r} is not of type {expected!r}", path)]
    if "const" in schema and value != schema["const"]:
        return [ValidationError(f"{value!r} was expected to equal {schema['const']!r}", path)]
    if "enum" in schema and value not in list(schema.get("enum", []) or []):
        return [ValidationError(f"{value!r} is not one of {schema['enum']!r}", path)]
    out: list[ValidationError] = []
    if isinstance(value, Mapping):
        required = [str(v) for v in schema.get("required", []) or []]
        for name in required:
            if name not in value:
                out.append(ValidationError(f"{name!r} is a required property", path))
        properties = dict(schema.get("properties", {}) or {})
        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in properties:
                    out.append(ValidationError(f"additional property {name!r} is not allowed", path + (name,)))
        for name, child in properties.items():
            if name in value:
                out.extend(_errors(value[name], child, path + (name,)))
    if isinstance(value, list):
        minimum = int(schema.get("minItems", 0) or 0)
        maximum = schema.get("maxItems")
        if len(value) < minimum:
            out.append(ValidationError(f"array is too short (minimum {minimum})", path))
        if maximum is not None and len(value) > int(maximum):
            out.append(ValidationError(f"array is too long (maximum {maximum})", path))
        if schema.get("uniqueItems"):
            for index, item in enumerate(value):
                if item in value[:index]:
                    out.append(ValidationError("array items are not unique", path + (index,)))
                    break
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                out.extend(_errors(item, item_schema, path + (index,)))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            out.append(ValidationError(f"{value!r} is less than minimum {schema['minimum']!r}", path))
        if "maximum" in schema and value > schema["maximum"]:
            out.append(ValidationError(f"{value!r} is greater than maximum {schema['maximum']!r}", path))
    return out


class MinimalValidator:
    def __init__(self, schema: Mapping[str, Any]) -> None:
        self.schema = dict(schema)

    @classmethod
    def check_schema(cls, schema: Mapping[str, Any]) -> None:
        if not isinstance(schema, Mapping):
            raise TypeError("schema must be a mapping")

    def iter_errors(self, value: Any) -> Iterable[ValidationError]:
        return iter(_errors(value, self.schema, ()))


def validator_for(schema: Mapping[str, Any]):
    del schema
    return MinimalValidator


__all__ = ["MinimalValidator", "ValidationError", "validator_for"]
