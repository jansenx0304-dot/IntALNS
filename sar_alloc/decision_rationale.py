"""Structural validation and trace resolution for compact decision rationale.

Each Agent action exposes one ``rationale`` object with ``reason`` and observation
refs. Rationale is explanatory context; it never changes runtime control.
Only structure and ref resolvability are hard-validated. Grounding breadth and
rationale quality are prompt guidance rather than hidden program constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping

@dataclass(frozen=True, slots=True)
class RationaleBlockView:
    block_id: str
    block_type: str
    summary: str
    signals: Dict[str, Any]


@dataclass(frozen=True, slots=True)
class RationaleContext:
    blocks: tuple[RationaleBlockView, ...]

    def block_ids(self) -> list[str]:
        return [block.block_id for block in self.blocks]

    def signal_ids(self, block_id: str) -> list[str]:
        for block in self.blocks:
            if block.block_id == block_id:
                return list(block.signals)
        return []

    def valid_rationale_refs(self) -> list[str]:
        return [
            _make_signal_ref(block.block_id, str(signal_name))
            for block in self.blocks
            for signal_name in block.signals
        ]


def build_rationale_context(observation: Mapping[str, Any]) -> RationaleContext:
    blocks: list[RationaleBlockView] = []
    for raw in observation.get("observation_blocks", []) or []:
        if not isinstance(raw, Mapping):
            continue
        block_id = str(raw.get("block_id", "") or "").strip()
        if not block_id:
            continue
        signals_raw = raw.get("signals", {}) or {}
        blocks.append(
            RationaleBlockView(
                block_id=block_id,
                block_type=str(raw.get("block_type", "") or ""),
                summary=str(raw.get("summary", "") or ""),
                signals=dict(signals_raw) if isinstance(signals_raw, Mapping) else {},
            )
        )
    return RationaleContext(tuple(blocks))






def valid_rationale_refs(observation: Mapping[str, Any]) -> list[str]:
    return build_rationale_context(observation).valid_rationale_refs()


def validate_rationale(
    rationale: Any,
    observation: Mapping[str, Any],
    *,
    field: str,
) -> None:
    if not isinstance(rationale, Mapping):
        raise ValueError(f"{field} must be an object")
    allowed_fields = {"reason", "refs"}
    extra = sorted(str(key) for key in rationale if str(key) not in allowed_fields)
    if extra:
        raise ValueError(f"{field} contains unknown fields: {extra}")
    reason = rationale.get("reason")
    if not isinstance(reason, str):
        raise ValueError(f"{field}.reason must be a string")
    validate_rationale_refs(rationale.get("refs"), observation, field=f"{field}.refs")


def validate_rationale_refs(refs: Any, observation: Mapping[str, Any], *, field: str) -> None:
    if not isinstance(refs, list):
        raise ValueError(f"{field} must be an array")
    normalized: list[str] = []
    for index, ref in enumerate(refs):
        if not isinstance(ref, str) or not ref.strip():
            raise ValueError(f"{field}[{index}] must be a non-empty string")
        value = ref.strip()
        if value != ref:
            raise ValueError(f"{field}[{index}] must not contain surrounding whitespace")
        if any(token in value for token in (":", "=", "<", ">")):
            raise ValueError(
                f"{field}[{index}] must be a pure observation path, not a value or explanation"
            )
        normalized.append(value)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field} contains duplicate refs")
    for index, ref in enumerate(normalized):
        try:
            resolve_rationale_ref(observation, ref)
        except ValueError as exc:
            raise ValueError(f"{field}[{index}] is invalid: {exc}") from exc


def resolve_decision_rationale(
    observation: Mapping[str, Any], decision: Mapping[str, Any]
) -> list[Dict[str, Any]]:
    rationale = decision.get("rationale")
    if not isinstance(rationale, Mapping):
        return []
    return [{
        "decision_block": "action",
        "reason": str(rationale.get("reason", "") or ""),
        "facts": resolve_rationale_refs(observation, rationale.get("refs", []) or []),
    }]


def resolve_rationale_ref(observation: Mapping[str, Any], ref: str) -> Dict[str, Any]:
    parts = str(ref).split(".")
    if len(parts) != 2:
        raise ValueError("expected format '<block_id>.<signal_name>'")
    block = _find_observation_block(observation, parts[0])
    if block is None:
        raise ValueError(f"observation block not found: {parts[0]}")
    signal_name = parts[1]
    if signal_name not in block.signals:
        raise ValueError(
            f"signal {signal_name!r} is not in block {block.block_id}; "
            f"allowed signals: {sorted(str(name) for name in block.signals)}"
        )
    return {
        "ref": ref,
        "block_id": block.block_id,
        "block_type": block.block_type,
        "signal": signal_name,
        "value": block.signals.get(signal_name),
    }


def resolve_rationale_refs(observation: Mapping[str, Any], refs: Iterable[Any]) -> list[Dict[str, Any]]:
    resolved: list[Dict[str, Any]] = []
    for ref in refs or []:
        if not isinstance(ref, str):
            continue
        try:
            resolved.append(resolve_rationale_ref(observation, ref))
        except ValueError:
            continue
    return resolved


def _find_observation_block(
    observation: Mapping[str, Any], block_id: str
) -> RationaleBlockView | None:
    for block in build_rationale_context(observation).blocks:
        if block.block_id == block_id:
            return block
    return None


def _make_signal_ref(block_id: str, signal_name: str) -> str:
    return f"{block_id}.{signal_name}"


__all__ = [
    "RationaleBlockView",
    "RationaleContext",
    "build_rationale_context",
    "resolve_decision_rationale",
    "resolve_rationale_ref",
    "resolve_rationale_refs",
    "valid_rationale_refs",
    "validate_rationale",
    "validate_rationale_refs",
]
