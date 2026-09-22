"""Replayable agent turn trace records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List



@dataclass(slots=True)
class AgentTurnRecord:
    turn_id: str
    role: str
    phase: str
    action: str | None
    agent_input: Dict[str, Any]
    agent_output: Dict[str, Any]
    validation: Dict[str, Any]
    result: Dict[str, Any] | None
    error: str | None = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "role": self.role,
            "phase": self.phase,
            "action": self.action,
            "agent_input": dict(self.agent_input or {}),
            "agent_output": dict(self.agent_output or {}),
            "validation": dict(self.validation or {}),
            "result": None if self.result is None else dict(self.result),
            "error": self.error,
        }


@dataclass(slots=True)
class RunTrace:
    turns: List[AgentTurnRecord] = field(default_factory=list)

    def add(self, record: AgentTurnRecord) -> None:
        self.turns.append(record)

    def as_artifact(self) -> Dict[str, Any]:
        return {"turns": [turn.as_dict() for turn in self.turns]}
