from __future__ import annotations

from enum import Enum


class LiveState(str, Enum):
    OFF = "OFF"
    PAPER = "PAPER"
    SHADOW = "SHADOW"
    CANARY = "CANARY"
    LIVE_LIMITED = "LIVE_LIMITED"


class LiveStateError(ValueError):
    pass


_ALLOWED_TRANSITIONS = {
    LiveState.OFF.value: {LiveState.PAPER.value},
    LiveState.PAPER.value: {LiveState.OFF.value, LiveState.SHADOW.value},
    LiveState.SHADOW.value: {LiveState.OFF.value, LiveState.CANARY.value},
    LiveState.CANARY.value: {LiveState.OFF.value, LiveState.LIVE_LIMITED.value},
    LiveState.LIVE_LIMITED.value: {LiveState.OFF.value},
}


def validate_transition(current: str, requested: str) -> str:
    if current not in _ALLOWED_TRANSITIONS:
        raise LiveStateError(f"Unknown live state: {current}")
    if requested not in _ALLOWED_TRANSITIONS[current]:
        raise LiveStateError(f"Invalid live state transition: {current} -> {requested}")
    return requested


def validate_live_state(state: str) -> str:
    if state not in _ALLOWED_TRANSITIONS:
        raise LiveStateError(f"Unknown live state: {state}")
    return state
