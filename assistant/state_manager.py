"""State manager for assistant."""

from .models import State, MarketContext, SetupResult


class StateManager:
    """Determines the current state based on score and setup."""

    def determine(self, ctx: MarketContext, setup: SetupResult, score: int) -> State:
        if not setup:
            return State.IDLE

        if score >= 70:
            return State.ACTION
        elif score >= 55:
            return State.ALERT
        elif score >= 25:
            return State.WATCH
        else:
            return State.IDLE