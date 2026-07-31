"""Main assistant orchestrator."""

from datetime import datetime, timezone
import asyncio
from loguru import logger

# ── اضافه کردن `State` به imports ────────────────────────────────────
from .models import AssistantOutput, State
from .context_builder import ContextBuilder
from .setup_detector import SetupDetector
from .score_engine import ScoreEngine
from .state_manager import StateManager
from .narrator import Narrator
from .notifier import Notifier


# ── Global notifier instance ────────────────────────────────────────────

_notifier = None


def set_notifier(notifier: Notifier):
    """Set the global notifier instance."""
    global _notifier
    _notifier = notifier


# ── Main function ──────────────────────────────────────────────────────

def run_assistant(force_refresh: bool = False, send_notification: bool = True) -> AssistantOutput:
    """Run the full assistant pipeline and return the output."""
    builder = ContextBuilder()
    ctx = builder.build(force_refresh)

    detector = SetupDetector()
    setup_result = detector.detect(ctx)

    score_engine = ScoreEngine()
    score = score_engine.calculate(ctx, setup_result)

    state_mgr = StateManager()
    state = state_mgr.determine(ctx, setup_result, score)

    output = AssistantOutput(
        state=state,
        setup=setup_result.setup_type if setup_result else None,
        confidence=score,
        price=ctx.price,
        entry_zone_low=setup_result.entry_zone_low if setup_result else 0.0,
        entry_zone_high=setup_result.entry_zone_high if setup_result else 0.0,
        stop_loss=setup_result.stop_loss if setup_result else 0.0,
        target=setup_result.target if setup_result else 0.0,
        reasons=setup_result.reasons if setup_result else [],
        raw_score=score,
        timestamp=datetime.now(timezone.utc)
    )
    if setup_result:
        output.setup_direction = setup_result.direction

    # ── Send notification if requested and state is ALERT/ACTION ──
    # ── اکنون `State` در اینجا تعریف شده است ──────────────────────
    if send_notification and output.state in (State.ACTION, State.ALERT):
        if _notifier:
            try:
                asyncio.create_task(_notifier.send(output))
            except Exception as e:
                logger.error(f"Notification error: {e}")
        else:
            logger.debug("Notifier not set; skipping notification")

    return output


def get_assistant_text(force_refresh: bool = False) -> str:
    """Convenience function to get the narrated text output."""
    output = run_assistant(force_refresh, send_notification=True)
    narrator = Narrator()
    return narrator.narrate(output)