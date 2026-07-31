"""Narrator – formats assistant output as clean text."""

from datetime import datetime
from .models import AssistantOutput, State, SetupType


class Narrator:
    """Generates the final text output."""

    def narrate(self, output: AssistantOutput) -> str:
        lines = []
        lines.append("━━━━━━━━━━━━━━━━━━━━")
        lines.append("")
        lines.append(f"STATE: {output.state.value.upper()}")
        lines.append("")
        
        # توضیح برای وضعیت IDLE
        if output.state == State.IDLE and not output.setup:
            lines.append("📌 No active setup detected.")
            lines.append("   Price is between major liquidity zones.")
            lines.append("   Waiting for price to approach a key level.")
            lines.append("")
        
        if output.setup:
            lines.append(f"SETUP: {self._format_setup(output.setup, output)}")
            lines.append("")
            lines.append(f"CONFIDENCE: {output.confidence} %")
            lines.append("")
        
        lines.append(f"PRICE: {output.price:.2f}")
        lines.append("")
        
        if output.entry_zone_low and output.entry_zone_high:
            lines.append(f"ENTRY ZONE:")
            lines.append(f"{output.entry_zone_low:.2f}–{output.entry_zone_high:.2f}")
            lines.append("")
        if output.stop_loss:
            lines.append(f"STOP LOSS:")
            lines.append(f"{output.stop_loss:.2f}")
            lines.append("")
        if output.target:
            lines.append(f"TARGET:")
            lines.append(f"{output.target:.2f}")
            lines.append("")
        if output.reasons:
            lines.append("REASONS:")
            for r in output.reasons:
                lines.append(f"✓ {r}")
            lines.append("")
        
        lines.append("━━━━━━━━━━━━━━━━━━━━")
        lines.append("")
        lines.append(f"⏱ {output.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        return "\n".join(lines)

    def _format_setup(self, setup: SetupType, output: AssistantOutput) -> str:
        direction = output.setup_direction if hasattr(output, 'setup_direction') else ""
        if setup == SetupType.LIQUIDITY_REVERSAL:
            return f"{direction.upper()} REVERSAL"
        elif setup == SetupType.STOP_HUNT_REVERSAL:
            return f"{direction.upper()} REVERSAL"
        elif setup == SetupType.BREAKOUT_CONTINUATION:
            return f"{direction.upper()} CONTINUATION"
        elif setup == SetupType.ABSORPTION_REVERSAL:
            return f"{direction.upper()} REVERSAL"
        elif setup == SetupType.EXHAUSTION:
            return f"{direction.upper()} EXHAUSTION"
        return setup.value.upper()