"""Calculate confidence score (0–100) based on detected setup and market conditions."""

from .models import MarketContext, SetupResult


class ScoreEngine:
    """Computes a score from 0 to 100."""

    def calculate(self, ctx: MarketContext, setup: SetupResult) -> int:
        if not setup:
            return 0

        score = 0

        # 1. Trend alignment (max 15)
        if ctx.trend == "bullish" and setup.direction == "buy":
            score += 15
        elif ctx.trend == "bearish" and setup.direction == "sell":
            score += 15

        # 2. Stop hunt detected (max 25) - اهمیت بیشتر
        if ctx.stop_hunt_signals:
            high_conf = any(s.get("confidence", 0) >= 0.7 for s in ctx.stop_hunt_signals)
            score += 25 if high_conf else 10

        # 3. Absorption detected (max 15)
        if ctx.absorption_signals:
            high_conf = any(s.get("confidence", 0) >= 0.6 for s in ctx.absorption_signals)
            score += 15 if high_conf else 8

        # 4. Fake breakout (max 15)
        if ctx.fake_breakout_signals:
            low_conf = any(s.get("confidence", 0) < 0.4 for s in ctx.fake_breakout_signals)
            score += 15 if low_conf else 8

        # 5. Volume spike (max 30) - اهمیت بیشتر
        if ctx.tick_volume > ctx.avg_volume * 2.0:
            score += 30
        elif ctx.tick_volume > ctx.avg_volume * 1.5:
            score += 20
        elif ctx.tick_volume > ctx.avg_volume * 1.2:
            score += 10

        return min(100, score)