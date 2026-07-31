"""Detect trading setups from market context."""

from typing import Optional, List
import numpy as np
from .models import MarketContext, SetupResult, SetupType


class SetupDetector:
    """Detects the five allowed setups."""

    def detect(self, ctx: MarketContext) -> Optional[SetupResult]:
        """Run all detectors and return the first valid setup."""
        detectors = [
            self._detect_liquidity_reversal,
            self._detect_stop_hunt_reversal,
            self._detect_breakout_continuation,
            self._detect_absorption_reversal,
            self._detect_exhaustion,
        ]
        for detector in detectors:
            result = detector(ctx)
            if result:
                return result
        return None

    # ---- Individual detectors ----

    def _detect_liquidity_reversal(self, ctx: MarketContext) -> Optional[SetupResult]:
        """Price near strong liquidity zone + reversal signal nearby."""
        if not ctx.liquidity_zones:
            return None

        # Find zones within 10 pips (0.10 price units for XAUUSD)
        price = ctx.price
        near_zones = []
        for zone in ctx.liquidity_zones:
            low = zone.get("price_low", 0)
            high = zone.get("price_high", 0)
            strength = zone.get("strength", 0)
            if strength < 0.5:
                continue
            distance = min(abs(price - low), abs(price - high))
            if distance <= 0.10:   # 10 pips
                near_zones.append((zone, distance))

        if not near_zones:
            return None

        # Check for reversal signals: absorption or stop hunt near same level
        has_absorption = False
        has_stop_hunt = False
        for signal in ctx.absorption_signals:
            if signal.get("confidence", 0) >= 0.6:
                if abs(signal.get("price_level", 0) - price) <= 0.10:
                    has_absorption = True
        for signal in ctx.stop_hunt_signals:
            if signal.get("confidence", 0) >= 0.6:
                if abs(signal.get("extreme_price", 0) - price) <= 0.10:
                    has_stop_hunt = True

        if not (has_absorption or has_stop_hunt):
            return None

        # Determine direction: if price is below zone, reversal up (buy); else sell
        zone_low = near_zones[0][0]["price_low"]
        zone_high = near_zones[0][0]["price_high"]
        if price < zone_low:
            direction = "buy"
            entry_zone_low = zone_low
            entry_zone_high = zone_high
            stop_loss = zone_low - ctx.atr * 0.5
            target = price + ctx.atr * 1.5
        else:
            direction = "sell"
            entry_zone_low = zone_low
            entry_zone_high = zone_high
            stop_loss = zone_high + ctx.atr * 0.5
            target = price - ctx.atr * 1.5

        reasons = ["Liquidity zone nearby", "Reversal signal detected"]
        return SetupResult(
            setup_type=SetupType.LIQUIDITY_REVERSAL,
            direction=direction,
            entry_zone_low=entry_zone_low,
            entry_zone_high=entry_zone_high,
            stop_loss=stop_loss,
            target=target,
            reasons=reasons
        )

    def _detect_stop_hunt_reversal(self, ctx: MarketContext) -> Optional[SetupResult]:
        """Recent stop hunt with reversal confirmed by price action."""
        if not ctx.stop_hunt_signals:
            return None

        # مرتب‌سازی بر اساس زمان (جدیدترین اول)
        sorted_signals = sorted(
            ctx.stop_hunt_signals,
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )
        # فقط سیگنال‌هایی با اطمینان بالا
        high_conf_signals = [s for s in sorted_signals if s.get("confidence", 0) >= 0.7]
        if not high_conf_signals:
            return None

        latest = high_conf_signals[0]  # جدیدترین سیگنال با اطمینان بالا
        extreme = latest.get("extreme_price", 0)
        trigger = latest.get("trigger_price", 0)
        direction = latest.get("hunt_direction", "above")
        price = ctx.price
        atr = max(ctx.atr, 0.20)  # حداقل ATR را ۰.۲۰ در نظر بگیر

        # فروش (شکار استاپ بالا)
        if direction == "above":
            # قیمت باید حداقل ۰.۰۵ پیپ زیر trigger برگشته باشد
            if price >= trigger - 0.05:
                return None

            # اگر قیمت خیلی دور از trigger افتاده باشد (بیش از ۰.۵۰ پیپ)، فرصت از دست رفته
            if trigger - price > 0.50:
                return None

            # منطقه ورود: نزدیک قیمت فعلی، اما نه بالاتر از trigger
            entry_low = price
            entry_high = min(price + 0.10, trigger)
            if entry_high <= entry_low:
                entry_high = entry_low + 0.05

            # حد ضرر: بالای extreme یا trigger + ATR*0.5
            stop_loss = max(extreme, trigger + atr * 0.5) + 0.02

            # هدف: حداقل ۱.۵ ATR پایین‌تر از قیمت ورود
            target = price - max(atr * 1.5, 1.0)
            if target > price - 0.50:
                return None  # هدف خیلی نزدیک است

            reasons = ["Stop hunt detected", "Price reversed back"]
            return SetupResult(
                setup_type=SetupType.STOP_HUNT_REVERSAL,
                direction="sell",
                entry_zone_low=round(entry_low, 2),
                entry_zone_high=round(entry_high, 2),
                stop_loss=round(stop_loss, 2),
                target=round(target, 2),
                reasons=reasons
            )

        # خرید (شکار استاپ پایین)
        elif direction == "below":
            if price <= trigger + 0.05:
                return None

            if price - trigger > 0.50:
                return None

            entry_low = max(price - 0.10, trigger)
            entry_high = price
            if entry_high <= entry_low:
                entry_high = entry_low + 0.05

            stop_loss = min(extreme, trigger - atr * 0.5) - 0.02
            target = price + max(atr * 1.5, 1.0)
            if target < price + 0.50:
                return None

            reasons = ["Stop hunt detected", "Price reversed back"]
            return SetupResult(
                setup_type=SetupType.STOP_HUNT_REVERSAL,
                direction="buy",
                entry_zone_low=round(entry_low, 2),
                entry_zone_high=round(entry_high, 2),
                stop_loss=round(stop_loss, 2),
                target=round(target, 2),
                reasons=reasons
            )

        return None

    def _detect_breakout_continuation(self, ctx: MarketContext) -> Optional[SetupResult]:
        """Breakout with strong volume and no fake signal."""
        if not ctx.fake_breakout_signals:
            return None

        # Look for fake breakout with low confidence -> might be real breakout
        fake_break = ctx.fake_breakout_signals[-1] if ctx.fake_breakout_signals else {}
        if fake_break.get("confidence", 0) >= 0.5:
            return None  # fake breakout is real -> no continuation

        # Check if price has broken a recent high/low
        # Use volume profile to see if volume is increasing
        if ctx.tick_volume > ctx.avg_volume * 1.2:
            # Determine direction
            if ctx.trend == "bullish":
                direction = "buy"
                entry_zone_low = ctx.price
                entry_zone_high = ctx.price + 0.05
                stop_loss = ctx.price - ctx.atr * 0.8
                target = ctx.price + ctx.atr * 1.5
            elif ctx.trend == "bearish":
                direction = "sell"
                entry_zone_low = ctx.price - 0.05
                entry_zone_high = ctx.price
                stop_loss = ctx.price + ctx.atr * 0.8
                target = ctx.price - ctx.atr * 1.5
            else:
                return None

            return SetupResult(
                setup_type=SetupType.BREAKOUT_CONTINUATION,
                direction=direction,
                entry_zone_low=entry_zone_low,
                entry_zone_high=entry_zone_high,
                stop_loss=stop_loss,
                target=target,
                reasons=["Breakout with volume", "Trend aligned"]
            )
        return None

    def _detect_absorption_reversal(self, ctx: MarketContext) -> Optional[SetupResult]:
        """Absorption signal with high confidence and price stalling."""
        if not ctx.absorption_signals:
            return None

        latest = ctx.absorption_signals[-1]
        if latest.get("confidence", 0) < 0.6:
            return None

        direction = latest.get("direction", "neutral")
        price_level = latest.get("price_level", 0)

        # Check if price is still near that level
        if abs(ctx.price - price_level) > 0.10:
            return None

        # Reversal opposite to absorption direction
        if direction == "bullish":
            entry_zone_low = price_level - 0.05
            entry_zone_high = price_level
            stop_loss = price_level + ctx.atr * 0.5
            target = ctx.price - ctx.atr * 1.0
            return SetupResult(
                setup_type=SetupType.ABSORPTION_REVERSAL,
                direction="sell",
                entry_zone_low=entry_zone_low,
                entry_zone_high=entry_zone_high,
                stop_loss=stop_loss,
                target=target,
                reasons=["Absorption detected", "Price stalling"]
            )
        elif direction == "bearish":
            entry_zone_low = price_level
            entry_zone_high = price_level + 0.05
            stop_loss = price_level - ctx.atr * 0.5
            target = ctx.price + ctx.atr * 1.0
            return SetupResult(
                setup_type=SetupType.ABSORPTION_REVERSAL,
                direction="buy",
                entry_zone_low=entry_zone_low,
                entry_zone_high=entry_zone_high,
                stop_loss=stop_loss,
                target=target,
                reasons=["Absorption detected", "Price stalling"]
            )
        return None

    def _detect_exhaustion(self, ctx: MarketContext) -> Optional[SetupResult]:
        """Extreme move with declining volume."""
        # Need at least 20 bars for comparison
        # Since we only have current volume and avg, we use them as proxy
        # A spike in volume followed by drop can indicate exhaustion
        # Here we use relative volume as proxy
        if ctx.tick_volume > ctx.avg_volume * 2.0:
            # Volume spike, check if price has moved significantly
            # Use ATR to judge move
            if ctx.atr > 0:
                move = abs(ctx.price - ctx.price)  # no history, so we use session range
                # In real impl, we need price change over last few bars.
                # As a simplification, we assume if volume is high but price not moving much, exhaustion.
                # For now, return None to avoid false positives.
                pass
        return None