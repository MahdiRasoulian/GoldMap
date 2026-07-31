"""Notifier – sends alerts via Telegram (directly) or WebSocket."""

import json
import asyncio
import requests
from loguru import logger

from .models import AssistantOutput
from config.loader import CONFIG


class Notifier:
    """Sends assistant output to connected clients and Telegram."""

    def __init__(self, websocket_manager=None):
        self.ws_manager = websocket_manager
        self.telegram_config = CONFIG.get("telegram", {})
        self.enabled = self.telegram_config.get("enabled", False)
        self.bot_token = self.telegram_config.get("bot_token")
        self.chat_id = self.telegram_config.get("chat_id")

        # ── تنظیمات سطح ارسال ──────────────────────────────────────
        # حداقل وضعیت: "action" یا "alert" (پیش‌فرض: "alert")
        self.min_state = self.telegram_config.get("min_state", "alert")
        # حداقل اطمینان (۰ تا ۱۰۰) - اگر تنظیم نشده باشد، ۰ در نظر گرفته می‌شود
        self.min_confidence = self.telegram_config.get("min_confidence", 0)

        if self.enabled and self.bot_token and self.chat_id:
            logger.info(f"Telegram notifier enabled (min_state={self.min_state}, min_confidence={self.min_confidence})")
        else:
            logger.warning("Telegram notifier disabled or misconfigured")

    def _send_telegram_message(self, text: str) -> bool:
        """Send a message to Telegram."""
        if not self.enabled or not self.bot_token or not self.chat_id:
            return False
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }
        try:
            response = requests.post(url, json=payload, timeout=5)
            if response.status_code == 200:
                return True
            else:
                logger.error(f"Telegram send failed: {response.status_code} {response.text}")
                return False
        except Exception as e:
            logger.error(f"Telegram send exception: {e}")
            return False

    async def send(self, output: AssistantOutput):
        """Send alert if state meets the configured minimum level."""
        # ── بررسی وضعیت ─────────────────────────────────────────────
        state_value = output.state.value
        # ترتیب اهمیت: action > alert > watch > idle
        state_priority = {"action": 3, "alert": 2, "watch": 1, "idle": 0}
        min_priority = state_priority.get(self.min_state, 2)  # پیش‌فرض: alert

        if state_priority.get(state_value, 0) < min_priority:
            return  # سطح وضعیت کمتر از حد مجاز است

        # ── بررسی اطمینان ──────────────────────────────────────────
        if output.confidence < self.min_confidence:
            return  # اطمینان کمتر از حد مجاز است

        # Build message
        message = self._format_message(output)

        # Send via WebSocket if available (optional)
        if self.ws_manager:
            data = {
                "type": "assistant_alert",
                "state": output.state.value,
                "setup": output.setup.value if output.setup else None,
                "confidence": output.confidence,
                "price": output.price,
                "entry_zone": f"{output.entry_zone_low:.2f}–{output.entry_zone_high:.2f}",
                "stop_loss": output.stop_loss,
                "target": output.target,
                "reasons": output.reasons,
                "timestamp": output.timestamp.isoformat(),
            }
            await self.ws_manager.broadcast(json.dumps(data))
            logger.info("WebSocket alert broadcasted")

        # Send via Telegram (synchronous, run in thread)
        if self.enabled:
            try:
                await asyncio.to_thread(self._send_telegram_message, message)
                logger.info(f"Telegram alert sent (state={state_value}, confidence={output.confidence}%)")
            except Exception as e:
                logger.error(f"Telegram send async error: {e}")

    def _format_message(self, output: AssistantOutput) -> str:
        """Format assistant output as HTML for Telegram."""
        state_emoji = "🔔" if output.state.value == "alert" else "🚨"
        direction = getattr(output, 'setup_direction', '')
        setup_label = output.setup.value.upper().replace('_', ' ') if output.setup else ''

        lines = []
        lines.append(f"<b>{state_emoji} GOLDMAP ASSISTANT</b>\n")
        lines.append(f"<b>STATE:</b> {output.state.value.upper()}")
        if setup_label:
            lines.append(f"<b>SETUP:</b> {direction.upper()} {setup_label}")
        lines.append(f"<b>CONFIDENCE:</b> {output.confidence}%")
        lines.append(f"<b>PRICE:</b> {output.price:.2f}\n")
        if output.entry_zone_low and output.entry_zone_high:
            lines.append(f"<b>ENTRY ZONE:</b> {output.entry_zone_low:.2f}–{output.entry_zone_high:.2f}")
        if output.stop_loss:
            lines.append(f"<b>STOP LOSS:</b> {output.stop_loss:.2f}")
        if output.target:
            lines.append(f"<b>TARGET:</b> {output.target:.2f}")
        if output.reasons:
            lines.append(f"\n<b>REASONS:</b>")
            for r in output.reasons:
                lines.append(f"✅ {r}")
        lines.append(f"\n⏱ {output.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append("\n#Goldmap #XAUUSD #TradingSignal")

        return "\n".join(lines)