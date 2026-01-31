"""Telegram notifications for trading alerts.

This module provides a TelegramAlerter class for sending alerts via Telegram Bot API.

Usage:
    from polymarket_trader.notifications import TelegramAlerter

    alerter = TelegramAlerter.from_env()
    if alerter:
        alerter.send_alert("Hello from polymarket-trader!")

Environment variables required:
    TELEGRAM_BOT_TOKEN: Bot token from @BotFather
    TELEGRAM_CHAT_ID: Chat/channel ID to send messages to
"""

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime

import httpx
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Telegram Bot API endpoint
TELEGRAM_API_BASE = "https://api.telegram.org"


@dataclass
class TelegramConfig:
    """Configuration for Telegram alerts.

    Attributes:
        enabled: Whether Telegram alerts are enabled.
        edge_threshold: Minimum total edge ($) to trigger alert.
        cooldown_minutes: Minimum time between alerts.
        include_trades: Whether to include trade details in message.
        max_trades: Maximum number of trades to show in message.

    """

    enabled: bool = False
    edge_threshold: float = 50.0
    cooldown_minutes: float = 5.0
    include_trades: bool = True
    max_trades: int = 5

    @classmethod
    def from_dict(cls, data: dict) -> "TelegramConfig":
        """Create config from dictionary (YAML section).

        Args:
            data: Dictionary with telegram config values.

        Returns:
            TelegramConfig instance.

        """
        return cls(
            enabled=data.get("enabled", False),
            edge_threshold=data.get("edge_threshold", 50.0),
            cooldown_minutes=data.get("cooldown_minutes", 5.0),
            include_trades=data.get("include_trades", True),
            max_trades=data.get("max_trades", 5),
        )


class TelegramAlerter:
    """Send alerts via Telegram Bot API.

    Handles:
    - Loading credentials from environment
    - Rate limiting with cooldown
    - Graceful failure (doesn't crash if Telegram fails)

    Example:
        alerter = TelegramAlerter.from_env()
        if alerter and alerter.should_alert(edge=75.0, threshold=50.0):
            alerter.send_alert("High edge detected!")

    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        timeout: float = 10.0,
    ) -> None:
        """Initialize the Telegram alerter.

        Args:
            bot_token: Telegram bot token from @BotFather.
            chat_id: Chat or channel ID to send messages to.
            timeout: HTTP request timeout in seconds.

        """
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._timeout = timeout
        self._last_alert_time: float = 0.0

        logger.info(
            "TelegramAlerter initialized: chat_id=%s",
            chat_id[:4] + "..." if len(chat_id) > 4 else chat_id,
        )

    @classmethod
    def from_env(cls, env_file: str | None = None) -> "TelegramAlerter | None":
        """Create alerter from environment variables.

        Args:
            env_file: Optional path to .env file.

        Returns:
            TelegramAlerter if credentials found, None otherwise.

        """
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()

        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")

        if not bot_token or not chat_id:
            logger.warning(
                "Telegram credentials not found in environment. "
                "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to enable alerts."
            )
            return None

        return cls(bot_token=bot_token, chat_id=chat_id)

    def should_alert(
        self,
        edge: float,
        threshold: float,
        cooldown_minutes: float,
    ) -> bool:
        """Check if an alert should be sent.

        Args:
            edge: Current total edge value.
            threshold: Minimum edge to trigger alert.
            cooldown_minutes: Minimum time between alerts.

        Returns:
            True if alert should be sent.

        """
        # Check edge threshold
        if edge < threshold:
            return False

        # Check cooldown
        now = time.time()
        cooldown_seconds = cooldown_minutes * 60
        if now - self._last_alert_time < cooldown_seconds:
            logger.debug(
                "Alert suppressed by cooldown (%.1f min remaining)",
                (cooldown_seconds - (now - self._last_alert_time)) / 60,
            )
            return False

        return True

    def send_alert(self, message: str, parse_mode: str = "HTML") -> bool:
        """Send an alert message via Telegram.

        Args:
            message: Message text to send.
            parse_mode: Telegram parse mode ("HTML" or "Markdown").

        Returns:
            True if message sent successfully, False otherwise.

        """
        url = f"{TELEGRAM_API_BASE}/bot{self._bot_token}/sendMessage"

        payload = {
            "chat_id": self._chat_id,
            "text": message,
            "parse_mode": parse_mode,
        }

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(url, json=payload)

            if response.status_code == 200:
                self._last_alert_time = time.time()
                logger.info("Telegram alert sent successfully")
                return True
            else:
                logger.error(
                    "Telegram API error: %d - %s",
                    response.status_code,
                    response.text[:200],
                )
                return False

        except httpx.TimeoutException:
            logger.error("Telegram request timed out after %.1fs", self._timeout)
            return False
        except httpx.RequestError as e:
            logger.error("Telegram request failed: %s", e)
            return False
        except Exception as e:
            logger.error("Unexpected error sending Telegram alert: %s", e, exc_info=True)
            return False

    def format_edge_alert(
        self,
        event: str,
        total_edge: float,
        budget: float,
        trades: list[dict],
        delta: float,
        theta: float,
        include_trades: bool = True,
        max_trades: int = 5,
    ) -> str:
        """Format an edge alert message.

        Args:
            event: Event name/slug.
            total_edge: Total expected edge in dollars.
            budget: Budget used.
            trades: List of trade dicts with keys: contract, action, qty, price, edge.
            delta: Portfolio delta exposure.
            theta: Portfolio theta exposure.
            include_trades: Whether to include trade details.
            max_trades: Maximum trades to show.

        Returns:
            Formatted HTML message.

        """
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M")

        lines = [
            f"🚨 <b>EDGE ALERT: {event}</b>",
            "",
            f"💰 Expected Edge: <b>${total_edge:.2f}</b>",
            f"📊 Budget: ${budget:.2f}",
        ]

        if include_trades and trades:
            lines.append("")
            lines.append("📈 <b>Top Trades:</b>")

            # Sort by absolute edge, take top N
            sorted_trades = sorted(trades, key=lambda t: abs(t["edge"]), reverse=True)
            for trade in sorted_trades[:max_trades]:
                edge_str = f"${trade['edge']:+.2f}"
                lines.append(
                    f"• {trade['contract']} {trade['action']} "
                    f"{trade['qty']:.0f} @ {trade['price']:.0%} → {edge_str}"
                )

        lines.extend(
            [
                "",
                f"⚖️ Portfolio: Δ={delta:+.2f} Θ={theta:+.2f}",
                f"🕐 {timestamp}",
            ]
        )

        return "\n".join(lines)
