from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from bat.research_config import RiskConfig
from bat.risk.live_state import LiveState


@dataclass(frozen=True)
class GuardDecision:
    allowed: bool
    state: str
    reason: str
    adjusted_quote_qty: float | None
    adjusted_quantity: float | None = None


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: str
    quote_qty: float | None
    quantity: float | None
    confidence: float
    data_age_seconds: float


@dataclass(frozen=True)
class AccountRiskSnapshot:
    equity_usdt: float
    daily_pnl_fraction: float
    consecutive_losses: int


class LiveGuard:
    def __init__(
        self,
        risk_config: RiskConfig,
        *,
        allowed_symbol: str = "BTCUSDT",
        min_confidence: float = 0.70,
        max_data_age_seconds: float = 1200.0,
    ):
        self.risk_config = risk_config
        self.allowed_symbol = allowed_symbol
        self.min_confidence = min_confidence
        self.max_data_age_seconds = max_data_age_seconds

    def evaluate_order_request(
        self,
        intent: OrderIntent,
        account: AccountRiskSnapshot,
    ) -> GuardDecision:
        state = self.risk_config.live_state

        if state != LiveState.CANARY.value:
            return self._block(state, "state_not_live")
        if self.risk_config.paper_only:
            return self._block(state, "paper_only")
        if not self.risk_config.trading_enabled:
            return self._block(state, "trading_disabled")
        if intent.symbol != self.allowed_symbol:
            return self._block(state, "symbol_not_allowed")
        if intent.side not in {"BUY", "SELL"}:
            return self._block(state, "invalid_order_side")
        if not isfinite(intent.confidence):
            return self._block(state, "invalid_confidence")
        if not isfinite(intent.data_age_seconds):
            return self._block(state, "invalid_data_age")
        if intent.quote_qty is not None and not isfinite(intent.quote_qty):
            return self._block(state, "invalid_order_size")
        if intent.quantity is not None and not isfinite(intent.quantity):
            return self._block(state, "invalid_order_size")
        if not isfinite(account.equity_usdt):
            return self._block(state, "invalid_equity")
        if not isfinite(account.daily_pnl_fraction):
            return self._block(state, "invalid_daily_pnl")
        if not isfinite(account.consecutive_losses) or account.consecutive_losses < 0:
            return self._block(state, "invalid_consecutive_losses")
        if intent.confidence < self.min_confidence:
            return self._block(state, "confidence_below_threshold")
        if account.equity_usdt <= 0:
            return self._block(state, "invalid_equity")
        if account.daily_pnl_fraction <= -self.risk_config.max_daily_loss_fraction:
            return self._block(state, "daily_loss_limit")
        if account.consecutive_losses >= self.risk_config.max_consecutive_losses:
            return self._block(state, "consecutive_loss_limit")
        if intent.data_age_seconds > self.max_data_age_seconds:
            return self._block(state, "stale_data")

        adjusted_quote_qty = self._adjusted_quote_qty(intent, account)
        if adjusted_quote_qty is None and intent.side == "BUY":
            return self._block(state, "invalid_order_size")
        if adjusted_quote_qty is not None and adjusted_quote_qty <= 0:
            return self._block(state, "invalid_order_size")
        if intent.side == "SELL" and (intent.quantity is None or intent.quantity <= 0):
            return self._block(state, "invalid_order_size")

        adjusted_quantity = intent.quantity if intent.side == "SELL" else None
        return GuardDecision(True, state, "allowed", adjusted_quote_qty, adjusted_quantity)

    def _adjusted_quote_qty(
        self,
        intent: OrderIntent,
        account: AccountRiskSnapshot,
    ) -> float | None:
        if intent.quote_qty is None:
            return None
        max_quote_qty = account.equity_usdt * self.risk_config.max_order_equity_fraction
        return min(intent.quote_qty, max_quote_qty)

    @staticmethod
    def _block(state: str, reason: str) -> GuardDecision:
        return GuardDecision(False, state, reason, None)
