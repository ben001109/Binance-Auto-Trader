from __future__ import annotations

from dataclasses import dataclass
import hashlib

from bat.logger import get_logger
from bat.risk.live_guard import AccountRiskSnapshot, GuardDecision, LiveGuard, OrderIntent
from bat.risk.live_state import LiveState


@dataclass(frozen=True)
class OrderGatewayResult:
    sent: bool
    blocked_reason: str | None
    order_id: str | None
    order: object | None
    intent: OrderIntent
    guard_decision: GuardDecision


class OrderGateway:
    def __init__(self, broker, guard: LiveGuard, audit_ledger=None):
        self.broker = broker
        self.guard = guard
        self.audit_ledger = audit_ledger
        self.logger = get_logger("bat.order_gateway")
        self._in_flight = {}

    async def submit(
        self,
        intent: OrderIntent,
        guard_context: AccountRiskSnapshot,
    ) -> OrderGatewayResult:
        decision = self.guard.evaluate_order_request(intent, guard_context)
        if decision.state == LiveState.SHADOW.value:
            self.logger.info("Order captured in shadow mode: %s", intent)
            return self._blocked(intent, decision, "shadow_mode")
        if not decision.allowed:
            self.logger.warning("Order blocked by live guard: %s", decision.reason)
            return self._blocked(intent, decision, decision.reason)

        broker_symbol = getattr(self.broker, "symbol", None)
        if broker_symbol is not None and broker_symbol != intent.symbol:
            self.logger.warning(
                "Order blocked: broker symbol %s does not match intent symbol %s",
                broker_symbol,
                intent.symbol,
            )
            return self._blocked(intent, decision, "broker_symbol_mismatch")

        if intent.side not in {"BUY", "SELL"}:
            return self._blocked(intent, decision, "invalid_order_side")

        in_flight_key = (intent.symbol, intent.side)
        if in_flight_key in self._in_flight:
            self.logger.warning("Order blocked by in-flight idempotency key: %s", in_flight_key)
            return self._blocked(intent, decision, "in_flight_order")

        client_order_id = self._client_order_id(intent)
        self._in_flight[in_flight_key] = client_order_id

        try:
            if intent.side == "BUY":
                order = await self.broker.buy(
                    quote_qty=decision.adjusted_quote_qty,
                    client_order_id=client_order_id,
                )
            else:
                if decision.adjusted_quantity is None:
                    self._in_flight.pop(in_flight_key, None)
                    return self._blocked(intent, decision, "invalid_order_size")
                order = await self.broker.sell(
                    quantity=decision.adjusted_quantity,
                    client_order_id=client_order_id,
                )
        except Exception as exc:
            self.logger.exception(
                "Broker submission failed for symbol=%s side=%s client_order_id=%s: %s",
                intent.symbol,
                intent.side,
                client_order_id,
                exc,
            )
            return self._blocked(
                intent,
                decision,
                "execution_unknown",
                client_order_id,
                event_type="broker_exception",
            )

        if order is None:
            self.logger.warning("Broker returned no order for submitted intent: %s", intent)
            return self._blocked(intent, decision, "execution_unknown", client_order_id)

        self._in_flight.pop(in_flight_key, None)
        order_id = self._order_id(order)
        self._audit(
            "order_sent",
            intent,
            decision,
            decision.reason,
            True,
            order_id=order_id,
            client_order_id=client_order_id,
        )

        return OrderGatewayResult(
            sent=True,
            blocked_reason=None,
            order_id=order_id,
            order=order,
            intent=intent,
            guard_decision=decision,
        )

    def _blocked(
        self,
        intent: OrderIntent,
        decision: GuardDecision,
        reason: str,
        client_order_id: str | None = None,
        *,
        event_type: str | None = None,
    ) -> OrderGatewayResult:
        audit_event_type = event_type or (
            "execution_unknown" if reason == "execution_unknown" else "order_blocked"
        )
        self._audit(audit_event_type, intent, decision, reason, False, client_order_id=client_order_id)
        return OrderGatewayResult(
            sent=False,
            blocked_reason=reason,
            order_id=None,
            order=None,
            intent=intent,
            guard_decision=decision,
        )

    def _audit(
        self,
        event_type: str,
        intent: OrderIntent,
        decision: GuardDecision,
        reason: str | None,
        sent: bool,
        *,
        order_id: str | None = None,
        client_order_id: str | None = None,
    ) -> None:
        if self.audit_ledger is None:
            return
        self.audit_ledger.append(
            event_type,
            {
                "symbol": intent.symbol,
                "side": intent.side,
                "state": decision.state,
                "reason": reason,
                "sent": sent,
                "order_id": order_id,
                "client_order_id": client_order_id,
            },
        )

    @staticmethod
    def _order_id(order: object | None) -> str | None:
        if order is None:
            return None
        if isinstance(order, dict):
            return order.get("order_id") or order.get("orderId")
        return getattr(order, "order_id", None) or getattr(order, "orderId", None)

    @staticmethod
    def _client_order_id(intent: OrderIntent) -> str:
        fields = (
            intent.symbol,
            intent.side,
            intent.quote_qty,
            intent.quantity,
        )
        digest = hashlib.sha256(repr(fields).encode("utf-8")).hexdigest()[:16]
        symbol = OrderGateway._client_id_part(intent.symbol, 10)
        side = OrderGateway._client_id_part(intent.side, 4)
        return f"bat-{symbol}-{side}-{digest}"

    @staticmethod
    def _client_id_part(value: str, max_length: int) -> str:
        allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"
        cleaned = "".join(char for char in value if char in allowed)
        return (cleaned or "X")[:max_length]
