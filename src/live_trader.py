"""
Quantum Trading System – Live Trader Commander
Executes real trades on Hyperliquid using the private key.
Uses the same ML signal pipeline as the paper broker but commits real orders.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import AppConfig
from src.risk_manager import PositionSpec
from src.utils import get_logger, utc_now_ms

log = get_logger(__name__)

try:
    import eth_account
    from eth_account.signers.local import LocalAccount

    _ETH_AVAILABLE = True
except ImportError:
    _ETH_AVAILABLE = False
    log.warning("eth_account not installed – live order signing unavailable")

try:
    import requests as _requests

    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

# Hyperliquid exchange endpoint
_HL_EXCHANGE_URL = "https://api.hyperliquid.xyz/exchange"
_HL_INFO_URL = "https://api.hyperliquid.xyz/info"
_REQUEST_TIMEOUT = 30


@dataclass
class LiveOrderResult:
    order_id: str
    symbol: str
    side: str
    filled_price: float
    filled_size: float
    status: str               # "filled" | "open" | "error"
    raw_response: Dict
    timestamp_ms: int


class LiveTrader:
    """
    Commander for live perpetuals trading on Hyperliquid.
    Signs and submits orders using Ethereum private key.
    Safety guards:
    - All positions are isolated-margin
    - Position size capped by risk manager
    - Requires explicit LIVE_TRADING_ENABLED=true env var
    """

    def __init__(self, config: AppConfig, private_key: Optional[str] = None) -> None:
        self.cfg = config
        self._private_key = private_key
        self._account: Optional["LocalAccount"] = None
        self._session = None
        self._live_enabled = self._is_live_enabled()

        if _ETH_AVAILABLE and private_key:
            self._account = eth_account.Account.from_key(private_key)
            log.info(
                "Live trader initialised. Wallet: %s",
                self._account.address[:10] + "…",
            )
        if _REQUESTS_AVAILABLE:
            import requests
            self._session = requests.Session()
            self._session.headers.update({"Content-Type": "application/json"})

    # ── Public interface ───────────────────────────────────────────────────────

    def place_market_order(self, spec: PositionSpec) -> Optional[LiveOrderResult]:
        """
        Place an isolated-margin market order.
        Returns LiveOrderResult or None if rejected / simulation mode.
        """
        if not self._live_enabled:
            log.info(
                "[DRY-RUN] Would place %s %s %.4f contracts @ market (lev=%dx)",
                spec.side.upper(), spec.symbol, spec.size_contracts, spec.leverage,
            )
            return self._dry_run_result(spec)

        if self._account is None:
            log.error("No private key – cannot place live order")
            return None

        # Set leverage first (isolated)
        self._set_leverage(spec.symbol, spec.leverage, isolated=True)
        time.sleep(0.3)

        is_buy = spec.side == "long"
        order = self._build_order(spec, is_buy)
        result = self._send_order(order)
        if result:
            log.info(
                "LIVE ORDER placed: %s %s | id=%s | filled@%.4f",
                spec.side.upper(), spec.symbol,
                result.order_id, result.filled_price,
            )
        return result

    def close_position(
        self,
        symbol: str,
        side: str,
        size_contracts: float,
        current_price: float,
    ) -> Optional[LiveOrderResult]:
        """Close (reduce-only) an existing position."""
        if not self._live_enabled:
            log.info("[DRY-RUN] Would close %s %s %.4f", side, symbol, size_contracts)
            return None

        is_buy = side == "short"  # to close short, buy; to close long, sell
        close_order = {
            "coin": symbol,
            "is_buy": is_buy,
            "sz": round(size_contracts, 6),
            "limit_px": None,   # market order
            "order_type": {"market": {}},
            "reduce_only": True,
        }
        return self._send_order(close_order)

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Fetch open positions from Hyperliquid."""
        if self._account is None or self._session is None:
            return []
        payload = {
            "type": "clearinghouseState",
            "user": self._account.address,
        }
        try:
            resp = self._session.post(_HL_INFO_URL, json=payload, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            return data.get("assetPositions", [])
        except Exception as exc:
            log.error("Failed to fetch open positions: %s", exc)
            return []

    def get_account_state(self) -> Dict[str, Any]:
        """Fetch full account state including equity and margin usage."""
        if self._account is None or self._session is None:
            return {}
        payload = {
            "type": "clearinghouseState",
            "user": self._account.address,
        }
        try:
            resp = self._session.post(_HL_INFO_URL, json=payload, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            log.error("Failed to fetch account state: %s", exc)
            return {}

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _is_live_enabled() -> bool:
        import os
        return os.environ.get("LIVE_TRADING_ENABLED", "false").lower() == "true"

    def _set_leverage(self, symbol: str, leverage: int, isolated: bool = True) -> bool:
        """Set leverage for a symbol via Hyperliquid API."""
        if self._account is None or self._session is None:
            return False
        action = {
            "type": "updateLeverage",
            "asset": self._asset_index(symbol),
            "isCross": not isolated,
            "leverage": leverage,
        }
        payload = self._sign_action(action)
        try:
            resp = self._session.post(_HL_EXCHANGE_URL, json=payload, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            log.info("Leverage set to %dx (isolated=%s) for %s", leverage, isolated, symbol)
            return True
        except Exception as exc:
            log.error("Failed to set leverage: %s", exc)
            return False

    def _build_order(self, spec: PositionSpec, is_buy: bool) -> Dict[str, Any]:
        return {
            "coin": spec.symbol,
            "is_buy": is_buy,
            "sz": round(spec.size_contracts, 6),
            "limit_px": None,
            "order_type": {"market": {}},
            "reduce_only": False,
        }

    def _send_order(self, order: Dict) -> Optional[LiveOrderResult]:
        if self._session is None:
            return None
        action = {"type": "order", "orders": [order], "grouping": "na"}
        payload = self._sign_action(action)
        try:
            resp = self._session.post(_HL_EXCHANGE_URL, json=payload, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            statuses = data.get("response", {}).get("data", {}).get("statuses", [{}])
            status = statuses[0] if statuses else {}
            filled = status.get("filled", {})
            return LiveOrderResult(
                order_id=str(status.get("resting", {}).get("oid", "market")),
                symbol=order["coin"],
                side="long" if order["is_buy"] else "short",
                filled_price=float(filled.get("avgPx", 0)),
                filled_size=float(filled.get("totalSz", 0)),
                status="filled" if filled else "open",
                raw_response=data,
                timestamp_ms=utc_now_ms(),
            )
        except Exception as exc:
            log.error("Order placement failed: %s", exc)
            return None

    def _sign_action(self, action: Dict) -> Dict:
        """Sign a Hyperliquid action with the private key (EIP-712 style)."""
        if self._account is None:
            raise RuntimeError("No account configured for signing")
        import hashlib, struct
        nonce = utc_now_ms()
        action_str = json.dumps(action, separators=(",", ":"), sort_keys=True)
        msg = f"\x19Ethereum Signed Message:\n{len(action_str)}{action_str}{nonce}"
        msg_hash = hashlib.sha256(msg.encode()).digest()
        signed = self._account.signHash(msg_hash)
        return {
            "action": action,
            "nonce": nonce,
            "signature": {
                "r": hex(signed.r),
                "s": hex(signed.s),
                "v": signed.v,
            },
            "vaultAddress": None,
        }

    @staticmethod
    def _asset_index(symbol: str) -> int:
        """Return Hyperliquid asset index. Needs full meta lookup in production."""
        _KNOWN = {"BTC": 0, "ETH": 1, "SOL": 2, "ARB": 3, "OP": 4, "MATIC": 5}
        return _KNOWN.get(symbol.upper(), 0)

    def _dry_run_result(self, spec: PositionSpec) -> LiveOrderResult:
        return LiveOrderResult(
            order_id="dry-run",
            symbol=spec.symbol,
            side=spec.side,
            filled_price=spec.entry_price,
            filled_size=spec.size_contracts,
            status="dry_run",
            raw_response={},
            timestamp_ms=utc_now_ms(),
        )
