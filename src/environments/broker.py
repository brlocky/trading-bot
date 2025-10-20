import numpy as np
from typing import Tuple, List, Dict


class TradingBroker:
    """
    Broker with correct equity model for PPO:
    - cash: realized funds
    - unrealized_pnl: floating PnL on open positions
    - equity: cash + unrealized_pnl (used for position sizing)
    """

    def __init__(self, initial_balance: float = 100.0, quantity_precision: float = 0.001):
        self.initial_balance = round(initial_balance, 2)
        self.quantity_precision = quantity_precision

        # Core state
        self.cash = round(initial_balance, 2)
        self.cash_used = 0.0
        self.entry_price = 0.0

        # Trade size and direction
        self.signal = 0  # -1: short, 0: flat, 1: long
        self.position_shares = 0.0

        # Tracking
        self.last_price = 0.0
        self.total_trades = 0
        self.step_history: List[Dict] = []

    def reset(self):
        self.cash = round(self.initial_balance, 2)
        self.cash_used = 0.0
        self.position_shares = 0.0
        self.entry_price = 0.0
        self.last_price = 0.0
        self.total_trades = 0
        self.step_history.clear()

    # -------------------
    # MAIN STEP
    # -------------------
    def step(self, signal: int, signal_power: float, price: float, step_index: int = 0) -> Tuple[float, bool, bool]:
        self._validate_inputs(signal, signal_power)

        price_before = self.last_price if self.last_price > 0 else price

        # Market PnL
        market_pnl = self.calculate_unrealized_pnl(price) - self.calculate_unrealized_pnl(price_before)

        # Execute trade
        trade_occurred = False
        realized_pnl = 0.0

        # Check bankruptcy
        if self.is_bankrupt(price):
            realized_pnl, trade_occurred = self._force_liquidate(price)
            self._record_step(step_index, price, signal, signal_power, realized_pnl, trade_occurred)
            print("Broker is bankrupt. Liquidating all positions.")
            return 0.0, True, True

        current_signal = 0
        if self.position_shares > 0:
            current_signal = 1
        elif self.position_shares < 0:
            current_signal = -1

        # Action handling
        if signal == 0:  # Hold / Close position
            realized_pnl, trade_occurred = self._close_position(price)
        elif signal == 1:  # handle LONG signal
            if current_signal == 1:
                # Adjust long position
                realized_pnl, trade_occurred = self._adjust_position(signal_power, price)
            elif current_signal == -1:
                # Reverse from short to long
                realized_pnl, trade_occurred = self._reverse_position(signal_power, price)
            else:
                # Open new long position
                share_size = self.calculate_max_share_size(signal_power * self.cash, price)
                realized_pnl, trade_occurred = self._open_position(share_size, price)

        elif signal == -1:  # handle SHORT signal
            if current_signal == -1:
                # Adjust short position
                realized_pnl, trade_occurred = self._adjust_position(signal_power, price)
            elif current_signal == 1:
                # Reverse from long to short
                realized_pnl, trade_occurred = self._reverse_position(signal_power, price)
            else:
                # Open new short position
                share_size = self.calculate_max_share_size(signal_power * self.cash, price)
                realized_pnl, trade_occurred = self._open_position(-share_size, price)

        step_pnl = market_pnl

        self.last_price = price
        self._record_step(step_index, price, signal, signal_power, step_pnl, trade_occurred)

        return step_pnl, trade_occurred, False

    # -------------------
    # POSITION MANAGEMENT
    # -------------------
    def _adjust_position(self, signal_power: float, price: float) -> Tuple[float, bool]:
        target_value = signal_power * self.cash
        current_value = abs(self.position_shares * self.entry_price) if self.position_shares != 0 else 0.0
        diff_value = target_value - current_value

        shares_to_trade = self.calculate_max_share_size(abs(diff_value), price)

        if diff_value > 0 and shares_to_trade > 0:
            return self._increase_position(shares_to_trade, price)

        if diff_value < 0 and shares_to_trade > 0:
            return self._decrease_position(shares_to_trade, price)

        return 0.0, False  # No adjustment needed

    def _reverse_position(self, signal_power: float, price: float) -> Tuple[float, bool]:
        if self.position_shares == 0.0:
            raise RuntimeError("Cannot reverse position when no position is open.")

        # Current Trade Direction
        current_direction = np.sign(self.position_shares)

        # Initialize variables
        realized_pnl = 0.0
        trade_occurred = False

        # Close existing long Position
        if current_direction == 1:
            realized_pnl, _ = self._close_position(price)

        # Close existing short Position
        if current_direction == -1:
            realized_pnl, _ = self._close_position(price)

        # Calculate new shares to open
        new_shares = self.calculate_max_share_size(self.cash * signal_power, price)

        # Open Short Position when last direction was long
        if current_direction == 1:
            _, trade_occurred = self._open_position(-new_shares, price)

        # Open Long Position when last direction was short
        if current_direction == -1:
            _, trade_occurred = self._open_position(new_shares, price)

        return realized_pnl, trade_occurred

    # -------------------
    # LONG / SHORT HANDLERS
    # -------------------

    def _open_position(self, share_size: float, price: float) -> Tuple[float, bool]:
        if self.position_shares != 0.0:
            raise RuntimeError("Position already open, cannot open new position.")

        self.position_shares = share_size
        self.entry_price = price
        self.total_trades += 1
        self.cash_used = round(self.entry_price * abs(self.position_shares), 2)
        return 0.0, True

    def _increase_position(self, share_size: float, price: float) -> Tuple[float, bool]:
        if self.position_shares == 0.0:
            raise RuntimeError("Cannot increase position when not currently holding.")
        sign = np.sign(self.position_shares)

        old_value = abs(self.position_shares) * self.entry_price
        new_value = abs(share_size) * price
        self.entry_price = (old_value + new_value) / abs(self.position_shares + share_size)
        self.position_shares += sign * share_size
        self.cash_used = round(self.entry_price * abs(self.position_shares), 2)
        self.total_trades += 1
        return 0.0, True

    def _decrease_position(self, shares: float, price: float) -> Tuple[float, bool]:
        if self.position_shares == 0.0:
            raise RuntimeError("Cannot decrease position when not currently holding.")

        if shares > abs(self.position_shares):
            return self._close_position(price)

        if shares == abs(self.position_shares):
            return self._close_position(price)

        realized_pnl = 0.0
        if self.position_shares > 0:
            realized_pnl = abs(shares) * (price - self.entry_price)
            self.position_shares -= shares
        if self.position_shares < 0:
            realized_pnl = abs(shares) * (self.entry_price - price)
            self.position_shares += shares

        self.cash += realized_pnl
        self.cash_used = round(self.entry_price * abs(self.position_shares), 2)
        self.total_trades += 1
        return realized_pnl, True

    def _close_position(self, price: float) -> Tuple[float, bool]:
        realized_pnl = self.calculate_unrealized_pnl(price)
        self.cash += realized_pnl
        self.cash_used = 0.0
        self.position_shares = 0.0
        self.entry_price = 0.0
        self.total_trades += 1
        return realized_pnl, True

    def _reverse_long_to_short(self, shares: float, price: float) -> Tuple[float, bool]:
        realized_pnl = self._close_position(price)[0]
        self.position_shares = shares  # negative
        self.entry_price = price
        self.total_trades += 1
        return realized_pnl, True

    # -------------------
    # UTILITIES
    # -------------------

    def calculate_max_share_size(
        self,
        cash: float,           # total available cash
        price: float,          # current price of asset
    ) -> float:
        """
        Calculate the number of shares to buy/sell based on position fraction, available cash,
        price, and quantity precision.
        """
        raw_size = cash / price
        # Round down to nearest precision
        size = int(raw_size / self.quantity_precision) * self.quantity_precision
        return round(size, 10)

    def calculate_unrealized_pnl(self, price: float) -> float:
        if self.position_shares == 0:
            return 0.0
        if self.position_shares > 0:
            return self.position_shares * (price - self.entry_price)
        else:
            return abs(self.position_shares) * (self.entry_price - price)

    def calculate_portfolio_value(self, price: float) -> float:
        return round(self.cash + self.calculate_unrealized_pnl(price), 2)

    def calculate_available_cash(self) -> float:
        return round(self.cash - self.cash_used, 2)

    def _round_shares(self, shares: float) -> float:
        rounded = round(shares / self.quantity_precision) * self.quantity_precision
        return max(abs(rounded), self.quantity_precision) * (1 if shares > 0 else -1)

    def is_bankrupt(self, price: float) -> bool:
        return self.calculate_portfolio_value(price) <= 0

    def _force_liquidate(self, price: float):
        if self.position_shares != 0:
            return self._close_position(price)
        return 0.0, False

    def _validate_inputs(self, signal: int, position_size: float):
        if signal not in [-1, 0, 1]:
            raise ValueError(f"Signal must be in [-1, 1], got {signal}")
        if not (0.0 <= position_size <= 1.0):
            raise ValueError(f"Position size must be in [0, 1], got {position_size}")

    def _record_step(self, step_index: int, price: float, signal: float, position_size: float, step_pnl: float, traded: bool):
        self.step_history.append(self._create_step_record(step_index, price, signal, position_size, step_pnl, traded))

    def _create_step_record(self, step_index: int, price: float, signal: float, position_size: float, step_pnl: float, traded: bool):
        return {
            'step': step_index,
            'signal': signal,
            'position_size': position_size,
            'price': price,
            'entry_price': self.entry_price,
            'position_shares': self.position_shares,
            'cash': self.cash,
            'cash_used': self.cash_used,
            'unrealized_pnl': self.calculate_unrealized_pnl(price),
            'equity': self.calculate_portfolio_value(price),
            'step_pnl': step_pnl,
            'traded': traded,
            'total_trades': self.total_trades

        }

    def get_state(self) -> Dict:
        if not self.step_history:
            raise ValueError("No steps recorded yet.")
        return self.step_history[-1]
