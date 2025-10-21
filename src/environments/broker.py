import math
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
        self.equity = self.cash
        self.cash_used = 0.0
        self.entry_price = 0.0

        # Trade size and direction
        self.signal = 0  # -1: short, 0: flat, 1: long
        self.position_shares = 0.0
        self.traded = False

        # Tracking
        self.last_price = 0.0
        self.total_trades = 0
        self.step_history: List[Dict] = []

    def reset(self):
        self.cash = round(self.initial_balance, 2)
        self.equity = self.cash
        self.cash_used = 0.0
        self.position_shares = 0.0
        self.entry_price = 0.0
        self.last_price = 0.0
        self.total_trades = 0
        self.traded = False
        self.step_history.clear()

    # -------------------
    # MAIN STEP
    # -------------------
    def step(self, signal: int, signal_power: float, price: float, step_index: int = 0) -> Tuple[float, bool, bool]:
        self._validate_inputs(signal, signal_power)

        # Execute trade
        self.traded = False
        step_pnl = self.calculate_unrealized_pnl(price) - self.calculate_unrealized_pnl(self.last_price)

        # Check bankruptcy
        if self.is_bankrupt(price):
            realized_pnl = self._force_liquidate(price)
            self._record_step(step_index, price, signal, signal_power, realized_pnl)
            print("Broker is bankrupt. Liquidating all positions.")
            return 0.0, True, True

        current_signal = 0
        if self.position_shares > 0:
            current_signal = 1
        elif self.position_shares < 0:
            current_signal = -1

        realized_pnl = 0.0
        # Action handling
        if signal == 0 or signal_power == 0.0:  # Hold / Close position
            if abs(self.position_shares) > 0:
                realized_pnl = self._close_position(price)
        elif signal == 1:  # handle LONG signal
            if current_signal == 1:
                # Adjust long position
                realized_pnl = self._adjust_position(signal_power, price)
            elif current_signal == -1:
                # Reverse from short to long
                realized_pnl = self._reverse_position(signal_power, price)
            else:
                # Open new long position
                share_size = self.calculate_max_share_size(signal_power * self.cash, price)
                realized_pnl = self._open_position(share_size, price)

        elif signal == -1:  # handle SHORT signal
            if current_signal == -1:
                # Adjust short position
                realized_pnl = self._adjust_position(signal_power, price)
            elif current_signal == 1:
                # Reverse from long to short
                realized_pnl = self._reverse_position(signal_power, price)
            else:
                # Open new short position
                share_size = self.calculate_max_share_size(signal_power * self.cash, price)
                realized_pnl = self._open_position(-share_size, price)

        self.last_price = price

        self.equity = self.calculate_portfolio_value(price)
        self._record_step(step_index, price, signal, signal_power, step_pnl)

        return step_pnl, self.traded, False

    # -------------------
    # POSITION MANAGEMENT
    # -------------------
    def _adjust_position(self, signal_power: float, price: float) -> float:
        target_value = round(signal_power * self.cash, 2)
        current_value = round(abs(self.position_shares * self.entry_price), 2) if self.position_shares != 0 else 0.0
        diff_value = round(target_value - current_value, 2)

        shares_to_trade = self.calculate_max_share_size(abs(diff_value), price)

        if diff_value > 0 and shares_to_trade > 0:
            return self._increase_position(shares_to_trade, price)

        if diff_value < 0 and shares_to_trade > 0:
            return self._decrease_position(shares_to_trade, price)

        return 0.0

    def _reverse_position(self, signal_power: float, price: float) -> float:
        if self.position_shares == 0.0:
            raise RuntimeError("Cannot reverse position when no position is open.")

        # Current Trade Direction
        current_direction = np.sign(self.position_shares)

        # Initialize variables
        realized_pnl = 0.0

        # Close existing long Position
        if current_direction == 1:
            realized_pnl = self._close_position(price)

        # Close existing short Position
        if current_direction == -1:
            realized_pnl = self._close_position(price)

        # Calculate new shares to open
        new_shares = self.calculate_max_share_size(self.cash * signal_power, price)

        # Open Short Position when last direction was long
        if current_direction == 1:
            self._open_position(-new_shares, price)

        # Open Long Position when last direction was short
        if current_direction == -1:
            self._open_position(new_shares, price)

        return realized_pnl

    # -------------------
    # LONG / SHORT HANDLERS
    # -------------------

    def _open_position(self, share_size: float, price: float) -> float:
        if self.position_shares != 0.0:
            raise RuntimeError("Position already open, cannot open new position.")
        if share_size == 0.0:
            return 0.0

        self.position_shares = share_size
        self.entry_price = price
        self.total_trades += 1
        self.cash_used = round(self.entry_price * abs(self.position_shares), 2)
        self.traded = True
        return 0.0

    def _increase_position(self, share_size: float, price: float) -> float:
        if self.position_shares == 0.0:
            raise RuntimeError("Cannot increase position when not currently holding.")
        sign = np.sign(self.position_shares)

        old_value = abs(self.position_shares) * self.entry_price
        new_value = abs(share_size) * price
        total_shares = abs(self.position_shares) + abs(share_size)

        self.entry_price = round((old_value + new_value) / total_shares, 2)

        self.position_shares = round(self.position_shares + sign * share_size, 10)
        self.cash_used = round(self.entry_price * abs(self.position_shares), 2)
        self.total_trades += 1
        self.traded = True
        return 0.0

    def _decrease_position(self, shares: float, price: float) -> float:
        if self.position_shares == 0.0:
            raise RuntimeError("Cannot decrease position when not currently holding.")

        if shares > abs(self.position_shares):
            return self._close_position(price)

        if shares == abs(self.position_shares):
            return self._close_position(price)

        realized_pnl = 0.0
        if self.position_shares > 0:
            realized_pnl = abs(shares) * (price - self.entry_price)
            self.position_shares = round(self.position_shares - shares, 10)
        if self.position_shares < 0:
            realized_pnl = abs(shares) * (self.entry_price - price)
            self.position_shares = round(self.position_shares + shares, 10)

        self.cash = round(self.cash + realized_pnl, 2)
        self.cash_used = round(self.entry_price * abs(self.position_shares), 2)
        self.total_trades += 1
        self.traded = True
        return realized_pnl

    def _close_position(self, price: float) -> float:
        realized_pnl = self.calculate_unrealized_pnl(price)
        self.cash = round(self.cash + realized_pnl, 2)
        self.cash_used = 0.0
        self.position_shares = 0.0
        self.entry_price = 0.0
        self.total_trades += 1
        self.traded = True
        return realized_pnl

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
        price, and quantity precision. Always rounds down and never exceeds available cash.
        """
        if price <= 0 or self.quantity_precision <= 0:
            return 0.0
        raw_size = cash / price
        # Floor to nearest precision
        floored_size = math.floor(raw_size / self.quantity_precision) * self.quantity_precision

        return round(floored_size, 10)

    def calculate_unrealized_pnl(self, price: float) -> float:
        if self.position_shares == 0:
            return 0.0
        if self.position_shares > 0:
            return self.position_shares * (price - self.entry_price)
        else:
            return -self.position_shares * (self.entry_price - price)

    def calculate_portfolio_value(self, price: float) -> float:
        return round(self.cash + self.calculate_unrealized_pnl(price), 2)

    def is_bankrupt(self, price: float) -> bool:
        return self.calculate_portfolio_value(price) <= 0

    def _force_liquidate(self, price: float) -> float:
        if self.position_shares != 0:
            return self._close_position(price)
        return 0.0

    def _validate_inputs(self, signal: int, position_size: float):
        if signal not in [-1, 0, 1]:
            raise ValueError(f"Signal must be in [-1, 1], got {signal}")
        if not (0.0 <= position_size <= 1.0):
            raise ValueError(f"Position size must be in [0, 1], got {position_size}")

    def _record_step(self, step_index: int, price: float, signal: float, position_size: float, step_pnl: float):
        self.step_history.append(self._create_step_record(step_index, price, signal, position_size, step_pnl))

    def _create_step_record(self, step_index: int, price: float, signal: float, position_size: float, step_pnl: float):
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
            'equity': self.equity,
            'step_pnl': step_pnl,
            'traded': self.traded,
            'total_trades': self.total_trades,
            'is_bankrupt': self.is_bankrupt(price)
        }

    def get_state(self) -> Dict:
        if not self.step_history:
            raise ValueError("No steps recorded yet.")
        return self.step_history[-1]
