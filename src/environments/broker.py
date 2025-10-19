"""
Trading Broker - Clean Refactored Implementation
Proper handling of long and short positions with clear separation of concerns
"""

import numpy as np
from typing import Tuple, List, Dict


class TradingBroker:
    """
    Simple broker with proper short position handling.

    Capital Model:
    - balance: Free cash available
    - For LONGS: capital_used = shares × entry_price (money tied up in position)
    - For SHORTS: capital_used = shares × entry_price (collateral/proceeds held)
    - Portfolio = balance + capital_used + unrealized_pnl
    """

    def __init__(self, initial_balance: float = 100.0, quantity_precision: float = 0.001):
        self.initial_balance = round(initial_balance, 2)
        self.quantity_precision = quantity_precision

        # Core state
        self.balance = round(initial_balance, 2)
        self.position_shares = 0.0
        self.entry_price = 0.0
        self.capital_used = 0.0

        # Tracking
        self.last_price = 0.0
        self.total_trades = 0
        self.step_history: List[Dict] = []

    def reset(self):
        """Reset broker to initial state"""
        self.balance = round(self.initial_balance, 2)
        self.position_shares = 0.0
        self.entry_price = 0.0
        self.capital_used = 0.0
        self.last_price = 0.0
        self.total_trades = 0
        self.step_history.clear()

    def step(self, signal: float, position_size: float, price: float, step_index: int = 0) -> Tuple[float, bool, bool]:
        """
        Execute trading step with proper P&L attribution

        P&L Calculation:
        - Market P&L: Price movement on existing position BEFORE any trade
        - Realized P&L: P&L from closing/reducing positions during the trade
        - Step P&L: Market P&L + Realized P&L (excludes capital reallocation)

        Args:
            signal: -1 (short), 0 (hold/reduce), 1 (long)
            position_size: 0-1, percentage of portfolio to use
            price: Current market price
            step_index: Step number for tracking

        Returns:
            (step_pnl, trade_occurred, is_bankrupt)
        """
        self._validate_inputs(signal, position_size)

        # Store old position state for P&L calculation
        old_position_shares = self.position_shares
        old_entry_price = self.entry_price
        price_before = self.last_price if self.last_price > 0 else price

        # Check bankruptcy
        is_bankrupt = self.is_bankrupt(price)
        if is_bankrupt:
            self._force_liquidate(price)
            return 0.0, True, True

        # Calculate market P&L BEFORE any trade (price movement on existing position)
        market_pnl = 0.0
        if abs(old_position_shares) > self.quantity_precision and price_before > 0:
            if old_position_shares > 0:  # Long position
                market_pnl = old_position_shares * (price - price_before)
            else:  # Short position
                market_pnl = abs(old_position_shares) * (price_before - price)
            market_pnl = round(market_pnl, 2)

        # Execute trading logic
        trade_occurred = False
        realized_pnl = 0.0

        if signal == 0:
            realized_pnl, trade_occurred = self._reduce_position(position_size, price)
        else:
            direction = int(np.sign(signal))
            current_direction = int(np.sign(self.position_shares)) if abs(self.position_shares) > self.quantity_precision else 0
            if current_direction == direction:
                realized_pnl, trade_occurred = self._reduce_position(position_size, price)
            else:
                realized_pnl, trade_occurred = self._target_position(direction, position_size, price)

        # Total step P&L = market P&L from price movement + realized P&L from trade
        step_pnl = market_pnl + realized_pnl

        self.last_price = price
        self._record_step(step_index, price, signal, position_size, step_pnl, trade_occurred)
        return step_pnl, trade_occurred, False

    def close(self, price: float, step_index: int) -> Dict:
        """
        Close any remaining position and update the last step values

        This is useful for properly finalizing the broker state at the end of a trading session.
        It will close out any open position and update the specified step in history with final values.

        Args:
            price: Current market price
            step_index: Step number to find and update in history

        Returns:
            updated_step: The updated step dictionary

        Raises:
            ValueError: If step_index is not found in step_history
        """
        # Find the step with matching step_index
        target_step = None
        if self.step_history:
            for step in self.step_history:
                if step['step'] == step_index:
                    target_step = step
                    break

        # Raise error if step not found
        if target_step is None:
            raise ValueError(f"Step with index {step_index} not found in step_history")

        # Check if there's a position to close
        if abs(self.position_shares) < self.quantity_precision:
            # No position to close, just return the existing step
            return target_step

        # Calculate portfolio value before closing
        portfolio_before = self.calculate_portfolio_value(price)

        # Close the position
        shares_to_close = -self.position_shares
        self._execute_trade(shares_to_close, price)

        # Calculate final portfolio value
        portfolio_after = self.calculate_portfolio_value(price)
        step_pnl = portfolio_after - portfolio_before

        # Update last price
        self.last_price = price

        # Update the step with final values
        target_step.update({
            'position_shares': self.position_shares,
            'entry_price': self.entry_price,
            'balance': self.balance,
            'capital_used': self.capital_used,
            'unrealized_pnl': self.calculate_unrealized_pnl(price),
            'portfolio_value': self.calculate_portfolio_value(price),
            'step_pnl': target_step['step_pnl'] + step_pnl,
            'traded': True,
            'total_trades': self.total_trades
        })

        return target_step

    # ============================================================================
    # POSITION MANAGEMENT - Clean separation of concerns
    # ============================================================================

    def _target_position(self, direction: int, exposure: float, price: float) -> Tuple[float, bool]:
        """Target a position with specified direction and exposure"""

        # Calculate current exposure as percentage of portfolio
        current_portfolio = self.calculate_portfolio_value(price)
        if current_portfolio <= 0:
            return 0.0, False

        current_exposure = self.capital_used / current_portfolio

        # Skip if already at target (same direction and similar exposure)
        if (int(np.sign(self.position_shares)) == direction and
                abs(current_exposure - exposure) < 0.01):  # 1% tolerance
            return 0.0, False

        # Calculate target position size based on CURRENT portfolio value
        # This makes exposure always relative to total portfolio (the standard definition)
        max_shares = self._calculate_max_affordable_shares(current_portfolio, price)
        target_shares = direction * exposure * max_shares
        shares_to_trade = target_shares - self.position_shares

        if abs(shares_to_trade) <= self.quantity_precision:
            return 0.0, False

        # Execute trade
        realized_pnl, success = self._execute_trade(shares_to_trade, price)

        return realized_pnl, success

    def _reduce_position(self, keep_fraction: float, price: float) -> Tuple[float, bool]:
        """Reduce current position to keep only a fraction of it

        Args:
            keep_fraction: Fraction of current position to keep (0-1)
                        e.g., 0.5 = keep 50%, reduce 50%
                        e.g., 0.0 = close entire position
            price: Current market price

        Returns:
            (realized_pnl, success)
        """
        if abs(self.position_shares) == 0:
            return 0.0, False

        # Calculate target shares: keep only the specified fraction
        target_shares = self.position_shares * keep_fraction
        shares_to_trade = target_shares - self.position_shares

        if abs(shares_to_trade) <= self.quantity_precision:
            return 0.0, False

        realized_pnl, success = self._execute_trade(shares_to_trade, price)

        return realized_pnl, success

    # ============================================================================
    # TRADE EXECUTION - Core trading logic
    # ============================================================================

    def _execute_trade(self, shares_to_trade: float, price: float) -> Tuple[float, bool]:
        """Execute trade by delegating to specific scenario handlers"""
        if abs(shares_to_trade) < self.quantity_precision:
            return 0.0, False

        shares_to_trade = self._round_shares(shares_to_trade)
        old_position = self.position_shares
        new_position = old_position + shares_to_trade

        # Detect scenario and delegate to specialized handler
        if abs(old_position) < self.quantity_precision:
            # Opening new position from flat
            if shares_to_trade > 0:
                return self._open_long(shares_to_trade, price)
            else:
                return self._open_short(shares_to_trade, price)

        elif old_position > 0:
            # Has long position
            if shares_to_trade > 0:
                return self._increase_long(shares_to_trade, price)
            elif new_position >= self.quantity_precision:
                return self._decrease_long(shares_to_trade, price)
            elif abs(new_position) <= self.quantity_precision:
                return self._close_long(shares_to_trade, price)
            else:
                return self._reverse_long_to_short(shares_to_trade, price)

        else:
            # Has short position
            if shares_to_trade < 0:
                return self._increase_short(shares_to_trade, price)
            elif new_position <= -self.quantity_precision:
                return self._decrease_short(shares_to_trade, price)
            elif abs(new_position) <= self.quantity_precision:
                return self._close_short(shares_to_trade, price)
            else:
                return self._reverse_short_to_long(shares_to_trade, price)

    # ============================================================================
    # LONG POSITION HANDLERS
    # ============================================================================

    def _open_long(self, shares: float, price: float) -> Tuple[float, bool]:
        """Open new long position"""
        trade_value = round(abs(shares) * price, 2)

        # Check affordability
        if trade_value > round(self.balance + 0.10, 2):
            return 0.0, False

        # Buy shares: PAY money
        self.balance = round(self.balance - trade_value, 2)
        self.position_shares = shares
        self.entry_price = price
        self.capital_used = trade_value
        self.total_trades += 1
        return 0.0, True

    def _increase_long(self, shares: float, price: float) -> Tuple[float, bool]:
        """Add to existing long position"""
        trade_value = round(abs(shares) * price, 2)

        # Check affordability
        if trade_value > round(self.balance + 0.10, 2):
            return 0.0, False

        # Buy more shares: PAY money
        old_position = self.position_shares
        old_entry_price = self.entry_price
        new_position = old_position + shares

        self.balance = round(self.balance - trade_value, 2)
        self.position_shares = new_position

        # Weighted average entry price
        old_value = old_position * old_entry_price
        new_value = abs(shares) * price
        self.entry_price = (old_value + new_value) / abs(new_position)
        self.capital_used = round(abs(self.position_shares) * self.entry_price, 2)
        self.total_trades += 1
        return 0.0, True

    def _decrease_long(self, shares: float, price: float) -> Tuple[float, bool]:
        """Partially close long position"""
        trade_value = round(abs(shares) * price, 2)
        old_entry_price = self.entry_price

        # Sell some shares: RECEIVE money
        self.balance = round(self.balance + trade_value, 2)
        self.position_shares = self.position_shares + shares

        # Calculate realized P&L on portion sold
        realized_pnl = round(abs(shares) * (price - old_entry_price), 2)
        self.capital_used = round(abs(self.position_shares) * self.entry_price, 2)
        self.total_trades += 1
        return realized_pnl, True

    def _close_long(self, shares: float, price: float) -> Tuple[float, bool]:
        """Close entire long position"""
        trade_value = round(abs(shares) * price, 2)
        old_position = self.position_shares
        old_entry_price = self.entry_price

        # Sell all shares: RECEIVE money
        self.balance = round(self.balance + trade_value, 2)
        realized_pnl = round(old_position * (price - old_entry_price), 2)

        self.position_shares = 0.0
        self.entry_price = 0.0
        self.capital_used = 0.0
        self.total_trades += 1
        return realized_pnl, True

    def _reverse_long_to_short(self, shares: float, price: float) -> Tuple[float, bool]:
        """Close long and open short in one trade"""
        old_position = self.position_shares
        old_entry_price = self.entry_price

        # Step 1: Close long position
        close_value = round(old_position * price, 2)
        realized_pnl = round(old_position * (price - old_entry_price), 2)

        # Sell long: RECEIVE money
        self.balance = round(self.balance + close_value, 2)

        # Step 2: Open short with remaining
        remaining_shares = old_position + shares  # Negative value
        short_value = round(abs(remaining_shares) * price, 2)

        # Check affordability for short
        if short_value > round(self.balance + 0.10, 2):
            # Can't afford the short, stay flat
            self.position_shares = 0.0
            self.entry_price = 0.0
            self.capital_used = 0.0
        else:
            # Lock up capital for short
            self.balance = round(self.balance - short_value, 2)
            self.position_shares = remaining_shares
            self.entry_price = price
            self.capital_used = short_value

        self.total_trades += 1
        return realized_pnl, True
    # ============================================================================
    # SHORT POSITION HANDLERS
    # ============================================================================

    def _open_short(self, shares: float, price: float) -> Tuple[float, bool]:
        """Open new short position"""
        trade_value = round(abs(shares) * price, 2)

        # Check affordability
        if trade_value > round(self.balance + 0.10, 2):
            return 0.0, False

        # Short: Lock up capital as collateral
        self.balance = round(self.balance - trade_value, 2)
        self.position_shares = shares
        self.entry_price = price
        self.capital_used = trade_value
        self.total_trades += 1
        return 0.0, True

    def _increase_short(self, shares: float, price: float) -> Tuple[float, bool]:
        """Add to existing short position"""
        trade_value = round(abs(shares) * price, 2)

        # Check affordability
        if trade_value > round(self.balance + 0.10, 2):
            return 0.0, False

        # Short more shares: Lock up more capital
        old_position = self.position_shares
        old_entry_price = self.entry_price
        new_position = old_position + shares

        self.balance = round(self.balance - trade_value, 2)
        self.position_shares = new_position

        # Weighted average entry price
        old_value = abs(old_position) * old_entry_price
        new_value = abs(shares) * price
        self.entry_price = (old_value + new_value) / abs(new_position)
        self.capital_used = round(abs(self.position_shares) * self.entry_price, 2)
        self.total_trades += 1
        return 0.0, True

    def _decrease_short(self, shares: float, price: float) -> Tuple[float, bool]:
        """Partially close short position"""
        old_entry_price = self.entry_price

        # Cost to buy back shares at current price
        buyback_cost = round(abs(shares) * price, 2)

        # Capital released from original short
        proportional_capital = round(abs(shares) * old_entry_price, 2)

        # Realized P&L = capital - buyback cost
        realized_pnl = round(proportional_capital - buyback_cost, 2)

        # Return capital + profit (NOT just profit!)
        cash_returned = proportional_capital + realized_pnl

        self.balance = round(self.balance + cash_returned, 2)
        self.position_shares = self.position_shares + shares
        self.capital_used = round(abs(self.position_shares) * old_entry_price, 2)
        self.total_trades += 1

        return realized_pnl, True

    def _close_short(self, shares: float, price: float) -> Tuple[float, bool]:
        """Close entire short position"""
        old_position = self.position_shares
        old_entry_price = self.entry_price

        # Capital released from original short
        proportional_capital = round(abs(old_position) * old_entry_price, 2)

        # Realized P&L
        realized_pnl = round(abs(old_position) * (old_entry_price - price), 2)

        # Return capital + profit
        cash_returned = proportional_capital + realized_pnl

        self.balance = round(self.balance + cash_returned, 2)
        self.position_shares = 0.0
        self.entry_price = 0.0
        self.capital_used = 0.0
        self.total_trades += 1
        return realized_pnl, True

    def _reverse_short_to_long(self, shares: float, price: float) -> Tuple[float, bool]:
        """Close short and open long in one trade"""
        old_position = self.position_shares
        old_entry_price = self.entry_price

        # Step 1: Close short position
        # Cost to buy back all shares
        buyback_cost = round(abs(old_position) * price, 2)

        # Capital released from original short
        proportional_capital = round(abs(old_position) * old_entry_price, 2)

        # Realized P&L
        realized_pnl = round(abs(old_position) * (old_entry_price - price), 2)

        # Return capital + profit
        cash_returned = proportional_capital + realized_pnl
        self.balance = round(self.balance + cash_returned, 2)

        # Step 2: Open long with remaining
        remaining_shares = old_position + shares  # Positive value
        long_value = round(remaining_shares * price, 2)

        # Check affordability for the long position
        if long_value > round(self.balance + 0.10, 2):
            # Can't afford the long after closing short, stay flat
            self.position_shares = 0.0
            self.entry_price = 0.0
            self.capital_used = 0.0
        else:
            # Open the long position: PAY money
            self.balance = round(self.balance - long_value, 2)
            self.position_shares = remaining_shares
            self.entry_price = price
            self.capital_used = long_value

        self.total_trades += 1
        return realized_pnl, True

    # ============================================================================
    # CALCULATIONS - Pure functions
    # ============================================================================

    def _calculate_max_affordable_shares(self, available_capital: float, price: float) -> float:
        """Calculate maximum shares affordable with available capital"""
        return available_capital / price

    def _calculate_realized_pnl(self, old_position: float, shares_to_trade: float, price: float) -> float:
        """Calculate realized P&L when closing a position"""
        if old_position == 0 or np.sign(shares_to_trade) == np.sign(old_position):
            return 0.0  # Not closing

        shares_closing = min(abs(shares_to_trade), abs(old_position))

        if old_position > 0:  # Closing long
            pnl = shares_closing * (price - self.entry_price)
        else:  # Closing short
            pnl = shares_closing * (self.entry_price - price)

        return round(pnl, 2)

    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate current unrealized P&L"""
        if abs(self.position_shares) == 0:
            return 0.0

        if self.position_shares > 0:  # Long
            pnl = self.position_shares * (current_price - self.entry_price)
        else:  # Short
            pnl = abs(self.position_shares) * (self.entry_price - current_price)

        return round(pnl, 2)

    def calculate_portfolio_value(self, current_price: float) -> float:
        """Calculate total portfolio value (balance + position value)"""
        unrealized_pnl = self.calculate_unrealized_pnl(current_price)
        portfolio = self.balance + self.capital_used + unrealized_pnl
        return round(portfolio, 2)

    # ============================================================================
    # ENTRY PRICE MANAGEMENT
    # ============================================================================

    def _update_entry_price(self, old_position: float, shares_to_trade: float, price: float):
        """Update entry price based on position changes"""
        new_position = old_position + shares_to_trade

        if abs(old_position) == 0:
            # Opening new position
            self.entry_price = price
        elif abs(new_position) < 0.001:
            # Closing position completely
            self.entry_price = 0.0
        elif np.sign(new_position) != np.sign(old_position):
            # Position reversal
            self.entry_price = price
        elif np.sign(shares_to_trade) == np.sign(old_position):
            # Adding to existing position - weighted average
            old_value = abs(old_position) * self.entry_price
            new_value = abs(shares_to_trade) * price
            total_shares = abs(old_position) + abs(shares_to_trade)
            if total_shares > 0:
                self.entry_price = (old_value + new_value) / total_shares

    # ============================================================================
    # UTILITIES
    # ============================================================================

    def _round_shares(self, shares: float) -> float:
        """Round shares to quantity precision"""
        rounded = round(shares / self.quantity_precision) * self.quantity_precision
        return max(abs(rounded), self.quantity_precision) * (1 if shares > 0 else -1)

    def _force_liquidate(self, price: float) -> float:
        """Force liquidate position on bankruptcy"""
        self._execute_trade(-self.position_shares, price)
        return 0.0

    def _validate_inputs(self, signal: float, position_size: float):
        """Validate step inputs"""
        if not (-1.0 <= signal <= 1.0):
            raise ValueError(f"Signal must be in [-1, 1], got {signal}")
        if not (0.0 <= position_size <= 1.0):
            raise ValueError(f"Position size must be in [0, 1], got {position_size}")

    # ============================================================================
    # RISK MANAGEMENT
    # ============================================================================

    def is_bankrupt(self, current_price: float) -> bool:
        """Check if portfolio is bankrupt"""
        portfolio_value = self.calculate_portfolio_value(current_price)
        return portfolio_value <= 0

    # ============================================================================
    # TRACKING & REPORTING
    # ============================================================================

    def _record_step(self, step_index: int, price: float, signal: float, position_size: float,
                     step_pnl: float, traded: bool):
        """Record step in history"""
        self.step_history.append({
            'step': step_index,
            'price': price,
            'signal': signal,
            'position_size': position_size,
            'position_shares': self.position_shares,
            'entry_price': self.entry_price,
            'balance': self.balance,
            'capital_used': self.capital_used,
            'unrealized_pnl': self.calculate_unrealized_pnl(price),
            'portfolio_value': self.calculate_portfolio_value(price),
            'step_pnl': step_pnl,
            'traded': traded,
            'total_trades': self.total_trades
        })

    def create_step_state(self, step_index: int, price: float, signal: float, position_size: float,
                          step_pnl: float, traded: bool) -> dict:
        """Record step in history"""
        return {
            'step': step_index,
            'price': price,
            'signal': signal,
            'position_size': position_size,
            'position_shares': self.position_shares,
            'entry_price': self.entry_price,
            'balance': self.balance,
            'capital_used': self.capital_used,
            'unrealized_pnl': self.calculate_unrealized_pnl(price),
            'portfolio_value': self.calculate_portfolio_value(price),
            'step_pnl': step_pnl,
            'traded': traded,
            'total_trades': self.total_trades
        }

    def get_state(self) -> Dict:
        """Get current broker state"""
        if not self.step_history:
            raise ValueError("No steps recorded yet.")
        return self.step_history[-1]
