import math
import numpy as np
from typing import Dict
from copy import deepcopy


class SimpleBroker:
    """
    Enhanced Broker with proper position sizing and risk management
    """

    def __init__(self,
                 initial_balance: float = 100.0,
                 quantity_precision: float = 0.001,
                 maker_commission: float = 0.0002,
                 taker_commission: float = 0.00055,
                 ):

        if quantity_precision <= 0:
            raise ValueError("Quantity precision must be positive")

        if initial_balance <= 0:
            raise ValueError("Initial balance must be positive")

        # Trading parameters
        self.quantity_precision = quantity_precision
        self.maker_commission = maker_commission
        self.taker_commission = taker_commission

        # Initial balance
        self.initial_balance = round(initial_balance, 2)

        # Initialize all variables
        self.reset()

    def reset(self):
        # --- Account Balance Information ---
        self.current_balance = round(self.initial_balance, 2)
        self.equity = self.current_balance

        # --- PnL Metrics ---
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.used_balance = 0.0
        self.total_commission = 0.0
        self.return_since_start = 0.0

        # --- Position Information ---
        self.direction = 0  # -1: short, 0: flat, 1: long
        self.position_size = 0.0
        self.position_value = 0.0
        self.avg_entry_price = 0.0
        self.traded = False

        # --- Risk Management ---
        self.stop_loss_price = None
        self.take_profit_price = None

        # Tracking
        self.current_step = 0
        self.current_price = 0.0
        self.open_trades = 0
        self.closed_trades = 0
        self.win_trades = 0
        self.lost_trades = 0
        self.close_reason = None
        self.is_bankrupt = False
        self.performance = {}

        # Performance tracking
        self.trade_history = []
        self.all_pnls = []
        self.all_returns = []
        self.gross_profit = 0.0
        self.gross_loss = 0.0

        # Drawdown tracking
        self.peak_balance = self.current_balance
        self.running_balance = [self.current_balance]
        self.max_drawdown = 0.0
        self.max_drawdown_value = 0.0

        # History
        self.step_history = []

    def step(self,
             step_index: int,
             signal: int,  # 0=HOLD, 1=LONG, 2=SHORT, 3=CLOSE
             close: float, high: float, low: float,
             tp_price: float | None,
             sl_price: float | None):

        self.current_step = step_index
        self.current_price = close
        self.traded = False

        # Convert signal to target direction
        # 0 = HOLD (keep current position, don't change anything)
        # 1 = LONG
        # 2 = SHORT
        # 3 = CLOSE (explicitly close position)
        if signal == 0:      # HOLD
            target_direction = None  # None means maintain current state
        elif signal == 1:    # LONG
            target_direction = 1
        elif signal == 2:    # SHORT
            target_direction = -1
        elif signal == 3:    # CLOSE
            target_direction = 0  # 0 means go flat (close position)
        else:
            raise ValueError(f"Invalid signal: {signal}. Expected 0, 1, 2, or 3")

        current_direction = np.sign(self.position_size)

        # Check stop loss/take profit first
        if self._check_stop_loss_take_profit(high, low):
            self._update_metrics(close)
            self._record_step(step_index)
            return

        # If signal is HOLD (target_direction=None), maintain current position
        if target_direction is None:
            self._update_metrics(close)
            self._record_step(step_index)
            return

        # If signal is CLOSE (target_direction=0) and we have a position, close it
        if target_direction == 0:
            self._close_position(close, 'Manual Close')
            self._update_metrics(close)
            self._record_step(step_index)
            return

        # If direction changed (reverse position), close and reopen
        if (current_direction != 0 and target_direction in [-1, 1] and target_direction != current_direction):
            if tp_price is None or sl_price is None:
                raise ValueError("TP and SL prices must be provided when changing direction")
            self._close_position(close, 'Direction Change')
            self._open_position(target_direction, close, tp_price, sl_price)  # Pass PRICES
            self._update_metrics(close)
            self._record_step(step_index)
            return

        # Open new position if needed
        if target_direction in [-1, 1] and self.position_size == 0:
            if tp_price is None or sl_price is None:
                raise ValueError("TP and SL prices must be provided when openning a position")
            self._open_position(target_direction, close, tp_price, sl_price)  # Pass PRICES

        self._update_metrics(close)
        self._record_step(step_index)

    def _open_position(self, direction: int, entry_price: float, tp: float, sl: float) -> bool:
        """Open new position with commission"""
        if self.position_size != 0:
            return False

        # Calculate position size
        share_size = self._calculate_share_size(self.current_balance, entry_price, sl) * direction
        if abs(share_size) <= 0:
            print("Insufficient funds to open position.")
            return False

        # Entry is a limit order (maker fee)
        commission = abs(share_size) * entry_price * self.maker_commission
        if commission > self.current_balance:
            print("Insufficient funds to cover commission.")
            return False

        # Update position
        self.direction = direction
        self.position_size = share_size
        self.position_value = abs(share_size) * entry_price
        self.avg_entry_price = entry_price

        # Update account
        self.current_balance -= commission
        self.total_commission += commission
        self.used_balance = round(entry_price * abs(share_size), 2)

        # Set risk management
        self.stop_loss_price = sl
        self.take_profit_price = tp
        self.open_trades += 1
        self.traded = True

        # Calculate actual risk-reward ratio from prices
        risk = abs(entry_price - sl)
        reward_potential = abs(tp - entry_price)
        risk_reward_ratio = reward_potential / risk if risk > 0 else 0

        # Record trade for performance metrics (will be processed in _update_metrics)
        trade_data = {
            'status': 'OPEN',
            'step_open': self.current_step,
            'entry_price': self.avg_entry_price,
            'position_size': self.position_size,
            'position_value': self.position_value,
            'commission': commission,
            'commission_type': 'maker',  # Limit order entry
            'direction': self.direction,
            'tp_price': self.take_profit_price,
            'sl_price': self.stop_loss_price,
            'risk_reward_ratio': risk_reward_ratio,
        }
        self.trade_history.append(trade_data)

        return True

    def _close_position(self, price: float, reason: str) -> float:
        """Close entire position with maker/taker fee based on close reason"""
        if self.position_size == 0:
            return 0.0

        # Calculate PnL
        realized_pnl = self._calculate_unrealized_pnl(price)

        # Determine commission type based on close reason
        if reason == 'TP':
            # Take profit hit = limit order fill = maker fee
            exit_commission = abs(self.position_size) * price * self.maker_commission
            commission_type = 'maker'
        elif reason == 'SL':
            # Stop loss hit = market order = taker fee
            exit_commission = abs(self.position_size) * price * self.taker_commission
            commission_type = 'taker'
        else:
            # Manual close or signal change = limit order fill = maker fee
            exit_commission = abs(self.position_size) * price * self.maker_commission
            commission_type = 'maker'

        true_pnl = realized_pnl - exit_commission

        # Update the last trade with closing information
        if not self.trade_history:
            raise ValueError("No open trade to close")

        last_trade = self.trade_history[-1]
        entry_commission = last_trade.get('commission', 0)

        last_trade.update({
            'status': 'CLOSED',
            'step_close': self.current_step,
            'exit_price': price,
            'commission': entry_commission + exit_commission,
            'entry_commission': entry_commission,
            'exit_commission': exit_commission,
            'exit_commission_type': commission_type,
            'duration': self.current_step - last_trade.get('step_open', 0),
            'pnl': true_pnl,
            'pnl_percent': (true_pnl / self.used_balance) * 100 if self.used_balance else 0,
            'reason': reason
        })

        # Update account (metrics will be updated in _update_metrics)
        self.current_balance += realized_pnl - exit_commission
        self.realized_pnl += realized_pnl
        self.total_commission += exit_commission

        # Reset position
        self.used_balance = 0.0
        self.position_size = 0.0
        self.position_value = 0.0
        self.avg_entry_price = 0.0
        self.stop_loss_price = None
        self.take_profit_price = None
        self.direction = 0

        self.closed_trades += 1
        self.traded = True
        self.close_reason = reason

        return true_pnl

    def _calculate_share_size(self, cash: float, entry_price: float, stop_loss: float, risk_percentage: float = 0.01) -> float:
        """Calculate position size risking 1% of account balance"""

        # print(f"DEBUG: cash={cash}, entry_price={entry_price}, stop_loss={stop_loss}")

        if entry_price <= 0:
            raise ValueError("Entry price must be positive")

        # Calculate the risk per share (distance to stop-loss)
        share_risk = abs(entry_price - stop_loss)
        if share_risk == 0:
            return 0.0

        # Risk 1% of current balance
        risk_per_trade = risk_percentage * cash

        # Calculate maximum shares based on risk
        max_shares_by_risk = risk_per_trade / share_risk

        # print(f"DEBUG: share_risk={share_risk}, risk_per_trade={risk_per_trade}, max_shares_by_risk={max_shares_by_risk}")

        if max_shares_by_risk <= 0:
            return 0.0

        # Floor to nearest precision
        floored_size = math.floor(max_shares_by_risk / self.quantity_precision) * self.quantity_precision

        # Ensure minimum viable position size
        min_position_value = entry_price * self.quantity_precision
        if floored_size * entry_price < min_position_value:
            return 0.0

        return max(0.0, round(floored_size, 10))

    def _update_metrics(self, price: float):
        """Update all metrics including performance ratios and drawdown"""
        # Basic metrics
        self.unrealized_pnl = self._calculate_unrealized_pnl(price)
        self.equity = round(self.current_balance + self.unrealized_pnl, 2)
        self.return_since_start = (self.equity - self.initial_balance) / self.initial_balance
        # More lenient bankruptcy threshold - only bankrupt if < 5% of initial balance
        self.is_bankrupt = self.equity <= self.initial_balance * 0.05

        # Update drawdown tracking
        self.running_balance.append(self.current_balance)
        self._update_drawdown()

        # Process recent trades for performance metrics
        self._update_performance_metrics()

        self.performance = self._calculate_performance_ratios()

    def _update_performance_metrics(self):
        """Update performance metrics from trade history"""
        # This can be called less frequently if needed for performance
        if not self.trade_history:
            return

        # Update win/loss counts and gross P&L
        recent_trades = [t for t in self.trade_history
                         if t.get('processed', False) is False and t['status'] == 'CLOSED']

        for trade in recent_trades:
            pnl = trade['pnl']
            if pnl > 0:
                self.win_trades += 1
                self.gross_profit += pnl
            else:
                self.lost_trades += 1
                self.gross_loss += abs(pnl)

            self.all_pnls.append(pnl)
            self.all_returns.append(trade['pnl_percent'] / 100)
            trade['processed'] = True

    def _update_drawdown(self):
        """Update drawdown metrics"""
        if not self.running_balance:
            return

        current_balance = self.running_balance[-1]

        # Update peak balance
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance

        # Calculate current drawdown
        current_drawdown = (self.peak_balance - current_balance) / self.peak_balance if self.peak_balance > 0 else 0

        # Update max drawdown
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
            self.max_drawdown_value = self.peak_balance - current_balance

    def _calculate_performance_ratios(self):
        """Calculate performance ratios - now uses pre-computed metrics"""
        if self.closed_trades == 0:
            return {
                'win_rate': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'expectancy': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'max_drawdown_value': 0,
                'calmar_ratio': 0,
                'total_return': 0,
                'total_trades': 0,
                'total_pnl': 0,
                'total_commission': 0,
                'final_balance': self.current_balance
            }

        win_rate = self.win_trades / self.closed_trades
        profit_factor = self.gross_profit / self.gross_loss if self.gross_loss > 0 else 0

        avg_win = self.gross_profit / self.win_trades if self.win_trades > 0 else 0
        avg_loss = self.gross_loss / self.lost_trades if self.lost_trades > 0 else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # Sharpe Ratio
        sharpe_ratio = 0
        if self.all_returns:
            avg_return = np.mean(self.all_returns)
            std_returns = np.std(self.all_returns)
            sharpe_ratio = avg_return / std_returns if std_returns > 0 else 0

        # Calmar Ratio
        total_return = (self.current_balance - self.initial_balance) / self.initial_balance
        calmar_ratio = total_return / self.max_drawdown if self.max_drawdown > 0 else 0

        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'expectancy': expectancy,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_value': self.max_drawdown_value,
            'calmar_ratio': calmar_ratio,
            'total_return': total_return,
            'total_trades': self.closed_trades,
            'total_pnl': sum(self.all_pnls),
            'total_commission': self.total_commission,
            'final_balance': self.current_balance
        }

    # -------------------
    # UTILITIES - IMPROVED
    # -------------------

    def _check_stop_loss_take_profit(self, high: float, low: float) -> bool:
        if low > high:
            raise ValueError("Low price cannot be higher than high price")

        """Check and execute stop loss/take profit."""
        if self.position_size == 0:
            return False

        is_long = self.position_size > 0

        if is_long:
            if self.stop_loss_price and low <= self.stop_loss_price:
                self._close_position(self.stop_loss_price, 'SL')
                return True
            elif self.take_profit_price and high >= self.take_profit_price:
                self._close_position(self.take_profit_price, 'TP')
                return True
        else:
            if self.stop_loss_price and high >= self.stop_loss_price:
                self._close_position(self.stop_loss_price, 'SL')
                return True
            elif self.take_profit_price and low <= self.take_profit_price:
                self._close_position(self.take_profit_price, 'TP')
                return True

        return False

    def _calculate_unrealized_pnl(self, price: float) -> float:
        if self.position_size == 0:
            return 0.0
        if self.position_size > 0:
            return self.position_size * (price - self.avg_entry_price)
        else:
            return abs(self.position_size) * (self.avg_entry_price - price)

    def _record_step(self, step_index: int):
        self.step_history.append({
            'step': step_index,
            'current_price': self.current_price,
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'equity': self.equity,

            'position_size': self.position_size,
            'position_value': self.position_value,
            'entry_price': self.avg_entry_price,

            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_commission': self.total_commission,
            'used_balance': self.used_balance,

            'traded': self.traded,
            'open_trades': self.open_trades,
            'closed_trades': self.closed_trades,
            'stop_loss_price': self.stop_loss_price,
            'take_profit_price': self.take_profit_price,
            'performance': self.performance,
            'trades': deepcopy(self.trade_history),  # Deep copy to prevent state mutation
        })

    def get_state(self) -> Dict:
        """Get current broker state"""
        if not self.step_history:
            return {}
        return self.step_history[-1]
