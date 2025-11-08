from environments.simple_broker import SimpleBroker
import pytest


def calculate_sl_tp_prices(entry_price, direction, tp_pct, sl_pct):
    """Helper to calculate SL/TP prices based on percentages."""
    if direction == 1:  # LONG
        tp_price = entry_price * (1 + tp_pct)
        sl_price = entry_price * (1 - sl_pct)
    else:  # SHORT (direction == 2)
        tp_price = entry_price * (1 - tp_pct)
        sl_price = entry_price * (1 + sl_pct)
    return tp_price, sl_price


class TestOvertradingPenalties:
    """Test overtrading penalties and commission impacts"""

    def setup_method(self):
        self.broker = SimpleBroker(initial_balance=10000.0, maker_commission=0.001)

    def test_overtrading_penalty_sequence(self):
        """Test: Model should avoid frequent switching (heavy penalties)"""
        # Test frequent direction reversals which close and reopen positions
        initial_balance = self.broker.current_balance
        initial_commission = self.broker.total_commission

        # Open LONG
        tp_price1, sl_price1 = calculate_sl_tp_prices(100.0, 1, 0.03, 0.03)
        self.broker.step(0, 1, 100.0, 101.0, 99.0, tp_price1, sl_price1)

        # Reverse to SHORT (closes long, opens short - commission on both)
        tp_price2, sl_price2 = calculate_sl_tp_prices(100.0, 2, 0.03, 0.03)
        self.broker.step(1, 2, 100.0, 101.0, 99.0, tp_price2, sl_price2)
        commission_after_first_reversal = self.broker.total_commission

        # Reverse back to LONG (closes short, opens long - more commission)
        tp_price3, sl_price3 = calculate_sl_tp_prices(100.0, 1, 0.03, 0.03)
        self.broker.step(2, 1, 100.0, 101.0, 99.0, tp_price3, sl_price3)
        commission_after_second_reversal = self.broker.total_commission

        # Verify heavy commission penalties from reversals
        total_commissions = self.broker.total_commission
        assert total_commissions > initial_commission
        assert commission_after_second_reversal > commission_after_first_reversal

        # Balance should be significantly reduced due to commissions
        # even with minimal price movement
        balance_reduction = initial_balance - self.broker.current_balance
        assert balance_reduction > 0

    def test_commission_impact_high_frequency(self):
        """Test commission erosion with high-frequency trading"""
        initial_balance = self.broker.current_balance

        # Execute many reversal cycles with minimal price movement
        price = 100.0
        for i in range(10):
            # Open LONG
            tp_price, sl_price = calculate_sl_tp_prices(price, 1, 0.05, 0.01)
            self.broker.step(i * 2, 1, price, price+1, price-1, tp_price, sl_price)

            # Reverse to SHORT (closes long, opens short - 2 commissions)
            tp_price2, sl_price2 = calculate_sl_tp_prices(price, 2, 0.05, 0.01)
            self.broker.step(i * 2 + 1, 2, price, price+1.1, price-0.9, tp_price2, sl_price2)

        # Commission should significantly erode balance
        commission_ratio = self.broker.total_commission / initial_balance
        assert commission_ratio > 0.01  # At least 1% in commissions

        # Balance should be lower due to commissions despite minimal price moves
        assert self.broker.current_balance < initial_balance
