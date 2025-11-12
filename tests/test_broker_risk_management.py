import math
from environments.simple_broker import SimpleBroker


def calculate_sl_tp_prices(entry_price, direction, tp_pct, sl_pct):
    """Helper to calculate SL/TP prices based on percentages."""
    if direction == 1:  # LONG
        tp_price = entry_price * (1 + tp_pct)
        sl_price = entry_price * (1 - sl_pct)
    else:  # SHORT (direction == 2)
        tp_price = entry_price * (1 - tp_pct)
        sl_price = entry_price * (1 + sl_pct)
    return tp_price, sl_price


class TestRiskManagement:
    """Test risk management features (SL/TP, position sizing, etc.)"""

    def setup_method(self):
        self.broker = SimpleBroker(initial_balance=10000.0, maker_commission=0.001)

    def test_stop_loss_execution_long(self):
        """Test proper SL execution on adverse move for long position"""
        # Open LONG with tight 2% SL, wide 4% TP
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 1, 0.04, 0.02)
        self.broker.step(0, 1, 100.0, 101.0, 99.0, tp_price, sl_price)
        sl_price_actual = self.broker.stop_loss_price
        initial_balance = self.broker.current_balance

        # Keep LONG (price starts dropping)
        self.broker.step(1, 1, 99.5, 100.5, 99.0, tp_price, sl_price)

        # Keep LONG until SL hits automatically
        self.broker.step(2, 1, sl_price_actual - 0.5, sl_price_actual + 0.5, sl_price_actual - 1.0, tp_price, sl_price)

        assert self.broker.position_size == 0.0
        assert self.broker.close_reason == 'SL'
        assert self.broker.current_balance < initial_balance  # Should have loss

    def test_take_profit_execution_short(self):
        """Test proper TP execution for short position"""
        # Open SHORT
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 2, 0.03, 0.03)
        self.broker.step(0, 2, 100.0, 101.0, 99.0, tp_price, sl_price)
        tp_price_actual = self.broker.take_profit_price
        initial_balance = self.broker.current_balance

        # Price moves toward TP
        self.broker.step(1, 2, tp_price_actual + 0.5, tp_price_actual + 1.0, tp_price_actual + 0.2, tp_price, sl_price)

        # TP hits automatically
        self.broker.step(2, 2, tp_price_actual - 0.5, tp_price_actual, tp_price_actual - 1.0, tp_price, sl_price)

        assert self.broker.position_size == 0.0
        assert self.broker.close_reason == 'TP'
        assert self.broker.current_balance > initial_balance  # Should have profit

    def test_volatility_survival(self):
        """Test position survives whipsaw without stopping out"""
        # Open LONG with wide 5% SL/TP
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 1, 0.05, 0.05)
        self.broker.step(0, 1, 100.0, 101.0, 99.0, tp_price, sl_price)
        sl_price_actual = self.broker.stop_loss_price
        tp_price_actual = self.broker.take_profit_price
        position_size = self.broker.position_size

        # Keep LONG (price dips near SL but recovers)
        self.broker.step(1, 1, sl_price_actual + 0.1, sl_price_actual + 1.0, sl_price_actual + 0.01, tp_price, sl_price)
        assert self.broker.position_size == position_size  # Survived near-SL

        # Keep LONG (price rallies to TP)
        self.broker.step(2, 1, tp_price_actual - 0.1, tp_price_actual - 0.01, tp_price_actual - 1.0, tp_price, sl_price)

        # Should hit TP
        self.broker.step(3, 1, tp_price_actual, tp_price_actual + 0.5, tp_price_actual - 0.5, tp_price, sl_price)
        assert self.broker.position_size == 0.0
        assert self.broker.close_reason == 'TP'

    def test_insufficient_funds_recovery(self):
        """Test broker handles insufficient funds gracefully"""
        small_broker = SimpleBroker(initial_balance=10.0, maker_commission=0.001)

        # Try to open large position (may fail due to funds)
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 1, 0.03, 0.03)
        small_broker.step(0, 1, 100.0, 101.0, 99.0, tp_price, sl_price)

        # Try to switch direction (may also fail)
        tp_price2, sl_price2 = calculate_sl_tp_prices(100.0, 2, 0.03, 0.03)
        small_broker.step(1, 2, 100.0, 101.0, 99.0, tp_price2, sl_price2)

        # Reset to flat
        small_broker.step(2, 0, 100.0, 101.0, 99.0, tp_price2, sl_price2)

        # Should not crash and should handle gracefully
        assert small_broker.is_bankrupt == (small_broker.equity <= small_broker.initial_balance * 0.1)

    def test_multi_period_hold(self):
        """Test long-term position holding with proper PnL tracking"""
        # Open LONG
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 1, 0.05, 0.04)
        self.broker.step(0, 1, 100.0, 101.0, 99.0, tp_price, sl_price)

        # Hold for multiple periods
        prices = [102.0, 102.1, 102.2, 102.3, 102.4]
        unrealized_pnls = []

        tp_price_hold, sl_price_hold = calculate_sl_tp_prices(100.0, 1, 0.04, 0.03)
        for i, price in enumerate(prices, 1):
            self.broker.step(i, 1, price, price+1, price-1, tp_price_hold, sl_price_hold)
            unrealized_pnls.append(self.broker.unrealized_pnl)

        # Unrealized PnL should track price movements
        assert all(pnl != 0 for pnl in unrealized_pnls)
        assert unrealized_pnls[-1] > unrealized_pnls[0]  # Overall profit

        # Manual close or TP/SL
        self.broker.step(len(prices)+1, 0, 112.0, 113.0, 111.0, tp_price_hold, sl_price_hold)
        assert self.broker.position_size == 0.0
        assert self.broker.realized_pnl > 0

    def test_calculate_share_size_risk_based(self):
        """Test _calculate_share_size respects 1% risk rule"""
        # Test normal risk-based sizing
        cash = 10000.0
        entry_price = 50000.0
        stop_loss = 49000.0  # 2% away from entry

        share_size = self.broker._calculate_share_size(cash, entry_price, stop_loss, risk_percentage=0.01)

        # Risk per share = |50000 - 49000| = 1000
        # Risk amount = 10000 * 0.01 = 100
        # Expected shares = 100 / 1000 = 0.1
        expected_shares = 0.1

        assert abs(share_size - expected_shares) < 0.001, f"Expected {expected_shares}, got {share_size}"

        # Verify position value is within reasonable bounds
        position_value = share_size * entry_price
        assert position_value <= cash * 10.0, "Position value exceeds 10x leverage"

    def test_calculate_share_size_leverage_limit(self):
        """Test _calculate_share_size enforces 10x leverage limit"""
        # Test with very tight stop loss (would exceed 10x without limit)
        cash = 1000.0
        entry_price = 50000.0
        stop_loss = 49950.0  # Only 0.1% away - very tight!

        share_size = self.broker._calculate_share_size(cash, entry_price, stop_loss, risk_percentage=0.01)

        # Risk per share = |50000 - 49950| = 50
        # Risk amount = 1000 * 0.01 = 10
        # Risk-based shares = 10 / 50 = 0.2 (position value = 10,000)
        # But max leverage = 1000 * 10 = 10,000
        # Max shares by leverage = 10000 / 50000 = 0.2
        # Should take minimum = 0.2

        position_value = share_size * entry_price
        max_position_value = cash * 10.0

        assert position_value <= max_position_value, f"Position {position_value} exceeds 10x leverage {max_position_value}"
        assert share_size > 0, "Share size should not be zero"

    def test_calculate_share_size_extreme_tight_stop(self):
        """Test leverage limit kicks in with extremely tight stop loss"""
        cash = 100.0
        entry_price = 50000.0
        stop_loss = 49990.0  # Only 0.02% away - extremely tight!

        share_size = self.broker._calculate_share_size(cash, entry_price, stop_loss, risk_percentage=0.01)

        # Risk per share = 10
        # Risk amount = 100 * 0.01 = 1
        # Risk-based shares = 1 / 10 = 0.1 (position value = 5,000)
        # Max leverage shares = (100 * 10) / 50000 = 0.02 (position value = 1,000)
        # Should be limited to 0.02

        position_value = share_size * entry_price
        max_position_value = cash * 10.0

        assert position_value <= max_position_value, f"Position {position_value} exceeds 10x limit {max_position_value}"

        # Verify it's actually the leverage limit, not risk limit
        risk_based_shares = (cash * 0.01) / abs(entry_price - stop_loss)
        leverage_limited_shares = (cash * 10.0) / entry_price

        assert share_size <= leverage_limited_shares, "Should be leverage-limited"
        assert share_size < risk_based_shares, "Leverage limit should be lower than risk-based"

    def test_max_position_size_enforcement_long(self):
        """Test 10x leverage limit is enforced when opening LONG position"""
        # Use small balance to make leverage limit hit easily
        small_broker = SimpleBroker(initial_balance=100.0, maker_commission=0.001)

        # Try to open position with very tight stop (would want large position)
        entry_price = 50000.0
        stop_loss = 49990.0  # Extremely tight 0.02% stop
        tp_price = 51000.0

        small_broker.step(0, 1, entry_price, entry_price+100, entry_price-100, tp_price, stop_loss)

        # Check position value doesn't exceed 10x balance
        if small_broker.position_size > 0:
            position_value = abs(small_broker.position_size) * entry_price
            max_allowed = small_broker.initial_balance * 10.0

            assert position_value <= max_allowed, f"Position {position_value} exceeds 10x leverage {max_allowed}"

    def test_max_position_size_enforcement_short(self):
        """Test 10x leverage limit is enforced when opening SHORT position"""
        small_broker = SimpleBroker(initial_balance=100.0, maker_commission=0.001)

        # Try to open SHORT with very tight stop
        entry_price = 50000.0
        stop_loss = 50010.0  # Extremely tight stop above entry
        tp_price = 49000.0

        small_broker.step(0, 2, entry_price, entry_price+100, entry_price-100, tp_price, stop_loss)

        # Check position value doesn't exceed 10x balance
        if small_broker.position_size != 0:
            position_value = abs(small_broker.position_size) * entry_price
            max_allowed = small_broker.initial_balance * 10.0

            assert position_value <= max_allowed, f"Position {position_value} exceeds 10x leverage {max_allowed}"

    def test_calculate_share_size_zero_stop_distance(self):
        """Test _calculate_share_size handles zero stop distance gracefully"""
        cash = 10000.0
        entry_price = 50000.0
        stop_loss = 50000.0  # Same as entry (invalid)

        share_size = self.broker._calculate_share_size(cash, entry_price, stop_loss)

        # Should return 0 when stop distance is 0
        assert share_size == 0.0, "Should return 0 for zero stop distance"

    def test_calculate_share_size_no_stop_loss(self):
        """Test _calculate_share_size uses full leverage when stop_loss is None"""
        cash = 10000.0
        entry_price = 50000.0
        stop_loss = None  # No stop loss

        share_size = self.broker._calculate_share_size(cash, entry_price, stop_loss)

        # Should use max 10x leverage
        max_position_value = cash * 10.0
        expected_shares = max_position_value / entry_price
        # Account for precision normalization
        expected_normalized = math.floor(expected_shares / self.broker.quantity_precision) * self.broker.quantity_precision

        assert abs(share_size - expected_normalized) < 0.0001, f"Expected {expected_normalized}, got {share_size}"

        # Verify it's much larger than typical risk-based position
        risk_based_size = self.broker._calculate_share_size(cash, entry_price, 49000.0, risk_percentage=0.01)
        assert share_size > risk_based_size, "Full leverage should be larger than risk-based sizing"

    def test_open_position_without_stop_loss(self):
        """Test opening a position without stop loss (SL=None)"""
        entry_price = 100.0
        initial_balance = self.broker.current_balance

        # Open LONG without SL, only TP
        self.broker.step(0, 1, entry_price, entry_price+1, entry_price-1, tp_price=105.0, sl_price=None)

        assert self.broker.position_size > 0, "Position should be opened"
        assert self.broker.stop_loss_price is None, "Stop loss should be None"
        assert self.broker.take_profit_price == 105.0, "Take profit should be set"

        # Calculate expected position size (10x leverage)
        max_position_value = initial_balance * 10.0
        expected_shares = max_position_value / entry_price
        expected_normalized = math.floor(expected_shares / self.broker.quantity_precision) * self.broker.quantity_precision

        # Allow small difference due to commission
        assert abs(self.broker.position_size - expected_normalized) < 0.01, \
            f"Position size {self.broker.position_size} should be close to full leverage {expected_normalized}"

    def test_open_position_without_take_profit(self):
        """Test opening a position without take profit (TP=None)"""
        entry_price = 100.0

        # Open SHORT without TP, only SL
        self.broker.step(0, 2, entry_price, entry_price+1, entry_price-1, tp_price=None, sl_price=105.0)

        assert self.broker.position_size < 0, "Short position should be opened"
        assert self.broker.take_profit_price is None, "Take profit should be None"
        assert self.broker.stop_loss_price == 105.0, "Stop loss should be set"

    def test_open_position_without_tp_and_sl(self):
        """Test opening a position without both TP and SL"""
        entry_price = 100.0
        initial_balance = self.broker.current_balance

        # Open LONG without TP or SL
        self.broker.step(0, 1, entry_price, entry_price+1, entry_price-1, tp_price=None, sl_price=None)

        assert self.broker.position_size > 0, "Position should be opened"
        assert self.broker.stop_loss_price is None, "Stop loss should be None"
        assert self.broker.take_profit_price is None, "Take profit should be None"

        # Should use full leverage
        max_position_value = initial_balance * 10.0
        expected_shares = max_position_value / entry_price
        expected_normalized = math.floor(expected_shares / self.broker.quantity_precision) * self.broker.quantity_precision

        assert abs(self.broker.position_size - expected_normalized) < 0.01, \
            "Position size should use full leverage"

    def test_no_stop_loss_trigger_when_none(self):
        """Test that SL doesn't trigger when it's None"""
        entry_price = 100.0

        # Open LONG without SL
        self.broker.step(0, 1, entry_price, entry_price+1, entry_price-1, tp_price=110.0, sl_price=None)
        position_size = self.broker.position_size

        # Price drops significantly (would normally hit SL)
        self.broker.step(1, 1, 90.0, 92.0, 88.0, tp_price=110.0, sl_price=None)

        assert self.broker.position_size == position_size, "Position should remain open (no SL to hit)"
        assert self.broker.close_reason != 'SL', "Should not close via stop loss"

    def test_no_take_profit_trigger_when_none(self):
        """Test that TP doesn't trigger when it's None"""
        entry_price = 100.0

        # Open LONG without TP
        self.broker.step(0, 1, entry_price, entry_price+1, entry_price-1, tp_price=None, sl_price=95.0)
        position_size = self.broker.position_size

        # Price rises significantly (would normally hit TP)
        self.broker.step(1, 1, 110.0, 112.0, 108.0, tp_price=None, sl_price=95.0)

        assert self.broker.position_size == position_size, "Position should remain open (no TP to hit)"
        assert self.broker.close_reason != 'TP', "Should not close via take profit"

    def test_manual_close_without_tp_sl(self):
        """Test manually closing a position that has no TP/SL"""
        entry_price = 100.0

        # Open LONG without TP or SL
        self.broker.step(0, 1, entry_price, entry_price+1, entry_price-1, tp_price=None, sl_price=None)
        assert self.broker.position_size > 0

        # Hold for a few steps
        self.broker.step(1, 1, 105.0, 106.0, 104.0, tp_price=None, sl_price=None)
        assert self.broker.position_size > 0

        # Manually close via signal=3
        self.broker.step(2, 3, 105.0, 106.0, 104.0, tp_price=None, sl_price=None)

        assert self.broker.position_size == 0, "Position should be closed"
        assert self.broker.close_reason == 'Manual Close', "Should close manually"
        assert self.broker.realized_pnl > 0, "Should have profit from price increase"

    def test_full_leverage_position_value(self):
        """Test that position opened with no SL uses correct balance"""
        initial_balance = 1000.0
        broker = SimpleBroker(initial_balance=initial_balance, maker_commission=0.001)
        entry_price = 50000.0

        # Open position without SL (should use full leverage)
        broker.step(0, 1, entry_price, entry_price+100, entry_price-100, tp_price=55000.0, sl_price=None)

        # Calculate expected values
        actual_position_value = abs(broker.position_size) * entry_price

        # Should be close to 10x initial balance (minus small commission)
        assert actual_position_value >= initial_balance * 9.0, \
            f"Position value {actual_position_value} should be close to 10x balance {initial_balance * 10}"
        assert actual_position_value <= initial_balance * 10.0, \
            f"Position value {actual_position_value} should not exceed 10x leverage"
