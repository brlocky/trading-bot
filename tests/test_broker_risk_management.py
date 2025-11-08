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
