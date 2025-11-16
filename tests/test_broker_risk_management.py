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
        import pandas as pd
        self.broker = SimpleBroker(initial_balance=10000.0, maker_commission=0.001)

    def test_stop_loss_execution_long(self):
        import pandas as pd
        df = pd.DataFrame({
            'open': [100, 99.5, 98],
            'close': [100, 99.5, 98],
            'high': [101, 100, 99],
            'low': [99, 98.5, 97]
        })
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 1, 0.04, 0.02)
        self.broker.step(0, 1, df.iloc[0], tp_price, sl_price)
        initial_balance = self.broker.current_balance
        self.broker.step(1, 1, df.iloc[1], tp_price, sl_price)
        self.broker.step(2, 1, df.iloc[2], tp_price, sl_price)
        assert self.broker.position_size == 0.0
        assert self.broker.close_reason == 'SL'
        assert self.broker.current_balance < initial_balance

    def test_take_profit_execution_short(self):
        import pandas as pd
        df = pd.DataFrame({
            'open': [100, 99, 97],
            'close': [100, 99, 97],
            'high': [101, 100, 98],
            'low': [99, 98, 96]
        })
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 2, 0.03, 0.03)
        self.broker.step(0, 2, df.iloc[0], tp_price, sl_price)
        initial_balance = self.broker.current_balance
        self.broker.step(1, 2, df.iloc[1], tp_price, sl_price)
        self.broker.step(2, 2, df.iloc[2], tp_price, sl_price)
        assert self.broker.position_size == 0.0
        assert self.broker.close_reason == 'TP'
        assert self.broker.current_balance > initial_balance

    def test_volatility_survival(self):
        import pandas as pd
        df = pd.DataFrame({
            'open': [100, 99.9, 104, 105],
            'close': [100, 99.9, 104, 105],
            'high': [101, 100, 105, 106],
            'low': [99, 98.5, 103, 104]
        })
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 1, 0.05, 0.05)
        self.broker.step(0, 1, df.iloc[0], tp_price, sl_price)
        position_size = self.broker.position_size
        self.broker.step(1, 1, df.iloc[1], tp_price, sl_price)
        assert self.broker.position_size == position_size
        self.broker.step(2, 1, df.iloc[2], tp_price, sl_price)
        self.broker.step(3, 1, df.iloc[3], tp_price, sl_price)
        assert self.broker.position_size == 0.0
        assert self.broker.close_reason == 'TP'

    def test_insufficient_funds_recovery(self):
        import pandas as pd
        small_broker = SimpleBroker(initial_balance=10.0, maker_commission=0.001)
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'close': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97]
        })
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 1, 0.03, 0.03)
        small_broker.step(0, 1, df.iloc[0], tp_price, sl_price)
        tp_price2, sl_price2 = calculate_sl_tp_prices(100.0, 2, 0.03, 0.03)
        small_broker.step(1, 2, df.iloc[1], tp_price2, sl_price2)
        small_broker.step(2, 0, df.iloc[2], tp_price2, sl_price2)
        assert small_broker.is_bankrupt == (small_broker.equity <= small_broker.initial_balance * 0.1)

    def test_multi_period_hold(self):
        import pandas as pd
        df = pd.DataFrame({
            'open': [100, 102, 104, 106, 108, 110],
            'close': [100, 102, 104, 106, 108, 110],
            'high': [101, 103, 105, 107, 109, 111],
            'low': [99, 101, 103, 105, 107, 109]
        })
        tp_price, sl_price = calculate_sl_tp_prices(100.0, 1, 0.50, 0.50)  # Very wide TP/SL
        self.broker.step(0, 1, df.iloc[0], tp_price, sl_price)
        assert self.broker.position_size > 0, 'Position should be opened for multi-period hold test.'
        unrealized_pnls = []
        for i in range(1, 6):
            self.broker.step(i, 1, df.iloc[i], tp_price, sl_price)
            unrealized_pnls.append(self.broker.unrealized_pnl)
        assert all(pnl > 0 for pnl in unrealized_pnls), f"Unrealized PnLs: {unrealized_pnls}"
        assert unrealized_pnls[-1] > unrealized_pnls[0]

    def test_open_position_without_stop_loss(self):
        import pandas as pd
        df = pd.DataFrame({
            'open': [100, 100, 100],
            'close': [100, 100, 100],
            'high': [100, 100, 100],
            'low': [100, 100, 100]
        })
        self.broker.step(0, 1, df.iloc[0], 105.0, None)
        assert self.broker.position_size > 0
        assert self.broker.stop_loss_price is None
        assert self.broker.take_profit_price == 105.0

    def test_open_position_without_take_profit(self):
        import pandas as pd
        df = pd.DataFrame({
            'open': [100, 100, 100],
            'close': [100, 100, 100],
            'high': [100, 100, 100],
            'low': [100, 100, 100]
        })
        self.broker.step(0, 2, df.iloc[1], None, 105.0)
        assert self.broker.position_size < 0
        assert self.broker.take_profit_price is None
        assert self.broker.stop_loss_price == 105.0

    def test_open_position_without_tp_and_sl(self):
        import pandas as pd
        df = pd.DataFrame({
            'open': [100, 100, 100],
            'close': [100, 100, 100],
            'high': [100, 100, 100],
            'low': [100, 100, 100]
        })
        self.broker.step(0, 1, df.iloc[0], None, None)
        assert self.broker.position_size > 0
        assert self.broker.stop_loss_price is None
        assert self.broker.take_profit_price is None

    def test_no_take_profit_trigger_when_none(self):
        import pandas as pd
        df = pd.DataFrame({
            'open': [100, 100, 100],
            'close': [100, 100, 100],
            'high': [100, 100, 100],
            'low': [100, 100, 100]
        })
        self.broker.step(0, 1, df.iloc[0], None, 95.0)
        position_size = self.broker.position_size
        self.broker.step(1, 1, df.iloc[2], None, 95.0)
        assert self.broker.position_size == position_size
        assert self.broker.close_reason != 'TP'

    def test_full_leverage_position_value(self):
        import pandas as pd
        initial_balance = 1000.0
        broker = SimpleBroker(initial_balance=initial_balance, maker_commission=0.001)
        df = pd.DataFrame({
            'open': [50000, 50000, 50000],
            'close': [50000, 50000, 50000],
            'high': [50000, 50000, 50000],
            'low': [50000, 50000, 50000]
        })
        broker.step(0, 1, df.iloc[0], 55000.0, None)
        actual_position_value = abs(broker.position_size) * 50000.0
        assert actual_position_value <= initial_balance * 10.0
