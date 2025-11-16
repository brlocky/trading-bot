import pandas as pd
from environments.simple_broker import SimpleBroker


class TestBrokerBasic:
    """Basic broker functionality tests"""

    def setup_method(self):
        self.broker = SimpleBroker(initial_balance=10000.0, maker_commission=0.001)
        data = {
            'open': [100, 101, 102],
            'close': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97]
        }
        self.df = pd.DataFrame(data)

    def test_initialization(self):
        assert self.broker.initial_balance == 10000.0
        assert self.broker.current_balance == 10000.0
        assert self.broker.position_size == 0.0

    def test_reset(self):
        self.broker.reset()
        assert self.broker.current_balance == 10000.0
        assert self.broker.position_size == 0.0

    def test_open_positions(self):
        # Test long position
        self.broker.step(0, 1, self.df.iloc[0], None, None)
        assert self.broker.direction == 1
        assert self.broker.position_size > 0

    def test_short_position(self):
        self.broker.step(0, 2, self.df.iloc[0], None, None)
        assert self.broker.direction == -1
        assert self.broker.position_size < 0
