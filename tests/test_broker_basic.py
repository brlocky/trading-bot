from environments.simple_broker import SimpleBroker
import pytest


class TestBrokerBasic:
    """Basic broker functionality tests"""

    def setup_method(self):
        self.broker = SimpleBroker(initial_balance=10000.0, maker_commission=0.001)

    def test_initialization(self):
        assert self.broker.initial_balance == 10000.0
        assert self.broker.current_balance == 10000.0
        assert self.broker.position_size == 0.0

    def test_reset(self):
        self.broker.step(0, 1, 100.0, 101.0, 99.0, 2, 2)
        self.broker.reset()
        assert self.broker.current_balance == 10000.0
        assert self.broker.position_size == 0.0

    def test_open_close_positions(self):
        # Test long position
        self.broker.step(0, 1, 100.0, 101.0, 99.0, 2, 2)
        assert self.broker.direction == 1
        assert self.broker.position_size > 0

        # Test close
        self.broker.step(1, 0, 105.0, 106.0, 104.0, 2, 2)
        assert self.broker.position_size == 0.0

    def test_short_position(self):
        self.broker.step(0, 2, 100.0, 101.0, 99.0, 2, 2)
        assert self.broker.direction == -1
        assert self.broker.position_size < 0
