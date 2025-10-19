"""
Simple Unit Tests for TradingBroker
Tests each action handler method independently
"""

from environments.broker import TradingBroker
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestBrokerInitialization:
    """Test broker initialization"""


class TestInputValidation:
    """Test input validation"""

    def test_valid_inputs(self):
        broker = TradingBroker()
        broker._validate_inputs(0.0, 0.5)
        broker._validate_inputs(1.0, 1.0)
        broker._validate_inputs(-1.0, 0.0)

    def test_invalid_signal(self):
        broker = TradingBroker()
        with pytest.raises(ValueError, match="Signal must be in"):
            broker._validate_inputs(1.5, 0.5)


class TestPortfolioValue:
    """Test calculate_portfolio_value method"""

    def test_portfolio_value_no_position(self):
        broker = TradingBroker(initial_balance=10000.0)
        assert broker.calculate_portfolio_value(100.0) == 10000.0
