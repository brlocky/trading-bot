"""
Functional Tests for TradingBroker
Tests step() method with realistic trading patterns
Validates all 4 return values: (step_pnl_pct,trade_occurred, is_bankrupt)
"""

from environments.broker import TradingBroker
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def test_buy_hold_sell_pattern():
    """Test: Buy -> Hold -> Sell pattern"""
    broker = TradingBroker(initial_balance=10000.0)
    price = 100.0

    # Step 1: Buy 50% long
    pnl_pct, traded, bankrupt = broker.step(1.0, 1, price, step_index=0)
    assert traded is True
    assert bankrupt is False
    assert broker.balance == 0.0
    assert broker.capital_used == 10000.0
    assert broker.position_shares == 100.0

    # Step 2: Hold (price unchanged)
    pnl_pct,  traded, bankrupt = broker.step(0.0, 1.0, price, step_index=1)
    assert traded is False
    assert bankrupt is False
    assert broker.balance == 0.0
    assert broker.capital_used == 10000.0
    assert broker.position_shares == 100.0
    assert broker.capital_used == 10000.0

    # Step 3: Sell (close position)
    pnl_pct, traded, bankrupt = broker.step(0.0, 0.0, price, step_index=2)
    assert traded is True
    assert bankrupt is False
    assert broker.balance == 10000.0
    assert broker.capital_used == 0.0
    assert broker.position_shares == 0.0
    assert broker.capital_used == 0.0


def test_buy_buy_pattern():
    """Test: Buy 30% -> Buy 60% (increase position)"""
    broker = TradingBroker(initial_balance=10000.0)
    price = 100.0

    # Step 1: Buy 30%
    pnl_pct, traded, bankrupt = broker.step(1.0, 0.3, price, step_index=0)
    assert traded is True
    assert broker.position_shares == 30  # Position increased

    pos_after_30 = broker.position_shares

    # Step 2: Buy 60% (increase)
    pnl_pct,  traded, bankrupt = broker.step(1.0, 0.6, price, step_index=1)
    assert traded is True
    assert broker.position_shares == pos_after_30 * 0.6  # Position increased


def test_sell_short_pattern():
    """Test: Sell short 50% -> Hold @ lower price -> Cover (should profit)"""
    broker = TradingBroker(initial_balance=20000.0)
    price_initial = 100.0
    price_lower = 50.0  # Price drops - good for shorts!

    # Step 1: Sell short 50%
    pnl, traded, bankrupt = broker.step(-1.0, 0.5, price_initial, step_index=0)
    assert traded is True
    assert broker.position_shares == -100.0  # No commission, exact shares
    assert broker.capital_used == 10000.0
    assert broker.balance == 10000.0
    assert pnl == 0.0  # No commission, no PnL yet

    # Step 2: Hold at lower price (price dropped 50% - we should have profit)
    portfolio_before_hold = broker.calculate_portfolio_value(price_initial)
    assert portfolio_before_hold == 20000.0  # No change yet
    pnl, traded, bankrupt = broker.step(0.0, 1.0, price_lower, step_index=1)
    assert traded is False
    assert pnl == 5000  # Gained $5k on $20k initial = 25%
    assert broker.position_shares == -100.0
    assert broker.capital_used == 10000.0
    assert broker.balance == 10000.0

    # CRITICAL: Portfolio should show profit since price dropped
    unrealized_pnl = broker.calculate_unrealized_pnl(price_lower)
    assert unrealized_pnl == 5000.0, f"Expected $5,000 profit, got ${unrealized_pnl:.2f}"

    # Step 3: Cover at lower price (close short) - lock in profit
    pnl, traded, bankrupt = broker.step(0.0, 0.0, price_lower, step_index=2)
    assert traded is True
    assert broker.position_shares == 0.0
    assert broker.capital_used == 0.0
    assert broker.balance == 25000.0  # $10k free + $10k capital + $5k pnl = $25k


def test_sell_short_pattern_with_loss():
    """Test: Sell short 50% -> Hold @ higher price -> Cover (should incur loss)"""
    broker = TradingBroker(initial_balance=20000.0)
    price_initial = 100.0
    price_higher = 150.0  # Price rises - bad for shorts!

    # Step 1: Sell short 50%
    pnl, traded, bankrupt = broker.step(-1.0, 0.5, price_initial, step_index=0)
    assert traded is True
    assert broker.position_shares == -100.0  # No commission, exact shares
    assert broker.capital_used == 10000.0
    assert broker.balance == 10000.0
    assert pnl == 0.0  # No commission, no PnL yet

    # Step 2: Hold at lower price (price dropped 50% - we should have profit)
    portfolio_before_hold = broker.calculate_portfolio_value(price_initial)
    assert portfolio_before_hold == 20000.0  # No change yet
    pnl, traded, bankrupt = broker.step(0.0, 1.0, price_higher, step_index=1)
    assert traded is False
    assert pnl == -5000  # Gained $5k on $20k initial = 25%
    assert broker.position_shares == -100.0
    assert broker.capital_used == 10000.0
    assert broker.balance == 10000.0

    # CRITICAL: Portfolio should show profit since price dropped
    unrealized_pnl = broker.calculate_unrealized_pnl(price_higher)
    assert unrealized_pnl == -5000.0, f"Expected -$5,000 loss, got ${unrealized_pnl:.2f}"

    # Step 3: Cover at lower price (close short) - lock in profit
    pnl, traded, bankrupt = broker.step(0.0, 0.0, price_higher, step_index=2)
    assert traded is True
    assert broker.position_shares == 0.0
    assert broker.capital_used == 0.0
    assert broker.balance == 15000.0  # $10k free + $10k capital - $5k pnl = $15k


def test_long_to_short_reversal():
    """Test: Long 50% @ $100 -> Price 4x to $400 -> Reverse to Short 50% (should double portfolio)"""
    broker = TradingBroker(initial_balance=10000.0)
    price_initial = 100.0
    price_higher = 400.0  # 400% increase (4x)

    # Step 1: Go long 50% @ $100
    # Spend $5,000 to buy 50 shares @ $100
    pnl_pct, traded, bankrupt = broker.step(1.0, 0.5, price_initial, step_index=0)
    assert traded is True
    assert broker.position_shares == 50.0
    assert broker.balance == 5000.0
    assert broker.capital_used == 5000.0

    # Step 2: Price rises to $400 (4x)
    # Portfolio = $5,000 (free) + $20,000 (50 shares @ $400) = $25,000
    portfolio_before_reverse = broker.calculate_portfolio_value(price_higher)
    unrealized_profit_from_long = broker.calculate_unrealized_pnl(price_higher)

    print(f"\nDEBUG test_long_to_short_reversal:")
    print(f"  Long position: 50 shares @ $100, now worth $400")
    print(f"  Unrealized profit: ${unrealized_profit_from_long:.2f}")
    print(f"  Portfolio before reverse: ${portfolio_before_reverse:.2f}")

    assert unrealized_profit_from_long == 15000.0, f"Expected $15,000 profit, got ${unrealized_profit_from_long:.2f}"
    assert portfolio_before_reverse == 25000.0, f"Expected $25,000 portfolio, got ${portfolio_before_reverse:.2f}"

    # Step 3: Reverse to short 50% @ $400
    # Close long: Sell 50 shares @ $400 = receive $20,000
    # Balance = $5,000 + $20,000 = $25,000
    # Open short 50%: Use $12,500 for short = 31.25 shares @ $400
    # Expected result: balance=$12,500, capital_used=$12,500, position=-31.25
    # Portfolio = $12,500 + $12,500 = $25,000 (doubled from initial $10k!)

    pnl_pct, traded, bankrupt = broker.step(-1.0, 0.5, price_higher, step_index=1)
    assert traded is True
    assert broker.position_shares < 0

    print(f"  After reverse:")
    print(f"    Short position: {broker.position_shares:.3f} shares @ ${price_higher}")
    print(f"    Balance: ${broker.balance:.2f}")
    print(f"    Capital used: ${broker.capital_used:.2f}")

    actual_portfolio = broker.calculate_portfolio_value(price_higher)
    print(f"    Portfolio: ${actual_portfolio:.2f}")

    # Check the math
    assert broker.position_shares == pytest.approx(-31.25, rel=0.01), "Should short 31.25 shares"
    assert broker.balance == pytest.approx(12500.0, abs=1.0), "Balance should be $12,500"
    assert broker.capital_used == pytest.approx(12500.0, abs=1.0), "Capital used should be $12,500"
    assert actual_portfolio == pytest.approx(25000.0, abs=1.0), "Portfolio should stay at $25,000"


def test_reduce_position_pattern():
    """Test: Buy 100% -> Reduce by 30% -> Reduce by 50% more"""
    broker = TradingBroker(initial_balance=10000.0)
    price = 100.0

    # Step 1: Buy 100%
    pnl_pct,  traded, bankrupt = broker.step(1.0, 1.0, price, step_index=0)
    assert traded is True
    initial_shares = broker.position_shares

    # Step 2: Reduce by 30%
    pnl_pct,  traded, bankrupt = broker.step(0.0, 0.3, price, step_index=1)
    assert traded is True
    after_30_reduce = broker.position_shares
    assert after_30_reduce < initial_shares
    assert after_30_reduce == pytest.approx(initial_shares * 0.3, rel=0.01)

    # Step 3: Reduce by 50% more
    pnl_pct,  traded, bankrupt = broker.step(0.0, 0.5, price, step_index=2)
    assert traded is True
    assert broker.position_shares == pytest.approx(after_30_reduce * 0.5, rel=0.01)


def test_price_movement_pnl():
    """Test: Buy -> Price up -> Price down - validate P&L"""
    broker = TradingBroker(initial_balance=10000.0)
    initial_price = 100.0

    # Step 1: Buy 50%
    pnl_pct,  traded, bankrupt = broker.step(1.0, 0.5, initial_price, step_index=0)
    assert traded is True

    # Step 2: Price goes up 10% - should have positive P&L
    new_price = 110.0
    pnl_pct,  traded, bankrupt = broker.step(0.0, 1.0, new_price, step_index=1)
    assert traded is False  # Just holding
    assert pnl_pct > 0  # Profit from price increase

    # Step 3: Price goes back down - should have negative P&L this step
    pnl_pct,  traded, bankrupt = broker.step(0.0, 1.0, initial_price, step_index=2)
    assert traded is False
    assert pnl_pct < 0  # Loss from price decrease


def test_bankruptcy_scenario():
    """Test: Large loss leading to bankruptcy"""
    broker = TradingBroker(initial_balance=1000.0)
    initial_price = 100.0

    # Step 1: Buy 100% (max leverage)
    pnl_pct,  traded, bankrupt = broker.step(1.0, 1.0, initial_price, step_index=0)
    assert traded is True
    assert not bankrupt

    # Step 2: Price drops 100% - complete loss (e.g., company goes bankrupt)
    # Portfolio value will be 0, triggering bankruptcy
    crash_price = 0.0
    pnl_pct, traded, bankrupt = broker.step(0.0, 0.0, crash_price, step_index=1)

    # Should trigger bankruptcy and force liquidation
    assert bankrupt is True
    assert traded is True  # Force liquidation occurred
    assert broker.position_shares == 0.0  # Position liquidated
    assert broker.balance == 0.0  # Everything lost


def test_step_history_recorded():
    """Test: Each step is recorded in history"""
    broker = TradingBroker(initial_balance=10000.0)
    price = 100.0

    # Execute 3 steps
    broker.step(1.0, 0.5, price, step_index=0)
    broker.step(0.0, 0.0, price, step_index=1)
    broker.step(0.0, 1.0, price, step_index=2)

    # Check history
    assert len(broker.step_history) == 3
    assert broker.step_history[0]['step'] == 0
    assert broker.step_history[1]['step'] == 1
    assert broker.step_history[2]['step'] == 2


def test_short_progression_with_price_changes():
    """
    Test: Short position with multiple adjustments and price changes
    Pattern: Sell 40% @ 300 -> Increase to 80% @ 150 -> Reduce to 20% @ 100 -> Close @ 50
    Tests all broker state variables through realistic short trading scenario
    """
    broker = TradingBroker(initial_balance=10000.0)

    # Step 1: Sell short 40% at price 300
    price_1 = 1000.0
    pnl_1, traded_1, bankrupt_1 = broker.step(-1.0, 1, price_1, step_index=0)

    # Validate Step 1 - Initial short position
    assert traded_1 is True
    assert bankrupt_1 is False
    assert broker.balance == 0.0
    assert broker.capital_used == 10000.0
    assert broker.position_shares == -10
    assert broker.entry_price == 1000.00, "Step 1: Entry price should be $300"
    assert broker.calculate_portfolio_value(price_1) == 10000

    # Step 2: Increase short to 80% at price 150 (price dropped - profit on short)
    price_2 = 500.0
    pnl_2,  traded_2, bankrupt_2 = broker.step(-1.0, 0.5, price_2, step_index=1)

    # Validate Step 2 - Increase short position (price dropped = profit)
    assert traded_2 is True
    assert bankrupt_2 is False
    assert broker.position_shares == -5
    assert broker.balance == 7500
    assert broker.capital_used == 5000
    assert broker.entry_price == 1000
    assert broker.calculate_portfolio_value(price_2) == 15000, "Step 2: Portfolio"

    # Step 4: Close position at price 50 (price dropped even more - final profit)
    price_4 = 50.0
    pnl_4, traded_4, bankrupt_4 = broker.step(0.0, 0.0, price_4, step_index=3)

    # Validate Step 4 - Close position completely
    assert traded_4 is True, "Step 4: Should execute trade (closing position)"
    assert bankrupt_4 is False, "Step 4: Should not be bankrupt"
    assert broker.position_shares == 0
    assert broker.capital_used == 0.0, "Step 4: No capital should be used"
    assert broker.balance == pytest.approx(17250, abs=0.01), "Step 4: Final balance"
    assert broker.position_shares == 0.0, "Step 4: Position fully closed"
    assert broker.entry_price == 0.0, "Step 4: Entry price reset"

    # Summary validation
    final_portfolio = broker.calculate_portfolio_value(price_4)
    total_pnl = final_portfolio - 10000.0

    assert len(broker.step_history) == 3, "Should have 4 steps in history"
    assert final_portfolio == pytest.approx(17250, abs=0.01), "Final portfolio value"
    assert total_pnl == pytest.approx(7250, abs=0.01), "Total P&L (gain)"
    assert broker.balance == final_portfolio, "Balance should equal portfolio value (no position)"
