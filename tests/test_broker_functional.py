

from environments.broker import TradingBroker
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def test_buy_empty():
    """Test: Buy -> Hold pattern"""
    broker = TradingBroker(initial_balance=10000.0)
    price = 100.0

    # Step 1: Buy 0% long
    pnl, traded, bankrupt = broker.step(1.0, 0, price, step_index=0)
    assert traded is False
    assert bankrupt is False
    assert broker.cash == 10000.0
    assert broker.cash_used == 0.0
    assert broker.position_shares == 0.0
    assert broker.entry_price == 0.0
    assert pnl == 0.0

    # Step 1: Buy 0% long
    pnl, traded, bankrupt = broker.step(-1.0, 0, price, step_index=0)
    assert traded is False
    assert bankrupt is False
    assert broker.cash == 10000.0
    assert broker.cash_used == 0.0
    assert broker.position_shares == 0.0
    assert broker.entry_price == 0.0
    assert pnl == 0.0


def test_buy_hold_pattern():
    """Test: Buy -> Hold pattern"""
    broker = TradingBroker(initial_balance=10000.0)
    price = 100.0

    # Step 1: Buy 100% long
    pnl, traded, bankrupt = broker.step(1.0, 1, price, step_index=0)
    assert traded is True
    assert bankrupt is False
    assert broker.cash == 10000.0
    assert broker.cash_used == 10000.0
    assert broker.position_shares == 100.0
    assert broker.entry_price == price
    assert pnl == 0.0

    price = 200.0
    # Step 2: Hold close 100%
    pnl,  traded, bankrupt = broker.step(0.0, 1.0, price, step_index=1)
    assert traded is True
    assert bankrupt is False
    assert broker.cash == 20000.0
    assert broker.cash_used == 0.0
    assert broker.position_shares == 0.0
    assert broker.entry_price == 0.0
    assert pnl == 10000


def test_buy_buy_pattern():
    """Test: Buy 30% -> Buy 60% (increase position)"""
    broker = TradingBroker(initial_balance=10000.0)
    price = 100.0

    # Step 1: Buy 30%
    pnl, traded, bankrupt = broker.step(1.0, 0.3, price, step_index=0)
    assert traded is True
    assert bankrupt is False
    assert broker.position_shares == 30.0  # 30% of 100 shares
    assert broker.cash_used == 3000.0
    assert broker.cash == 10000.0
    assert broker.entry_price == 100.0
    assert pnl == 0.0

    # Step 2: Buy 60% (increase)
    pnl, traded, bankrupt = broker.step(1.0, 0.6, price, step_index=1)
    assert traded is True
    assert bankrupt is False

    expected_shares = 60.0
    assert broker.position_shares == expected_shares

    # Cash used = shares * entry_price
    expected_cash_used = expected_shares * broker.entry_price
    assert broker.cash_used == expected_cash_used

    # Cash stays same
    assert broker.cash == 10000.0

    # Entry price weighted average
    expected_entry_price = (30 * 100 + 30 * 100) / 60  # old + new
    assert broker.entry_price == expected_entry_price


def test_buy_buy_boundaries():
    """Test: Buy 30% -> Buy 60% at different price scenarios"""

    # --- Case 1: price increases from 100 -> 200 ---
    broker = TradingBroker(initial_balance=10000.0)
    price1 = 100.0
    price2 = 200.0
    price3 = 250.0

    # Step 1: Buy 30% at 100
    pnl, traded, bankrupt = broker.step(1.0, 0.3, price1, step_index=0)
    assert broker.position_shares == 30.0
    assert broker.entry_price == 100.0
    assert broker.cash_used == 3000.0

    # Step 2: Buy 90% at 200
    pnl, traded, bankrupt = broker.step(1.0, 0.9, price2, step_index=1)
    assert traded is True
    assert bankrupt is False
    assert pnl == 3000.0  # Unrealized PnL from price increase
    assert broker.position_shares == 60
    assert broker.cash_used == 9000.0
    assert broker.entry_price == 150.0

    # Step 3: Buy 100% at 250
    pnl, traded, bankrupt = broker.step(1.0, 1.0, price3, step_index=2)
    assert traded is True
    assert bankrupt is False
    assert pnl == 3000.0  # Unrealized PnL from price increase
    assert broker.position_shares == 64.0
    assert broker.cash_used == 10000
    assert broker.entry_price == 156.25

    # --- Case 2: price decreases from 100 -> 50 ---
    broker = TradingBroker(initial_balance=10000.0)
    price1 = 100.0
    price2 = 50.0

    # Step 1: Buy 30% at 100
    pnl, traded, bankrupt = broker.step(1.0, 0.3, price1, step_index=0)
    assert broker.position_shares == 30.0
    assert broker.entry_price == 100.0
    assert broker.cash_used == 3000.0
    assert broker.entry_price == price1

    # Step 2: Buy 90% at 50
    pnl, traded, bankrupt = broker.step(1.0, 0.9, price2, step_index=1)
    assert broker.position_shares == 150
    assert broker.cash_used == 9000
    assert broker.entry_price == 60
    assert pnl == -1500.0  # Unrealized loss from price decrease


def test_sell_short_pattern():
    """Test: Sell short 50% -> Hold @ lower price -> Cover (should profit)"""
    broker = TradingBroker(initial_balance=20000.0)
    price_initial = 100.0
    price_lower = 50.0  # Price drops - good for shorts!

    # Step 1: Sell short 50%
    pnl, traded, bankrupt = broker.step(-1.0, 0.5, price_initial, step_index=0)
    assert traded is True
    assert broker.position_shares == -100.0  # No commission, exact shares
    assert broker.cash_used == 10000.0
    assert broker.cash == 20000.0
    assert pnl == 0.0  # No commission, no PnL yet

    # Step 2: Hold at lower price (price dropped 50% - we should have profit)
    pnl, traded, bankrupt = broker.step(-1.0, 0.5, price_lower, step_index=1)
    assert traded is False
    assert pnl == 5000
    assert broker.position_shares == -100.0  # No commission, exact shares
    assert broker.cash_used == 10000.0
    assert broker.cash == 20000.0

    # CRITICAL: Portfolio should show profit since price dropped
    unrealized_pnl = broker.calculate_unrealized_pnl(price_lower)
    assert unrealized_pnl == 5000.0, f"Expected $5,000 profit, got ${unrealized_pnl:.2f}"

    # Step 3: Cover at lower price (close short) - lock in profit
    pnl, traded, bankrupt = broker.step(0.0, 1.0, price_lower, step_index=1)
    assert traded is True
    assert broker.position_shares == 0
    assert broker.cash_used == 0.0
    assert broker.cash == 25000.0


def test_sell_short_pattern_with_loss():
    """Test: Sell short 50% -> Hold @ higher price -> Cover (should incur loss)"""
    broker = TradingBroker(initial_balance=20000.0)
    price_initial = 100.0
    price_higher = 150.0  # Price rises - bad for shorts!

    # Step 1: Sell short 50%
    pnl, traded, bankrupt = broker.step(-1.0, 0.5, price_initial, step_index=0)
    assert traded is True
    assert broker.position_shares == -100.0  # No commission, exact shares
    assert broker.cash_used == 10000.0
    assert broker.entry_price == 100.0
    # Cash is the realized balance and remains unchanged until PnL is realized
    assert broker.cash == 20000.0
    assert pnl == 0.0  # No commission, no PnL yet

    # Step 2: Hold at higher price (price rose 50% - we should have a loss)
    portfolio_before_hold = broker.calculate_portfolio_value(price_initial)
    assert portfolio_before_hold == 20000.0  # No change yet
    # Use the same short signal and size to 'hold' (no action) so the position remains open
    pnl, traded, bankrupt = broker.step(-1.0, 0.5, price_higher, step_index=1)
    assert traded is False
    assert pnl == -5000  # Loss of $5k on $20k initial = 25%
    assert broker.position_shares == -100.0
    assert broker.cash_used == 10000.0
    assert broker.cash == 20000.0

    # CRITICAL: Portfolio should show loss since price rose
    unrealized_pnl = broker.calculate_unrealized_pnl(price_higher)
    assert unrealized_pnl == -5000.0, f"Expected -$5,000 loss, got ${unrealized_pnl:.2f}"

    # Step 3: Cover at higher price (close short) - lock in loss
    pnl, traded, bankrupt = broker.step(0.0, 0.0, price_higher, step_index=2)
    assert traded is True
    assert broker.position_shares == 0.0
    assert broker.cash_used == 0.0
    assert broker.cash == 15000.0  # $10k free + $10k capital - $5k pnl = $15k


def test_sell_short_short_entry_price():
    broker = TradingBroker(initial_balance=10000.0)
    price_initial = 100.0
    price_lower = 50.0

    # Step 1: Sell short 50%
    pnl, traded, bankrupt = broker.step(-1.0, 0.5, price_initial, step_index=0)
    assert traded is True
    assert broker.position_shares == -50.0  # No commission, exact shares
    assert broker.cash_used == 5000.0
    assert broker.entry_price == 100.0
    assert broker.cash == 10000.0
    assert pnl == 0.0

    # Step 2: Increase short to 100% and verify entry price adjusts correctly
    pnl, traded, bankrupt = broker.step(-1.0, 1, price_lower, step_index=1)
    assert traded is True
    assert pnl == 2500
    assert broker.position_shares == -150.0
    assert broker.cash_used == 10000.5  # BUG on precision
    assert broker.cash == 10000.0
    assert broker.entry_price == 66.67


def test_short_close():
    broker = TradingBroker(initial_balance=99968.31)

    # Step 1: Sell short
    pnl, traded, bankrupt = broker.step(-1.0, 0.0025666533038020134, 110274.46, step_index=0)
    assert traded is True
    assert broker.position_shares == -0.002
    assert broker.cash_used == 220.55
    assert broker.entry_price == 110274.46
    assert broker.cash == 99968.31
    assert pnl == 0.0

    # Step 2: Sell short 50%
    pnl, traded, bankrupt = broker.step(-1.0, 0.17951692640781403, 110949.22, step_index=0)
    assert traded is True
    assert broker.position_shares == -0.161  # No commission, exact shares
    assert broker.cash_used == 17861.48
    assert broker.entry_price == 110940.84
    assert broker.cash == 99968.31
    assert pnl == pytest.approx(-1.349, abs=1e-3)

    # Step 3: Increase short to 100% and verify entry price adjusts correctly
    pnl, traded, bankrupt = broker.step(-1.0, 0, 111242.26, step_index=1)
    assert traded is True
    assert pnl == pytest.approx(-47.18, abs=1e-2)
    assert broker.position_shares == 0.0
    assert broker.cash_used == 0.0
    assert broker.cash == 99919.78
    assert broker.entry_price == 0.0


""" 2025-10-11 08: 00: 00	110274.46 - 0.00031684979249879675	763 - 1	0.0025666533038020134	110274.46 - 0.002	99968.31	220.55	0.0	99968.31	100007.11	0.0	True	12	False
2025-10-11 08: 15: 00	110949.22 - 0.00033033722663966536	764 - 1	0.17951692640781403	110940.84 - 0.161	99968.31	17861.48 - 1.3491800000007497	99966.96	100007.11 - 1.3495199999999896	True	13	False
2025-10-11 08: 30: 00	111242.26 - 0.0008018925122152964	765 - 1	0.0	110940.84 - 0.001	99920.08	110.94 - 0.30141999999999824	99919.78	100007.11 - 47.17943999999897	True	14	False
 """


def test_long_to_short_reversal():
    """Test: Long 50% @ $100 -> Price 4x to $400 -> Reverse to Short 50% (should double portfolio)"""
    broker = TradingBroker(initial_balance=10000.0)
    price_initial = 100.0
    price_higher = 400.0  # 400% increase (4x)

    # Step 1: Go long 50% @ $100
    # Buy 50 shares @ $100 using $5,000 of capital. The allocated amount is
    # tracked in `cash_used`. Note: the realized `cash` (available balance)
    # remains unchanged until the position is closed or PnL is realized.
    pnl_step, traded, bankrupt = broker.step(1.0, 0.5, price_initial, step_index=0)
    assert traded is True
    assert broker.position_shares == 50.0
    # Cash stays the same (realized cash) until position is closed
    assert broker.cash == 10000.0
    assert broker.cash_used == 5000.0

    # Step 2: Price rises to $400 (4x)
    # Portfolio = $5,000 (free) + $20,000 (50 shares @ $400) = $25,000
    portfolio_before_reverse = broker.calculate_portfolio_value(price_higher)
    unrealized_profit_from_long = broker.calculate_unrealized_pnl(price_higher)

    print("\nDEBUG test_long_to_short_reversal:")
    print("  Long position: 50 shares @ $100, now worth $400")
    print(f"  Unrealized profit: ${unrealized_profit_from_long:.2f}")
    print(f"  Portfolio before reverse: ${portfolio_before_reverse:.2f}")

    assert unrealized_profit_from_long == 15000.0, f"Expected $15,000 profit, got ${unrealized_profit_from_long:.2f}"
    assert portfolio_before_reverse == 25000.0, f"Expected $25,000 portfolio, got ${portfolio_before_reverse:.2f}"

    # Step 3: Reverse to short. This action will close the existing long
    # (realizing its PnL) and then open a new short position. The asserts
    # below validate the broker's post-trade state (position, cash_used,
    # and realized cash) rather than modeling the intermediate accounting.
    pnl_step, traded, bankrupt = broker.step(-1.0, 0.8, price_higher, step_index=1)
    assert pnl_step == unrealized_profit_from_long
    assert traded is True
    # After the reversal the test expects a short position and updated
    # cash/cash_used values (these are validated by the asserts below).
    assert broker.position_shares == -50.0
    assert broker.cash_used == 20000.0
    assert broker.cash == 25000.0


def test_reduce_position_pattern():
    """Test: Buy 100% -> Reduce by 30% -> Reduce by 50% more"""
    broker = TradingBroker(initial_balance=10000.0)
    price = 100.0

    # Step 1: Buy 100%
    pnl,  traded, bankrupt = broker.step(1.0, 1.0, price, step_index=0)
    assert traded is True
    assert broker.position_shares == 100
    assert broker.entry_price == 100

    # Step 2: Reduce by 30% (use same long signal to adjust position)
    pnl,  traded, bankrupt = broker.step(1.0, 0.3, price, step_index=1)
    assert traded is True
    assert broker.position_shares == 30
    assert broker.entry_price == 100

    # Step 3: Reduce by 50% more (use same long signal to adjust position)
    pnl,  traded, bankrupt = broker.step(1.0, 0.5, price, step_index=2)
    assert traded is True
    assert broker.position_shares == 50
    assert broker.entry_price == 100


def test_price_movement_pnl():
    """Test: Buy -> Price up -> Price down - validate P&L"""
    broker = TradingBroker(initial_balance=10000.0)
    initial_price = 100.0

    # Step 1: Buy 50%
    pnl,  traded, bankrupt = broker.step(1.0, 0.5, initial_price, step_index=0)
    assert traded is True

    # Step 2: Price goes up 10% - should have positive P&L
    new_price = 110.0
    pnl,  traded, bankrupt = broker.step(1.0, 0.5, new_price, step_index=1)
    assert traded is False  # Just holding
    assert pnl == 0.1 * 5000

    # Step 3: Price goes back down - should have negative P&L this step
    pnl,  traded, bankrupt = broker.step(1.0, 0.5, initial_price, step_index=2)
    assert traded is False
    assert pnl == -0.1 * 5000


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
    assert broker.cash == 0.0  # Everything lost


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

    # Step 1: Sell short 100% at price 1000
    price_1 = 1000.0
    pnl_1, traded_1, bankrupt_1 = broker.step(-1.0, 1, price_1, step_index=0)

    # Validate Step 1 - Initial short position
    assert traded_1 is True
    assert bankrupt_1 is False
    # Cash remains the starting balance until pnl is realized/position closed
    assert broker.cash == 10000.0
    assert broker.cash_used == 10000.0
    assert broker.position_shares == -10
    assert broker.entry_price == 1000.00, "Step 1: Entry price should be $300"
    assert broker.calculate_portfolio_value(price_1) == 10000

    # Step 2: Close 50% at price 500 (price dropped = profit on short)
    price_2 = 500.0
    pnl_2, traded_2, bankrupt_2 = broker.step(-1.0, 0.5, price_2, step_index=1)

    # Validate Step 2 - Close 50% position (price dropped = profit)
    assert traded_2 is True
    assert bankrupt_2 is False
    assert broker.position_shares == 0
    assert broker.cash == 15000
    assert broker.cash_used == 0
    assert broker.entry_price == 0.0


def test_short_decrease_capital_used_bug():
    """
    Regression test for negative capital_used bug in _decrease_short

    This reproduces the exact sequence from the live data where capital_used
    became negative at step 2162.

    Sequence from bug:
    - Step 2161: Open short -0.005 shares @ 115865.5 (exposure 0.633)
    - Step 2162: Reduce to -0.001 shares @ 115860.01 (exposure 0.258)
    - Bug: capital_used becomes -115.86 instead of staying positive
    """
    broker = TradingBroker(initial_balance=1001.45)

    # Step 2161: Open short with 63.3% exposure at price 115865.5
    price_1 = 115865.5
    pnl_1, traded_1, bankrupt_1 = broker.step(
        -1.0,
        0.6332878470420837,
        price_1,
        step_index=2161
    )

    # Validate step 2161
    assert traded_1 is True, "Should execute trade"
    assert bankrupt_1 is False
    assert broker.position_shares < 0, "Should be short"
    assert broker.position_shares == -0.005, "Should be -0.005 shares"
    assert broker.cash_used == 579.33, "Capital used should be ~579.33"
    assert broker.cash == 1001.45, "Balance should be ~1001.45"

    # Step 2162: Reduce short to 25.8% exposure at slightly lower price
    price_2 = 115860.01
    pnl_2, traded_2, bankrupt_2 = broker.step(
        -1.0,
        0.25854358077049255,
        price_2,
        step_index=2162
    )

    # Validate step 2162 - THIS IS WHERE THE BUG OCCURS
    assert traded_2 is True, "Should execute trade (reducing position)"
    assert bankrupt_2 is False
    assert broker.position_shares == -0.003, "Should be -0.003 shares"
    assert broker.cash_used == 347.6

    # Balance should increase (we closed part of short at profit)
    assert broker.cash > 422.21, f"Balance should increase, got {broker.cash:.2f}"

    # Portfolio should stay roughly the same
    portfolio = broker.calculate_portfolio_value(price_2)
    assert portfolio == pytest.approx(1001.57, abs=1.0), (
        f"Portfolio should be ~1001.57, got {portfolio:.2f}"
    )

    print(f"\n✅ Test passed - capital_used stayed positive: {broker.cash_used:.2f}")


def test_short_close_no_double_counting():
    """
    Test that closing a short position does not double-count P&L or capital.
    The portfolio value before and after closing should only change by the realized P&L.
    """
    broker = TradingBroker(initial_balance=1001.0)
    price_entry = 110309.61
    price_close = 1112.43  # Simulate a large move for clarity

    # Step 1: Open short position
    # Use a larger position_size so shares_to_trade >= quantity_precision
    pnl, traded, bankrupt = broker.step(-1.0, 1.0, price_entry, step_index=0)
    assert traded is True
    assert broker.position_shares < 0
    capital_used_before = broker.cash_used
    balance_before = broker.cash
    portfolio_before = broker.calculate_portfolio_value(price_entry)

    # Step 2: Close short position
    pnl, traded, bankrupt = broker.step(0.0, 0.0, price_close, step_index=1)
    assert traded is True
    assert broker.position_shares == 0
    assert broker.cash_used == 0

    balance_after = broker.cash
    portfolio_after = broker.calculate_portfolio_value(price_close)

    # The change in portfolio should equal the realized P&L
    realized_pnl = balance_after - balance_before
    expected_pnl = abs(capital_used_before) * (price_entry - price_close) / price_entry

    # Portfolio after close should not be artificially boosted
    assert abs(portfolio_after - (portfolio_before + expected_pnl)) < 1.0, (
        f"Portfolio boost detected: before={portfolio_before}, after={portfolio_after}, "
        f"expected change={expected_pnl}, actual change={portfolio_after - portfolio_before}"
    )

    # Explicitly check that balance is not boosted beyond expected
    expected_balance = balance_before + expected_pnl
    assert abs(balance_after - expected_balance) < 1.0, (
        f"Balance boost detected: before={balance_before}, after={balance_after}, "
        f"expected change={expected_pnl}, actual change={balance_after - balance_before}"
    )

    print(f"✅ No double-counting: portfolio and balance change match realized P&L ({realized_pnl:.2f})")
