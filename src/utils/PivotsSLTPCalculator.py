import pandas as pd
from typing import Tuple


class PivotSLTPCalculator:
    """
    Calculates dynamic Stop Loss and Take Profit levels based on ATR (Average True Range).

    For LONG positions:
        - SL: Entry - (ATR * multiplier) - adaptive to volatility
        - TP: Entry + (risk * risk_reward_ratio) - default 3:1 ratio

    For SHORT positions:
        - SL: Entry + (ATR * multiplier) - adaptive to volatility
        - TP: Entry - (risk * risk_reward_ratio) - default 3:1 ratio

    ATR-based stops adapt to market volatility and don't require pivot detection.
    """

    @staticmethod
    def calculate_sl_tp(data: pd.DataFrame,
                        current_step: int,
                        entry_price: float,
                        direction: int,
                        risk_reward_ratio: float = 3.0,
                        atr_multiplier: float = 2.0) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit prices based on ATR.

        Args:
            data: DataFrame with 'atr' column
            current_step: Current position in the data
            entry_price: Entry price of the position
            direction: 1 for LONG, 2 for SHORT
            risk_reward_ratio: TP distance as multiple of SL distance (default: 3.0)
            atr_multiplier: ATR multiplier for stop loss (default: 2.0)

        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        if direction not in [1, 2]:
            raise ValueError(f"Invalid direction: {direction}. Must be 1 (LONG) or 2 (SHORT)")

        # Get current ATR value
        current_atr = data['atr'].iloc[max(0, current_step-9):current_step+1].fillna(0).mean()
        # Calculate stop loss based on ATR
        if direction == 1:  # LONG
            sl_price = entry_price - (current_atr * atr_multiplier)
        else:  # SHORT
            sl_price = entry_price + (current_atr * atr_multiplier)

        # Calculate risk distance AFTER adjustment
        risk = abs(entry_price - sl_price)

        # Calculate TP based on risk-reward ratio
        if direction == 1:  # LONG
            tp_price = entry_price + (risk * risk_reward_ratio)
        else:  # SHORT
            tp_price = entry_price - (risk * risk_reward_ratio)

        return sl_price, tp_price
