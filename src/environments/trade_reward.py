import numpy as np


class TradeReward:
    def __init__(self, broker, df, vp_levels=None, window_size=96, wick_lookback=3):
        """
        broker: your broker instance
        df: dataframe with OHLCV
        vp_levels: dict with keys like 'daily', 'prev_day', 'weekly' containing VP info
        window_size: how many steps to look back for positions
        wick_lookback: number of candles to check for wick rejections
        """
        self.broker = broker
        self.df = df
        self.vp_levels = vp_levels or {}
        self.window_size = window_size
        self.wick_lookback = wick_lookback

    # ---------------- Helper functions ----------------
    @staticmethod
    def touched_level(price, level, tol=0.001):
        """Check if price is within a tolerance of a level"""
        return abs(price - level) / max(level, 1e-6) < tol

    def wick_rejection(self, idx, level, kind='resistance'):
        """
        Checks wick rejections over wick_lookback candles ending at idx
        Returns a normalized strength 0-1
        """
        start = max(0, idx - self.wick_lookback + 1)
        candles = self.df.iloc[start:idx + 1]
        strength = 0.0
        for _, c in candles.iterrows():
            high, low, close = c['high'], c['low'], c['close']
            if kind == 'resistance' and high >= level > close:
                strength += (high - level) / max(high - low, 1e-6)
            elif kind == 'support' and low <= level < close:
                strength += (level - low) / max(high - low, 1e-6)
        return min(strength, 1.0)

    def volume_profile_bonus(self, price, signal):
        reward = 0.0
        for vp in self.vp_levels.values():
            if vp:
                scale = max(vp['vah'] - vp['val'], 1e-6)
                reward += 0.15 * max(0, 1 - abs(price - vp['poc']) / scale)
                if signal == 1:
                    reward += 0.15 * max(0, 1 - abs(price - vp['val']) / scale)
                elif signal == -1:
                    reward += 0.15 * max(0, 1 - abs(price - vp['vah']) / scale)
        return reward

    def support_resistance_bonus(self, idx, price, signal):
        """Optional: simple support/resistance check using recent highs/lows"""
        recent = self.df.iloc[max(0, idx - 10):idx]  # last 10 candles
        reward = 0.0
        if signal == 1:  # long
            s_level = recent['low'].min()
            if self.touched_level(price, s_level):
                reward += 0.1
        elif signal == -1:  # short
            r_level = recent['high'].max()
            if self.touched_level(price, r_level):
                reward += 0.1
        return reward

    # ---------------- Main reward calculation ----------------
    def calculate(self, current_step, current_price):
        if current_step <= self.window_size:
            return 0.0

        reward = 0.0
        idx = current_step
        last_step = self.broker.step_history[idx - 1 - self.window_size]
        prev_pos = last_step['position_shares']
        curr_pos = self.broker.position_shares
        signal = self.broker.signal

        opened = prev_pos == 0 and curr_pos != 0
        closed = prev_pos != 0 and curr_pos == 0
        increased = np.sign(prev_pos) == np.sign(curr_pos) and abs(curr_pos) > abs(prev_pos)
        reduced = np.sign(prev_pos) == np.sign(curr_pos) and abs(curr_pos) < abs(prev_pos)
        reversed_pos = prev_pos != 0 and curr_pos != 0 and np.sign(prev_pos) != np.sign(curr_pos)

        # -------- Open Trade Reward --------
        if opened:
            reward += 0.1
            reward += self.volume_profile_bonus(current_price, signal)
            reward += self.support_resistance_bonus(idx, current_price, signal)
            reward += self.wick_rejection(idx, current_price, kind='support' if signal == 1 else 'resistance') * 0.2

        # -------- Increase Position Reward --------
        if increased:
            reward += 0.4 if self.broker.unrealized_pnl > 0 else -0.2

        # -------- Reduce Position Reward --------
        if reduced:
            pnl = self.broker.cash - last_step['cash']
            reward += 0.4 if pnl > 0 else 0.05

        # -------- Reversal Reward --------
        if reversed_pos:
            next_close = self.df['close'].iloc[idx + 1] if idx + 1 < len(self.df) else current_price
            pnl = self.broker.cash - last_step['cash']
            good_dir = (signal == 1 and next_close > current_price) or (signal == -1 and next_close < current_price)
            if good_dir:
                reward += 0.6 + (0.3 if pnl > 0 else 0.05)
            else:
                reward -= 0.3

        # -------- Close Position Reward --------
        if closed:
            pnl = self.broker.cash - last_step['cash']
            cash_used = last_step['cash_used'] or 1e-6
            pct = abs(pnl / cash_used) * 100
            if pnl > 0:
                if pct >= 10:
                    reward += 1.5
                elif pct >= 5:
                    reward += 1.0
                elif pct >= 1:
                    reward += 0.7
                else:
                    reward += 0.3
            else:
                reward -= 0.7
            reward += self.volume_profile_bonus(current_price, signal)

        # -------- Small holding bonus --------
        if curr_pos != 0:
            reward += 0.01

        return float(np.clip(reward, -1, 1))
