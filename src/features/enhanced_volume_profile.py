"""
Enhanced Session Volume Profile - OPTIMIZED VERSION

Changes:
- Use typical price (H+L+C)/3 instead of H/L split
- Normalize bins once when storing
- Skip touched levels in naked check
- Early exit for flat markets
- Add buffer overflow warning
"""

import torch
from collections import deque


class EnhancedVolumeProfile:
    """Volume Profile calculator - session-based, optimized."""

    def __init__(self, n_bins=100, lookback_window=288, device="cuda", session_start_hour=0):
        self.device = torch.device(device)
        self.n_bins = n_bins
        self.lookback_window = lookback_window
        self.session_start_hour = session_start_hour

        # Pre-allocated buffers for VP calculations
        self.session_weights = torch.zeros(n_bins, device=device, dtype=torch.float32)
        self.session_bins = torch.zeros(n_bins + 1, device=device, dtype=torch.float32)

        # Current day session data (OPTIMIZED: single price per bar)
        self.max_session_bars = 5000  # Max bars per day (was 10000 prices)
        self.current_session_prices = torch.zeros(self.max_session_bars, device=device, dtype=torch.float32)
        self.current_session_volumes = torch.zeros(self.max_session_bars, device=device, dtype=torch.float32)
        self.current_session_idx = 0
        self.current_session_date = None
        self.current_session_open = None
        self.current_session_high = None
        self.current_session_low = None

        # Current day VP levels (updated live)
        self.current_day_vah = None
        self.current_day_val = None
        self.current_day_poc = None
        self.current_day_high = None
        self.current_day_low = None

        # Session history (last 10 days/weeks)
        self.max_session_history = 10
        self.daily_sessions = deque(maxlen=self.max_session_history)
        self.weekly_sessions = deque(maxlen=self.max_session_history)

        # Current week tracking
        self.current_week = None
        self.weekly_prices = torch.zeros(self.max_session_bars * 7, device=device, dtype=torch.float32)
        self.weekly_volumes = torch.zeros(self.max_session_bars * 7, device=device, dtype=torch.float32)
        self.weekly_idx = 0

        # Current day VP bins history (INTRADAY only - NORMALIZED on store)
        self.daily_bins_history = torch.zeros(lookback_window, n_bins, device=device, dtype=torch.float32)
        self.daily_bins_idx = 0
        self.daily_bins_count = 0
        self.bins = torch.zeros(n_bins + 1, device=device, dtype=torch.float32)
        self.weights = torch.zeros(n_bins, device=device, dtype=torch.float32)

    def update(self, timestamp, open_price, high_price, low_price, close_price, volume):
        """Update VP with new bar."""
        open_price = float(open_price)
        high_price = float(high_price)
        low_price = float(low_price)
        close_price = float(close_price)
        volume = float(volume)

        current_date = timestamp.date()
        current_week = timestamp.isocalendar()[1]

        # Check for new day
        if self.current_session_date != current_date:
            self._end_session(close_price, 'day')
            self._start_new_session(open_price, high_price, low_price, current_date, 'day')

        # Check for new week
        if self.current_week != current_week:
            self._end_session(close_price, 'week')
            self._start_new_session(open_price, high_price, low_price, current_date, 'week')
            self.current_week = current_week

        # Update session data
        self._update_current_session(open_price, high_price, low_price, close_price, volume)

        # Calculate current day VP from session data
        self._calculate_current_day_vp()

        # Update intraday bins history
        self._update_daily_bins_history()

    def _start_new_session(self, open_price, high_price, low_price, current_date, session_type='day'):
        """Start new session."""
        if session_type == 'day':
            self.current_session_date = current_date
            self.current_session_open = open_price
            self.current_session_high = high_price
            self.current_session_low = low_price
            self.current_session_idx = 0
            self.current_day_high = high_price
            self.current_day_low = low_price

            # Reset daily bins history
            self.daily_bins_history.zero_()
            self.daily_bins_idx = 0
            self.daily_bins_count = 0

        elif session_type == 'week':
            self.weekly_idx = 0

    def _end_session(self, close_price, session_type='day'):
        """Store completed session."""
        if session_type == 'day':
            if self.current_session_idx == 0:
                return

            vah, val, poc = self._calculate_value_area_fast(
                self.current_session_prices[:self.current_session_idx],
                self.current_session_volumes[:self.current_session_idx]
            )

            # OPTIMIZED: Skip storing if flat market (all levels None)
            if vah is None:
                return

            session_info = {
                'date': self.current_session_date,
                'open': self.current_session_open,
                'high': self.current_session_high,
                'low': self.current_session_low,
                'close': close_price,
                'vah': vah,
                'val': val,
                'poc': poc,
                'vah_touched': False,
                'val_touched': False,
                'poc_touched': False
            }
            self.daily_sessions.append(session_info)

        elif session_type == 'week':
            if self.weekly_idx == 0:
                return

            vah, val, poc = self._calculate_value_area_fast(
                self.weekly_prices[:self.weekly_idx],
                self.weekly_volumes[:self.weekly_idx]
            )

            if vah is None:
                return

            session_info = {
                'week': self.current_week,
                'vah': vah,
                'val': val,
                'poc': poc,
                'vah_touched': False,
                'val_touched': False,
                'poc_touched': False
            }
            self.weekly_sessions.append(session_info)

    def _update_current_session(self, open_price, high_price, low_price, close_price, volume):
        """Update current session data - OPTIMIZED: typical price only."""
        # OPTIMIZED: Use typical price (single entry per bar)
        typical_price = (high_price + low_price + close_price) / 3

        # Check buffer overflow
        if self.current_session_idx >= self.max_session_bars:
            print(f"⚠️ WARNING: Session buffer full ({self.max_session_bars} bars), skipping data")
            return

        self.current_session_prices[self.current_session_idx] = typical_price
        self.current_session_volumes[self.current_session_idx] = volume
        self.current_session_idx += 1

        # Weekly data
        if self.weekly_idx < len(self.weekly_prices):
            self.weekly_prices[self.weekly_idx] = typical_price
            self.weekly_volumes[self.weekly_idx] = volume
            self.weekly_idx += 1

        # Update high/low
        if high_price > self.current_session_high:
            self.current_session_high = high_price
        if low_price < self.current_session_low:
            self.current_session_low = low_price

        if high_price > self.current_day_high:
            self.current_day_high = high_price
        if low_price < self.current_day_low:
            self.current_day_low = low_price

        # Check naked levels
        self._check_naked_levels_touched(low_price, high_price)

    def _check_naked_levels_touched(self, low_price, high_price):
        """Mark levels as touched - OPTIMIZED: skip already touched."""
        # OPTIMIZED: Only check untouched levels
        for session in self.daily_sessions:
            if not session['vah_touched']:
                if low_price <= session['vah'] <= high_price:
                    session['vah_touched'] = True
            if not session['val_touched']:
                if low_price <= session['val'] <= high_price:
                    session['val_touched'] = True
            if not session['poc_touched']:
                if low_price <= session['poc'] <= high_price:
                    session['poc_touched'] = True

        for session in self.weekly_sessions:
            if not session['vah_touched']:
                if low_price <= session['vah'] <= high_price:
                    session['vah_touched'] = True
            if not session['val_touched']:
                if low_price <= session['val'] <= high_price:
                    session['val_touched'] = True
            if not session['poc_touched']:
                if low_price <= session['poc'] <= high_price:
                    session['poc_touched'] = True

    def _calculate_current_day_vp(self):
        """Calculate current day VAH/VAL/POC from session data."""
        if self.current_session_idx > 0:
            vah, val, poc = self._calculate_value_area_fast(
                self.current_session_prices[:self.current_session_idx],
                self.current_session_volumes[:self.current_session_idx]
            )
            self.current_day_vah = vah
            self.current_day_val = val
            self.current_day_poc = poc

    def _update_daily_bins_history(self):
        """Update current day bins history - OPTIMIZED: normalize once."""
        if self.current_session_idx < 2:
            return

        # Calculate bins from current session data
        prices = self.current_session_prices[:self.current_session_idx]
        volumes = self.current_session_volumes[:self.current_session_idx]

        price_min = prices.min()
        price_max = prices.max()

        # OPTIMIZED: Skip flat markets
        if price_max - price_min < 1e-6:
            return

        self.bins = torch.linspace(price_min, price_max, self.n_bins + 1,
                                   device=self.device, dtype=torch.float32)

        bin_indices = torch.bucketize(prices, self.bins[:-1], right=True)
        bin_indices = torch.clamp(bin_indices, 0, self.n_bins - 1)

        self.weights.zero_()
        self.weights.index_add_(0, bin_indices, volumes)

        total_vol = self.weights.sum()
        if total_vol > 0:
            self.weights /= total_vol

        # OPTIMIZED: Normalize before storing (not on every get)
        weights_normalized = self.weights.clone()
        weights_min = weights_normalized.min()
        weights_max = weights_normalized.max()
        if weights_max > weights_min:
            weights_normalized = (weights_normalized - weights_min) / (weights_max - weights_min)

        # Store normalized weights in history
        self.daily_bins_history[self.daily_bins_idx] = weights_normalized
        self.daily_bins_idx = (self.daily_bins_idx + 1) % self.lookback_window
        self.daily_bins_count = min(self.daily_bins_count + 1, self.lookback_window)

    def _calculate_value_area_fast(self, prices_t, volumes_t):
        """Calculate VAH/VAL/POC - OPTIMIZED: return None for flat markets."""
        if len(prices_t) == 0:
            return None, None, None

        price_min = prices_t.min()
        price_max = prices_t.max()

        # OPTIMIZED: Return None instead of flat POC
        if price_max - price_min < 1e-6:
            return None, None, None

        torch.linspace(price_min, price_max, self.n_bins + 1,
                       out=self.session_bins, dtype=torch.float32)

        bin_indices = torch.bucketize(prices_t, self.session_bins[:-1], right=True)
        bin_indices = torch.clamp(bin_indices, 0, self.n_bins - 1)

        self.session_weights.zero_()
        self.session_weights.index_add_(0, bin_indices, volumes_t)

        total_vol = self.session_weights.sum()
        if total_vol > 0:
            self.session_weights /= total_vol

        poc_bin = torch.argmax(self.session_weights)
        poc = (self.session_bins[poc_bin] + self.session_bins[poc_bin + 1]) / 2

        sorted_weights, sorted_indices = torch.sort(self.session_weights, descending=True)
        cumulative_vol = torch.cumsum(sorted_weights, dim=0)
        va_mask = cumulative_vol <= 0.7

        if torch.any(va_mask):
            va_bins = sorted_indices[va_mask]
            vah_bin = va_bins.max()
            val_bin = va_bins.min()
            vah = (self.session_bins[vah_bin] + self.session_bins[vah_bin + 1]) / 2
            val = (self.session_bins[val_bin] + self.session_bins[val_bin + 1]) / 2
        else:
            vah = poc
            val = poc

        return float(vah), float(val), float(poc)

    def reset(self):
        """Reset for new episode."""
        self.current_session_idx = 0
        self.weekly_idx = 0
        self.current_session_date = None
        self.current_day_vah = None
        self.current_day_val = None
        self.current_day_poc = None
        self.current_day_high = None
        self.current_day_low = None
        self.daily_sessions.clear()
        self.weekly_sessions.clear()
        self.daily_bins_history.zero_()
        self.daily_bins_idx = 0
        self.daily_bins_count = 0

    def get_poc(self):
        """Get current POC."""
        return self.current_day_poc

    def get_daily_sessions(self):
        """Get historical daily sessions."""
        return list(self.daily_sessions)

    def get_weekly_sessions(self):
        """Get historical weekly sessions."""
        return list(self.weekly_sessions)
