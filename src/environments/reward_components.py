"""
Modular Reward Components for Trading Environment

Simple, focused structure for entry/exit/hold quality evaluation.
Each component uses actual available data from the environment.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class RewardComponent(ABC):
    """Base class for modular reward components."""

    def __init__(self, weight: float = 1.0, enabled: bool = True):
        self.weight = weight
        self.enabled = enabled
        self.debug_info = {}

    @abstractmethod
    def calculate(
        self,
        action: int,
        current_state: Dict[str, Any],
        previous_state: Dict[str, Any],
        current_step: int,
        data
    ) -> float:
        """
        Calculate reward component.

        Args:
            action: 0=HOLD, 1=LONG, 2=SHORT, 3=CLOSE
            current_state: Broker state after action
            previous_state: Broker state before action
            current_step: Current data index
            data: Full dataframe with OHLC + indicators

        Returns:
            float: Reward in [-1, 1] range
        """
        pass


class VPEntryQualityComponent(RewardComponent):
    """
    Reward entries based on Volume Profile levels.

    Simple logic:
    - LONG near VAL/POC: +1.0 (buying support)
    - SHORT near VAH/POC: +1.0 (selling resistance)
    - LONG at VAH: -0.5 (buying resistance)
    - SHORT at VAL: -0.5 (selling support)
    """

    def calculate(
        self,
        action: int,
        current_state: Dict[str, Any],
        previous_state: Dict[str, Any],
        current_step: int,
        data
    ) -> float:
        # Only evaluate entries (LONG/SHORT when no position)
        if action not in [1, 2]:
            return 0.0
        if previous_state.get('position_size', 0) != 0:
            return 0.0

        # Get current price
        close = data['close'].iloc[current_step]

        # Calculate VP levels on-the-fly (same as environment does)
        from features.visible_range_vp import VisibleRangeVP

        lookback = 288  # Match environment lookback
        start_idx = max(0, current_step - lookback)
        window_data = data.iloc[start_idx:current_step]

        if len(window_data) < 2:
            self.debug_info = {'reason': 'insufficient_data'}
            return 0.0

        vp = VisibleRangeVP(n_bins=50)
        _, levels = vp.calculate_vp(window_data)

        vah = levels['vah']
        val = levels['val']
        poc = levels['poc']

        # Calculate normalized distances
        vp_range = vah - val
        if vp_range < 1e-6:
            self.debug_info = {'reason': 'flat_vp'}
            return 0.0

        if action == 1:  # LONG
            if close <= val:
                reward = 1.0  # Perfect: buying at VAL (support)
            elif close <= poc:
                reward = 0.5  # Good: buying between VAL-POC
            elif close >= vah:
                reward = -0.5  # Bad: buying at VAH (resistance)
            else:
                reward = 0.0

            self.debug_info = {
                'action': 'LONG',
                'close': close,
                'val': val,
                'poc': poc,
                'vah': vah,
                'reward': reward
            }
            return reward

        elif action == 2:  # SHORT
            if close >= vah:
                reward = 1.0  # Perfect: shorting at VAH (resistance)
            elif close >= poc:
                reward = 0.5  # Good: shorting between POC-VAH
            elif close <= val:
                reward = -0.5  # Bad: shorting at VAL (support)
            else:
                reward = 0.0

            self.debug_info = {
                'action': 'SHORT',
                'close': close,
                'val': val,
                'poc': poc,
                'vah': vah,
                'reward': reward
            }
            return reward

        return 0.0


class ModularRewardFunction:
    """
    Composable reward function that combines multiple components.

    Usage:
        reward_fn = ModularRewardFunction()
        reward_fn.add_component('pnl', PnLComponent(), weight=0.7)
        reward_fn.add_component('vp_entry', VPEntryQualityComponent(), weight=0.3)

        reward, breakdown = reward_fn.calculate(action, current_state, previous_state, step, data)
    """

    def __init__(self):
        self.components = {}  # {name: (component, weight)}
        self.last_breakdown = {}

    def add_component(self, name: str, component: RewardComponent, weight: float):
        """Add a weighted component."""
        self.components[name] = (component, weight)

    def calculate(
        self,
        action: int,
        current_state: Dict[str, Any],
        previous_state: Dict[str, Any],
        current_step: int,
        data
    ) -> tuple[float, Dict[str, Any]]:
        """
        Calculate total reward from all components.

        Returns:
            (total_reward, breakdown_dict)
        """
        total_reward = 0.0
        breakdown = {}

        for name, (component, weight) in self.components.items():
            if not component.enabled:
                continue

            component_reward = component.calculate(
                action, current_state, previous_state, current_step, data
            )
            weighted_reward = weight * component_reward
            total_reward += weighted_reward

            breakdown[name] = {
                'raw': component_reward,
                'weight': weight,
                'weighted': weighted_reward,
                'debug': component.debug_info.copy()
            }

        self.last_breakdown = breakdown
        return total_reward, breakdown

    def get_breakdown(self) -> Dict[str, Any]:
        """Get detailed breakdown of last calculation."""
        return self.last_breakdown
