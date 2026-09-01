"""Numeric silence policy driven by the learned EOT probability."""

from __future__ import annotations


class TurnEndPolicy:
    """Map model confidence and finality to a bounded silence threshold."""

    def __init__(
        self,
        t_max: float = 2.2,
        t_urgent: float = 0.18,
        t_fast: float = 0.25,
        t_mid: float = 1.0,
        t_deep: float = 2.0,
        is_final_reduction: float = 0.2,
    ) -> None:
        self.T_MAX = t_max
        self.T_URGENT = t_urgent
        self.T_FAST = t_fast
        self.T_MID = t_mid
        self.T_DEEP = t_deep
        self._is_final_reduction = is_final_reduction

    def get_dynamic_threshold(
        self,
        p_complete: float,
        is_final: bool,
    ) -> float:
        score = max(0.0, min(1.0, float(p_complete)))
        if score >= 0.8:
            threshold = self.T_FAST
        elif score >= 0.4:
            threshold = self.T_MID
        else:
            threshold = self.T_DEEP
        if is_final:
            threshold = max(self.T_URGENT, threshold - self._is_final_reduction)
        return max(self.T_URGENT, min(self.T_MAX, threshold))
