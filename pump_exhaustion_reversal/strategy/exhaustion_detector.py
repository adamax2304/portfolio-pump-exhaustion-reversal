"""Exhaustion Detector (spec sections 11-12): after a pump is being watched,
decide whether its underlying mechanism has broken. Never opens a trade by
itself - it produces the Exhaustion Score and the individual component
flags that entry_engine.py's hard gates and per-mode requirements consume.

compute_exhaustion_flags_at() takes already-computed feature frames (from
features/structure.py, cvd.py, oi.py, volume.py, liquidations.py, avwap.py,
funding.py, rsi.py) and reads only index [0, i] - so it is safe to call for
i = "now" during a live scan or for every i while replaying history, with
identical results either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..config import AppConfig
from . import scoring


@dataclass
class ExhaustionFlags:
    structure_break: bool | None = None
    failed_reclaim_lower_high: bool | None = None
    cvd_bearish_divergence: bool | None = None
    oi_rollover: bool | None = None
    volume_decay: bool | None = None
    short_liquidation_exhaustion: bool | None = None
    sweep_rejection: bool | None = None
    avwap_loss: bool | None = None
    funding_long_crowding: bool | None = None
    rsi_divergence: bool | None = None

    def as_dict(self) -> dict[str, bool | None]:
        return {
            "structure_break": self.structure_break,
            "failed_reclaim_lower_high": self.failed_reclaim_lower_high,
            "cvd_bearish_divergence": self.cvd_bearish_divergence,
            "oi_rollover": self.oi_rollover,
            "volume_decay": self.volume_decay,
            "short_liquidation_exhaustion": self.short_liquidation_exhaustion,
            "sweep_rejection": self.sweep_rejection,
            "avwap_loss": self.avwap_loss,
            "funding_long_crowding": self.funding_long_crowding,
            "rsi_divergence": self.rsi_divergence,
        }


@dataclass
class ExhaustionInputs:
    """Bundle of per-bar feature frames covering [0, i] of a single post-pump
    event window. Any frame/column may be omitted if that data source is
    unavailable - the corresponding flag comes back None (missing), not False,
    and scoring.py renormalizes around it rather than scoring it as "not exhausted".
    """

    structure: pd.DataFrame  # features.structure.compute_structure output
    failed_reclaim: pd.Series  # features.structure.detect_failed_reclaim output
    sweep_rejection: pd.Series  # features.structure.detect_sweep_rejection output
    cvd_divergence: pd.Series | None = None  # features.cvd.cvd_bearish_divergence
    cvd_rollover: pd.Series | None = None  # features.cvd.cvd_rollover
    oi_rollover: pd.Series | None = None  # features.oi.detect_oi_rollover()["oi_rollover"]
    volume_decayed: pd.Series | None = None  # features.volume.volume_decay_flags()["volume_decayed"]
    liq_depletion_ratio: pd.Series | None = None  # features.liquidations.liquidation_depletion_ratio
    avwap_loss_columns: list[pd.Series] = field(default_factory=list)  # one per configured anchor
    funding_overheated: pd.Series | None = None  # features.funding.funding_heat_flags()["funding_overheated"]
    rsi_divergence: pd.Series | None = None  # features.rsi.rsi_bearish_divergence


def compute_exhaustion_flags_at(i: int, inputs: ExhaustionInputs, cfg: AppConfig) -> ExhaustionFlags:
    structure_break = bool(inputs.structure["structure_break_bar"].iloc[: i + 1].any())

    return ExhaustionFlags(
        structure_break=structure_break,
        failed_reclaim_lower_high=(bool(inputs.failed_reclaim.iloc[i]) if cfg.failed_reclaim.enabled else None),
        cvd_bearish_divergence=(
            bool(inputs.cvd_divergence.iloc[i]) if cfg.cvd.enabled and inputs.cvd_divergence is not None else None
        ),
        oi_rollover=(bool(inputs.oi_rollover.iloc[i]) if inputs.oi_rollover is not None else None),
        volume_decay=(bool(inputs.volume_decayed.iloc[i]) if inputs.volume_decayed is not None else None),
        short_liquidation_exhaustion=(
            bool(inputs.liq_depletion_ratio.iloc[i] <= cfg.liquidations.depletion_ratio_threshold)
            if inputs.liq_depletion_ratio is not None and not pd.isna(inputs.liq_depletion_ratio.iloc[i])
            else None
        ),
        sweep_rejection=bool(inputs.sweep_rejection.iloc[i]),
        avwap_loss=(
            bool(any(col.iloc[i] for col in inputs.avwap_loss_columns))
            if cfg.avwap.enabled and inputs.avwap_loss_columns
            else None
        ),
        funding_long_crowding=(
            bool(inputs.funding_overheated.iloc[i]) if inputs.funding_overheated is not None else None
        ),
        rsi_divergence=(bool(inputs.rsi_divergence.iloc[i]) if cfg.rsi.enabled and inputs.rsi_divergence is not None else None),
    )


def compute_exhaustion_score_at(flags: ExhaustionFlags, cfg: AppConfig) -> dict:
    return scoring.compute_exhaustion_score(flags.as_dict(), cfg.exhaustion_score_weights)


def classify_exhaustion_state(exhaustion_score: float, cfg: AppConfig) -> str:
    """WATCHING (< watch) -> EXHAUSTION_FORMING (>= watch) -> ARMED_SHORT (>=
    armed). The higher trade/high_confidence thresholds are consumed by hard
    gates and per-entry-mode min_exhaustion_score, not by this state map -
    see spec section 20's state machine transitions."""
    t = cfg.exhaustion_thresholds
    if exhaustion_score >= t.armed:
        return "ARMED_SHORT"
    if exhaustion_score >= t.watch:
        return "EXHAUSTION_FORMING"
    return "WATCHING"
