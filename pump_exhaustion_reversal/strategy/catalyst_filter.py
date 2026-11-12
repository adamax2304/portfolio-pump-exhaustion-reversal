"""CatalystFilter (spec section 10): which entry modes are allowed given
recent news activity. features/news.py only loads/queries catalyst data;
the blocking POLICY lives here since it's a strategy decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..config import CatalystFilterConfig
from ..features.news import recent_catalysts


@dataclass
class CatalystStatus:
    aggressive_allowed: bool
    standard_allowed: bool
    conservative_allowed: str  # "yes" | "no" | "only_if_strong_confirmation"
    major_positive: bool
    news_confidence: str  # "KNOWN" | "UNKNOWN"
    requires_manual_review: bool
    reasons: list[str] = field(default_factory=list)


def check_catalyst(
    catalyst_df: pd.DataFrame | None,
    symbol: str,
    as_of_ms: int,
    cfg: CatalystFilterConfig,
    is_major_listing_event: bool = False,
) -> CatalystStatus:
    """spec section 10. catalyst_df=None means no catalyst data source was
    loaded at all (distinct from "loaded but empty") - treated conservatively:
    news_confidence='UNKNOWN', aggressive/standard blocked, conservative only
    with strong confirmation (spec: "снизить max position size, повысить
    required score").
    """
    if not cfg.enabled:
        return CatalystStatus(True, True, "yes", False, "KNOWN", False, ["catalyst_filter disabled in config"])

    if catalyst_df is None:
        return CatalystStatus(
            aggressive_allowed=False,
            standard_allowed=False,
            conservative_allowed="only_if_strong_confirmation",
            major_positive=False,
            news_confidence="UNKNOWN",
            requires_manual_review=False,
            reasons=["no catalyst data source loaded - treated conservatively per spec section 10"],
        )

    recent_agg = recent_catalysts(catalyst_df, symbol, as_of_ms, cfg.block_aggressive_if_positive_news_within_hours, "positive")
    recent_std = recent_catalysts(catalyst_df, symbol, as_of_ms, cfg.block_standard_if_positive_news_within_hours, "positive")
    recent_cons_block = recent_catalysts(catalyst_df, symbol, as_of_ms, cfg.allow_conservative_after_hours, "positive")

    major_positive = not recent_agg.empty
    requires_manual_review = (
        cfg.require_manual_review_for_major_listing and is_major_listing_event and major_positive
    )
    reasons = (
        [f"positive catalyst within {cfg.block_aggressive_if_positive_news_within_hours}h -> aggressive/standard blocked"]
        if major_positive
        else ["no major positive catalyst found in lookback windows"]
    )

    return CatalystStatus(
        aggressive_allowed=recent_agg.empty,
        standard_allowed=recent_std.empty,
        conservative_allowed="no" if not recent_cons_block.empty else "yes",
        major_positive=major_positive,
        news_confidence="KNOWN",
        requires_manual_review=requires_manual_review,
        reasons=reasons,
    )
