"""Meta labeling functions used by SALUS."""

from __future__ import annotations

from sqlsalus_core.labeling.base import ABSTAIN, CORRECT, INCORRECT


# ---------------------------------------------------------------------------
# LF44: Majority of non-abstaining LFs vote INCORRECT -> INCORRECT
# ---------------------------------------------------------------------------

def lf_44(votes: list[int]) -> int:
    """Bipolar: majority vote INCORRECT -> INCORRECT; majority vote CORRECT -> CORRECT."""
    non_abstain = [v for v in votes if v != ABSTAIN]
    if not non_abstain:
        return ABSTAIN

    incorrect_count = sum(1 for v in non_abstain if v == INCORRECT)
    correct_count = sum(1 for v in non_abstain if v == CORRECT)

    if incorrect_count > correct_count:
        return INCORRECT
    if correct_count > incorrect_count:
        return CORRECT

    return ABSTAIN
