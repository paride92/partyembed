from __future__ import annotations

from typing import List, Optional

import numpy as np

BASE_LEXICON = [
    # Left on economic axis
    ["affordable_housing", "decent_housing", "eradicate_poverty", "poverty",
     "gap_rich_poor", "wealthiest", "low_income", "inequality", "unequal",
     "workers", "minimum_wage", "unemployment", "unemployed", "protective_tariff",
     "redistribution", "redistribution_wealth", "safety_net", "social_security",
     "homelessness", "labor_unions", "labour_unions", "trade_unions", "working_classes"],
    # Right on economic axis
    ["decentralization", "bureaucracy", "business", "businesses", "creating_jobs",
     "job_creators", "free_enterprise", "free_trade", "debt_relief", "debt_reduction",
     "taxpayers", "taxpayers_money", "taxpayer_money", "commerce", "privatisation",
     "privatization", "competitive", "industry", "productivity", "deficit_reduction",
     "hard_working", "hardworking", "home_owners", "homeowners", "open_market",
     "free_market", "private_enterprise", "private_sector", "property_rights",
     "property_owners"],
    # Progressive on social axis
    ["minority_rights", "gay_lesbian", "affirmative_action", "employment_equity",
     "pay_equity", "racial_minorities", "racism", "gun_control", "minorities",
     "prochoice", "pro-choice", "civil_rights", "environment", "greenhouse_gas",
     "pollution", "climate_change", "child_care", "childcare", "planned_parenthood",
     "access_abortion"],
    # Conservative on social axis
    ["law_enforcement", "moral_fabric", "social_fabric", "moral_decay", "moral_values",
     "sentences", "tougher_sentences", "traditional_values", "tradition",
     "secure_borders", "illegal_immigrants", "illegal_immigration", "criminals",
     "fight_crime", "prolife", "pro-life", "sanctity_life", "unborn_child",
     "abortionist", "church"],
]


def _get_vectors(model, words: List[str], M: int) -> np.ndarray:
    vocab = model.wv.key_to_index
    valid = [w for w in words if w in vocab]
    if not valid:
        return np.zeros((1, M))
    mat = np.zeros((len(valid), M))
    for i, w in enumerate(valid):
        mat[i] = model.wv[w]
    return mat


def _proj_1d(vec: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    axis = right.mean(axis=0) - left.mean(axis=0)
    return float(np.dot(vec, axis))


def _proj_2d(
    vec: np.ndarray,
    left: np.ndarray, right: np.ndarray,
    down: np.ndarray, up: np.ndarray,
) -> tuple[float, float]:
    x_axis = right.mean(axis=0) - left.mean(axis=0)
    y_axis = up.mean(axis=0) - down.mean(axis=0)
    return float(np.dot(vec, x_axis)), float(np.dot(vec, y_axis))


def custom_projection_1D(
    z: np.ndarray,
    model,
    custom_lexicon: Optional[List[List[str]]] = None,
) -> np.ndarray:
    """Project document vectors onto a single ideological axis."""
    M = model.vector_size
    lex = custom_lexicon or [BASE_LEXICON[0] + BASE_LEXICON[2], BASE_LEXICON[1] + BASE_LEXICON[3]]
    if len(lex) != 2:
        raise ValueError("custom_lexicon for 1-D projection must have exactly 2 word lists.")
    xl, xr = [_get_vectors(model, words, M) for words in lex]
    return np.array([_proj_1d(row, xl, xr) for row in z])


def custom_projection_2D(
    z: np.ndarray,
    model,
    custom_lexicon: Optional[List[List[str]]] = None,
) -> np.ndarray:
    """Project document vectors onto two ideological axes."""
    M = model.vector_size
    lex = custom_lexicon or BASE_LEXICON
    if len(lex) != 4:
        raise ValueError("custom_lexicon for 2-D projection must have exactly 4 word lists.")
    xl, xr, yd, yu = [_get_vectors(model, words, M) for words in lex]
    projections = [_proj_2d(row, xl, xr, yd, yu) for row in z]
    return np.array(projections)
