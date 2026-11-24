"""
Direct refinement of pair correlations against diffuse scattering.

Ported from discordant-parsimony: the diffuse intensity is linear in
symmetry-reduced pair correlations :math:`C_{ij}(\\mathbf{R})`, which are
refined by (constrained) least squares with exact bin averaging and
held-out model selection. No supercell or Monte Carlo is involved.
"""

from rmc_discord.pairs.structure import Structure, Sites
from rmc_discord.pairs.pairterms import (
    PairTerms,
    pair_terms_from_covariance,
    thermal_laue_terms,
)
from rmc_discord.pairs.design import evaluate

__all__ = [
    "Structure",
    "Sites",
    "PairTerms",
    "pair_terms_from_covariance",
    "thermal_laue_terms",
    "evaluate",
]
