# -*- coding: utf-8 -*-
"""Vectorized VOI Monte Carlo (same formulas as landmap_econ.voi_for_survey/monte_carlo_voi,
just without its per-draw Python loop), used by the sensor-selection and decision-viability
maps, which each evaluate many (grid point x sensor-set x repeat) combinations and would be
too slow with the original loop-based implementation.
"""
import numpy as np

import landmap_synth as LS


def sample_econ_arrays(econ, n_mc, rng):
    mass = rng.uniform(*econ["mass_per_cell_t"], n_mc)
    ex = rng.uniform(*econ["excavation_cost_usd_per_t"], n_mc)
    pr = rng.uniform(*econ["processing_cost_usd_per_t"], n_mc)
    surv = rng.uniform(*econ["survey_cost_usd_total"], n_mc)
    n_classes = LS.N_CLASSES
    price = np.zeros((n_classes, n_mc))
    rec = np.zeros((n_classes, n_mc))
    for c, cls in enumerate(LS.CLASSES):
        price[c] = rng.uniform(*econ["price_usd_per_t"][cls], n_mc)
        rec[c] = rng.uniform(*econ["recovery_fraction"][cls], n_mc)
    return mass, ex, pr, surv, price, rec


def mean_voi_fast(proba, dominant, econ, n_mc, seed):
    """Returns the Monte Carlo VOI array (n_mc,); mean/median/P(>0) as needed by the caller."""
    rng = np.random.default_rng(seed)
    mass, ex, pr, surv, price, rec = sample_econ_arrays(econ, n_mc, rng)
    val_per_class = price * rec                      # (n_classes, n_mc)
    rev = (proba @ val_per_class) * mass[None, :]     # (n_cells, n_mc)
    cost = (mass * (ex + pr))[None, :]                # (1, n_mc)
    value_expected = rev - cost                       # (n_cells, n_mc)
    decide = value_expected > 0
    price_true = price[dominant, :]                   # (n_cells, n_mc)
    rec_true = rec[dominant, :]
    tv = mass[None, :] * rec_true * price_true - cost  # (n_cells, n_mc)
    ev_survey = np.where(decide, tv, 0.0).sum(axis=0)
    mean_tv = tv.mean(axis=0)
    ev_blind = np.where(mean_tv > 0, tv.sum(axis=0), 0.0)
    return ev_survey - ev_blind - surv
