# -*- coding: utf-8 -*-
"""Economic decision layer: Resource Value Score per cell, Excavation Priority Map,
Value of Information (survey-guided vs blind excavation), Monte Carlo propagation over
price/cost/recovery uncertainty, break-even survey cost, and per-sensor VOI attribution.

Every price, cost and recovery-fraction range is an ASSUMPTION (labelled "illustrative" in
the source proposal); they are stated once in ECON_ASSUMPTIONS below and swept in the Monte
Carlo rather than treated as point estimates.

    python landmap_econ.py
"""
import json

import numpy as np

import landmap_synth as LS
from landmap_model import make_classifier, spatial_split

SEED = 20260926
N_MC = 20000

# ------------------------------------------------------------------ economic assumptions
# Secondary-material price ($/tonne recovered), recovery fraction (mass actually recoverable
# and saleable after sorting/processing) and per-tonne excavation+processing cost, by class.
# All ASSUMPTIONS, deliberately labelled illustrative; ranges are sampled uniformly.
ECON = {
    "price_usd_per_t":   {"metal": (250, 550), "plastic": (60, 180), "glass": (10, 40),
                          "organic": (0, 15), "cd": (5, 25), "inert": (0, 5)},
    "recovery_fraction": {"metal": (0.55, 0.85), "plastic": (0.15, 0.40), "glass": (0.10, 0.30),
                          "organic": (0.0, 0.05), "cd": (0.20, 0.45), "inert": (0.0, 0.02)},
    "mass_per_cell_t": (0.9, 1.6),           # tonnes of waste per 1 m^2 x ~3 m cell, illustrative
    "excavation_cost_usd_per_t": (18, 32),
    "processing_cost_usd_per_t": (12, 28),
    "survey_cost_usd_total": (400, 1200),    # cost of running the multimodal survey over the site
}


def sample_econ(n, rng):
    s = {}
    for k, v in ECON.items():
        if isinstance(v, dict):
            s[k] = {c: rng.uniform(*rng_, n) for c, rng_ in v.items()}
        else:
            s[k] = rng.uniform(*v, n)
    return s


def resource_value_score(proba, econ_sample, i):
    """Expected net value per cell (n_cells, n_classes) given classifier probabilities and one
    Monte Carlo draw i of the economic parameters."""
    n_cells, n_classes = proba.shape
    mass = econ_sample["mass_per_cell_t"][i]
    ex_cost = econ_sample["excavation_cost_usd_per_t"][i]
    pr_cost = econ_sample["processing_cost_usd_per_t"][i]
    rev = np.zeros(n_cells)
    for c, cls in enumerate(LS.CLASSES):
        price = econ_sample["price_usd_per_t"][cls][i]
        rec = econ_sample["recovery_fraction"][cls][i]
        rev += proba[:, c] * mass * rec * price
    cost = mass * (ex_cost + pr_cost)
    return rev - cost                      # net value per cell if excavated


def true_value(dominant, econ_sample, i):
    """The realised (not expected) net value per cell, given the TRUE material -- used to
    score decisions against ground truth for VOI."""
    n_cells = len(dominant)
    mass = econ_sample["mass_per_cell_t"][i]
    ex_cost = econ_sample["excavation_cost_usd_per_t"][i]
    pr_cost = econ_sample["processing_cost_usd_per_t"][i]
    price = np.array([econ_sample["price_usd_per_t"][LS.CLASSES[m]][i] for m in dominant])
    rec = np.array([econ_sample["recovery_fraction"][LS.CLASSES[m]][i] for m in dominant])
    rev = mass * rec * price
    cost = mass * (ex_cost + pr_cost)
    return rev - cost


def voi_for_survey(proba, dominant, econ_sample, i):
    """VOI = E[value | survey-guided decision] - E[value | best blind decision] - survey cost.
    The survey-guided decision excavates cell c iff its predicted expected value > 0; the blind
    decision excavates every cell iff the population-average expected value > 0 (the best an
    operator can do with no cell-level information), else excavates none."""
    rvs = resource_value_score(proba, econ_sample, i)
    tv = true_value(dominant, econ_sample, i)
    decide_survey = rvs > 0
    ev_survey = float(tv[decide_survey].sum()) if decide_survey.any() else 0.0
    mean_ev_all = float(tv.mean())
    ev_blind = float(tv.sum()) if mean_ev_all > 0 else 0.0
    return ev_survey - ev_blind - econ_sample["survey_cost_usd_total"][i], ev_survey, ev_blind


def monte_carlo_voi(proba, dominant, n_mc=N_MC, seed=SEED):
    rng = np.random.default_rng(seed)
    econ_sample = sample_econ(n_mc, rng)
    voi, ev_s, ev_b = np.zeros(n_mc), np.zeros(n_mc), np.zeros(n_mc)
    for i in range(n_mc):
        voi[i], ev_s[i], ev_b[i] = voi_for_survey(proba, dominant, econ_sample, i)
    return voi, ev_s, ev_b, econ_sample


def breakeven_survey_cost(proba, dominant, n_mc=N_MC, seed=SEED):
    """C_s* = EV_survey - EV_blind (median over the non-survey-cost draws): the maximum a
    rational operator should pay for the survey, before subtracting the assumed survey cost."""
    rng = np.random.default_rng(seed + 1)
    econ_sample = sample_econ(n_mc, rng)
    econ_sample["survey_cost_usd_total"] = np.zeros(n_mc)      # exclude survey cost from this figure
    diffs = np.zeros(n_mc)
    for i in range(n_mc):
        _, ev_s, ev_b = voi_for_survey(proba, dominant, econ_sample, i)
        diffs[i] = ev_s - ev_b
    return diffs


def sensor_value(sensor_sets_proba, dominant, n_mc=4000, seed=SEED):
    """VOI_j = VOI_all - VOI_without_j: the incremental economic value each sensor channel
    contributes, holding the same economic Monte Carlo draws across sensor configurations for
    a fair, paired comparison."""
    rng = np.random.default_rng(seed + 2)
    econ_sample = sample_econ(n_mc, rng)
    out = {}
    for name, proba in sensor_sets_proba.items():
        voi = np.zeros(n_mc)
        for i in range(n_mc):
            voi[i], _, _ = voi_for_survey(proba, dominant, econ_sample, i)
        out[name] = voi
    return out


def get_holdout_proba(kind="rf", sensors=None, seed=SEED, block=8):
    """Train on one representative synthetic survey (spatial holdout) and return the held-out
    region's predicted probabilities and true dominant material, for use as the VOI test set."""
    sensors = sensors or LS.SENSORS
    surv = LS.generate(seed=seed)
    grid = surv["grid"]
    Xf, y, haz, feat = LS.to_table(surv, sensors=sensors)
    mask = spatial_split(grid, block=block, seed=seed)
    clf = make_classifier(kind)
    clf.fit(Xf[~mask], y[~mask])
    proba = clf.predict_proba(Xf[mask])
    return proba, y[mask], surv, mask


def summarize(a):
    return {"mean": round(float(np.mean(a)), 2), "median": round(float(np.median(a)), 2),
            "p05": round(float(np.percentile(a, 5)), 2), "p95": round(float(np.percentile(a, 95)), 2),
            "p_positive": round(float(np.mean(np.array(a) > 0)), 4)}


N_REP_ECON = 20        # independent synthetic surveys, to average out single-split noise in VOI


def main():
    res = {"seed": SEED, "n_mc": N_MC, "n_rep_econ": N_REP_ECON, "econ_assumptions": ECON}

    # ---- headline VOI and break-even survey cost, on one representative survey (reported
    # together with the multi-repeat average below, so a reader sees both a single concrete
    # example and the averaged result)
    proba, y_true, surv, mask = get_holdout_proba(kind="rf", seed=SEED)
    res["holdout_n_cells"] = int(len(y_true))
    voi, ev_s, ev_b, _ = monte_carlo_voi(proba, y_true)
    res["voi_example"] = summarize(voi)
    res["ev_survey_example"] = summarize(ev_s)
    res["ev_blind_example"] = summarize(ev_b)
    diffs = breakeven_survey_cost(proba, y_true)
    res["breakeven_survey_cost_usd_example"] = summarize(diffs)
    print("VOI (single example survey):", res["voi_example"])
    print("Break-even survey cost (example):", res["breakeven_survey_cost_usd_example"])

    # ---- averaged over N_REP_ECON independent surveys
    voi_means, be_means = [], []
    ablation_names = ["all"] + ["minus_" + s for s in LS.SENSORS] + ["plus_nir"]
    incr_by_rep = {s: [] for s in LS.SENSORS}
    nir_delta_by_rep = []
    for rep in range(N_REP_ECON):
        seed_r = SEED + 1000 + rep
        p_all, y_r, _, _ = get_holdout_proba(kind="rf", seed=seed_r)
        v, _, _, _ = monte_carlo_voi(p_all, y_r, n_mc=3000, seed=seed_r)
        voi_means.append(float(np.mean(v)))
        d = breakeven_survey_cost(p_all, y_r, n_mc=3000, seed=seed_r)
        be_means.append(float(np.mean(d)))

        econ_sample = sample_econ(2500, np.random.default_rng(seed_r + 5))
        def mean_voi(proba_):
            vv = np.zeros(2500)
            for i in range(2500):
                vv[i], _, _ = voi_for_survey(proba_, y_r, econ_sample, i)
            return float(np.mean(vv))
        voi_all_r = mean_voi(p_all)
        for s in LS.SENSORS:
            p_wo, y_wo, _, _ = get_holdout_proba(kind="rf", sensors=[x for x in LS.SENSORS if x != s], seed=seed_r)
            assert np.array_equal(y_wo, y_r)
            incr_by_rep[s].append(voi_all_r - mean_voi(p_wo))
        p_nir, y_nir, _, _ = get_holdout_proba(kind="rf", sensors=LS.SENSORS + ["optical"], seed=seed_r)
        assert np.array_equal(y_nir, y_r)
        nir_delta_by_rep.append(mean_voi(p_nir) - voi_all_r)
        print("  repeat %d/%d done" % (rep + 1, N_REP_ECON))

    res["voi_averaged"] = summarize(voi_means)
    res["voi_averaged"]["sd"] = round(float(np.std(voi_means, ddof=1)), 2)
    res["voi_averaged_raw"] = [round(v, 2) for v in voi_means]
    res["breakeven_survey_cost_usd_averaged"] = summarize(be_means)
    res["breakeven_survey_cost_usd_averaged_raw"] = [round(b, 2) for b in be_means]
    res["sensor_value"] = {s: {"incremental_voi_mean": round(float(np.mean(v)), 1),
                               "incremental_voi_sd": round(float(np.std(v, ddof=1)), 1),
                               "p_positive": round(float(np.mean(np.array(v) > 0)), 3)}
                           for s, v in incr_by_rep.items()}
    res["nir_delta_voi"] = {"mean": round(float(np.mean(nir_delta_by_rep)), 1),
                            "sd": round(float(np.std(nir_delta_by_rep, ddof=1)), 1),
                            "p_positive": round(float(np.mean(np.array(nir_delta_by_rep) > 0)), 3)}
    print("\nVOI averaged over %d surveys:" % N_REP_ECON, res["voi_averaged"])
    print("Break-even survey cost averaged:", res["breakeven_survey_cost_usd_averaged"])
    print("\nsensor value (incremental VOI, averaged):")
    for s, v in res["sensor_value"].items():
        print(" ", s, v)
    print("\nNIR delta VOI:", res["nir_delta_voi"])

    json.dump(res, open("landmap_econ_results.json", "w"), indent=1)
    print("\nsaved landmap_econ_results.json")


if __name__ == "__main__":
    main()
