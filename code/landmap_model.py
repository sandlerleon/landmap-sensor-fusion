# -*- coding: utf-8 -*-
"""Classification pipeline: spatial vs random train/test splitting, three classifiers
(logistic regression, random forest, XGBoost), probability calibration, sensor ablation,
and the NIR-augmentation test.

    python landmap_model.py
"""
import json

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import landmap_synth as LS

SEED = 20260926
N_REPEATS = 40            # independent synthetic surveys, for confidence intervals on every metric


def spatial_split(grid, block=8, seed=0):
    """Hold out one contiguous block x block region entirely (rows and columns), the
    'non-leaky' protocol the white paper describes, chosen pseudo-randomly per repeat."""
    rng = np.random.default_rng(seed)
    r0 = rng.integers(0, grid - block + 1)
    c0 = rng.integers(0, grid - block + 1)
    mask_test = np.zeros((grid, grid), dtype=bool)
    mask_test[r0:r0 + block, c0:c0 + block] = True
    return mask_test.reshape(-1)


def random_split(grid, frac=None, seed=0, n_test=None):
    rng = np.random.default_rng(seed)
    n = grid * grid
    if n_test is None:
        n_test = int(round(frac * n))
    idx = rng.permutation(n)[:n_test]
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask


def make_classifier(kind):
    if kind == "logreg":
        return Pipeline([("impute", SimpleImputer(strategy="mean")), ("scale", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000))])
    if kind == "rf":
        return Pipeline([("impute", SimpleImputer(strategy="mean")),
                         ("clf", RandomForestClassifier(n_estimators=300, max_depth=None,
                                                        min_samples_leaf=2, random_state=SEED))])
    if kind == "xgb":
        return Pipeline([("impute", SimpleImputer(strategy="mean")),
                         ("clf", XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08,
                                               subsample=0.85, colsample_bytree=0.85,
                                               eval_metric="mlogloss", random_state=SEED,
                                               verbosity=0))])
    raise ValueError(kind)


def per_class_f1(y_true, y_pred, n_classes):
    return f1_score(y_true, y_pred, labels=list(range(n_classes)), average=None, zero_division=0)


def fit_eval(Xf, y, mask_test, kind, n_classes):
    Xtr, Xte = Xf[~mask_test], Xf[mask_test]
    ytr, yte = y[~mask_test], y[mask_test]
    clf = make_classifier(kind)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    proba = clf.predict_proba(Xte)
    f1 = per_class_f1(yte, pred, n_classes)
    macro = float(np.mean(f1))
    return {"f1_per_class": f1.tolist(), "macro_f1": macro, "proba": proba, "y_true": yte,
            "y_pred": pred, "n_train": len(ytr), "n_test": len(yte)}


def pooled_report(sensor_sets, kind, block=8, n_repeats=N_REPEATS, split="spatial", seed=SEED):
    """Pool (y_true, y_pred) across repeats before scoring, which is far less noisy for rare
    classes than averaging per-repeat F1 over small (single-block) test sets."""
    yt_all, yp_all = {name: [] for name in sensor_sets}, {name: [] for name in sensor_sets}
    for rep in range(n_repeats):
        surv = LS.generate(seed=seed + rep)
        grid = surv["grid"]
        mask_test = spatial_split(grid, block=block, seed=seed + rep) if split == "spatial" \
            else random_split(grid, frac=(block * block) / (grid * grid), seed=seed + rep)
        for name, sensors in sensor_sets.items():
            Xf, y, haz, feat = LS.to_table(surv, sensors=sensors)
            r = fit_eval(Xf, y, mask_test, kind, LS.N_CLASSES)
            yt_all[name].append(r["y_true"])
            yp_all[name].append(r["y_pred"])
    out = {}
    for name in sensor_sets:
        yt = np.concatenate(yt_all[name])
        yp = np.concatenate(yp_all[name])
        f1 = per_class_f1(yt, yp, LS.N_CLASSES)
        out[name] = {"per_class_f1": {LS.CLASSES[i]: round(float(f1[i]), 4) for i in range(LS.N_CLASSES)},
                     "macro_f1": round(float(f1.mean()), 4), "n_pooled": len(yt),
                     "support": {LS.CLASSES[i]: int((yt == i).sum()) for i in range(LS.N_CLASSES)}}
    return out


def calibration_metrics(y_true_bin_by_class, proba, n_classes):
    """One-vs-rest Brier score and a reliability curve, macro-averaged over classes."""
    briers = []
    for c in range(n_classes):
        yb = (y_true_bin_by_class == c).astype(int)
        briers.append(brier_score_loss(yb, proba[:, c]))
    return float(np.mean(briers))


def hazard_eval(haz_train_feats, haz, mask_test, seed):
    Xtr, Xte = haz_train_feats[~mask_test], haz_train_feats[mask_test]
    ytr, yte = haz[~mask_test], haz[mask_test]
    if ytr.sum() < 2 or (len(ytr) - ytr.sum()) < 2:
        return None
    clf = Pipeline([("impute", SimpleImputer(strategy="mean")),
                    ("clf", RandomForestClassifier(n_estimators=300, random_state=seed,
                                                   class_weight="balanced"))])
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    if yte.sum() == 0 or yte.sum() == len(yte):
        return None
    auc = roc_auc_score(yte, proba)
    pr_auc = average_precision_score(yte, proba)
    prec, rec, thr = precision_recall_curve(yte, proba)
    # a safety-oriented operating point: the lowest threshold achieving >=95% recall
    ok = rec[:-1] >= 0.95
    thr_choice = float(thr[ok][-1]) if ok.any() else float(thr[np.argmax(rec[:-1])])
    pred = (proba >= thr_choice).astype(int)
    tp = int(((pred == 1) & (yte == 1)).sum()); fp = int(((pred == 1) & (yte == 0)).sum())
    tn = int(((pred == 0) & (yte == 0)).sum()); fn = int(((pred == 0) & (yte == 1)).sum())
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    precision = precision_score(yte, pred, zero_division=0)
    return {"auc": float(auc), "pr_auc": float(pr_auc), "recall": float(recall_score(yte, pred)),
            "precision": float(precision), "specificity": float(specificity),
            "frac_flagged": float(pred.mean()), "operating_threshold": thr_choice,
            "n_hazard_test": int(yte.sum()), "n_test": len(yte),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def run_repeats(sensor_sets, block=8, n_repeats=N_REPEATS, split="spatial", classifiers=("logreg", "rf", "xgb")):
    out = {k: {kind: {"macro_f1": [], "per_class_f1": []} for kind in classifiers} for k in sensor_sets}
    haz_out = {"auc": [], "recall": []}
    calib_out = {kind: [] for kind in classifiers}
    for rep in range(n_repeats):
        surv = LS.generate(seed=SEED + rep)
        grid = surv["grid"]
        mask_test = spatial_split(grid, block=block, seed=SEED + rep) if split == "spatial" \
            else random_split(grid, frac=(block * block) / (grid * grid), seed=SEED + rep)
        for name, sensors in sensor_sets.items():
            Xf, y, haz, feat = LS.to_table(surv, sensors=sensors)
            for kind in classifiers:
                r = fit_eval(Xf, y, mask_test, kind, LS.N_CLASSES)
                out[name][kind]["macro_f1"].append(r["macro_f1"])
                out[name][kind]["per_class_f1"].append(r["f1_per_class"])
                if name == "baseline":
                    cb = calibration_metrics(r["y_true"], r["proba"], LS.N_CLASSES)
                    calib_out[kind].append(cb)
        # hazard classifier uses the baseline sensor features (not material-derived)
        Xf, y, haz, feat = LS.to_table(surv, sensors=LS.SENSORS)
        hr = hazard_eval(Xf, haz, mask_test, SEED + rep)
        if hr:
            haz_out["auc"].append(hr["auc"])
            haz_out["recall"].append(hr["recall"])
    return out, haz_out, calib_out


def distance_decay(kind="rf", n_repeats=15, block_sizes=(4, 6, 8, 10, 12, 14, 16, 20), seed=SEED):
    """For spatial blocks of varying size (and so varying distance from each held-out cell to
    its nearest training cell), pool (distance, correct) pairs across repeats and block sizes,
    then bin by distance. This is the standard spatial-cross-validation diagnostic (error as a
    function of distance to the nearest training observation) and does not depend on any one
    choice of held-out fraction."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed + 777)
    dists, corrects = [], []
    for rep in range(n_repeats):
        surv = LS.generate(seed=seed + rep)
        grid = surv["grid"]
        Xf, y, haz, feat = LS.to_table(surv, sensors=LS.SENSORS)
        coords = np.column_stack([surv["rows"].reshape(-1), surv["cols"].reshape(-1)]).astype(float)
        for b in block_sizes:
            r0 = rng.integers(0, grid - b + 1)
            c0 = rng.integers(0, grid - b + 1)
            mask = np.zeros((grid, grid), dtype=bool)
            mask[r0:r0 + b, c0:c0 + b] = True
            mask = mask.reshape(-1)
            clf = make_classifier(kind)
            clf.fit(Xf[~mask], y[~mask])
            pred = clf.predict(Xf[mask])
            correct = (pred == y[mask]).astype(int)
            tree = cKDTree(coords[~mask])
            d, _ = tree.query(coords[mask], k=1)
            dists.append(d)
            corrects.append(correct)
    d = np.concatenate(dists)
    c = np.concatenate(corrects)
    bin_edges = [0, 1, 2, 3, 4, 5, 7, 9, 100]
    curve = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        m = (d >= lo) & (d < hi)
        if m.sum() > 0:
            curve.append({"dist_lo": lo, "dist_hi": hi if hi < 100 else None,
                         "n": int(m.sum()), "accuracy": round(float(c[m].mean()), 4)})
    return {"curve": curve, "overall_accuracy": round(float(c.mean()), 4), "n_total": len(c)}


def summarize(vals):
    a = np.array(vals)
    return {"mean": round(float(a.mean()), 4), "sd": round(float(a.std()), 4),
            "p05": round(float(np.percentile(a, 5)), 4), "p95": round(float(np.percentile(a, 95)), 4)}


def main():
    res = {"seed": SEED, "n_repeats": N_REPEATS, "grid": LS.GRID, "block": 8, "classes": LS.CLASSES}

    # ---- 1. spatial vs random split, baseline six sensors, all three classifiers
    sensor_sets = {"baseline": LS.SENSORS}
    sp, haz_sp, calib_sp = run_repeats(sensor_sets, split="spatial")
    rd, haz_rd, calib_rd = run_repeats(sensor_sets, split="random")
    res["split_comparison"] = {}
    for kind in ("logreg", "rf", "xgb"):
        a = np.array(sp["baseline"][kind]["macro_f1"])
        b = np.array(rd["baseline"][kind]["macro_f1"])
        diff = b - a                                     # random minus spatial, paired by seed
        res["split_comparison"][kind] = {
            "spatial_macro_f1": summarize(a),
            "random_macro_f1": summarize(b),
            "paired_diff_random_minus_spatial": {
                "mean": round(float(diff.mean()), 4), "sd": round(float(diff.std(ddof=1)), 4),
                "ci95": [round(float(diff.mean() - 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))), 4),
                        round(float(diff.mean() + 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))), 4)]},
        }
    for kind in ("logreg", "rf", "xgb"):
        pc_sp = np.array(sp["baseline"][kind]["per_class_f1"])
        pc_rd = np.array(rd["baseline"][kind]["per_class_f1"])
        res["split_comparison"][kind]["per_class_f1_spatial"] = {
            LS.CLASSES[i]: summarize(pc_sp[:, i]) for i in range(LS.N_CLASSES)}
        res["split_comparison"][kind]["per_class_f1_random"] = {
            LS.CLASSES[i]: summarize(pc_rd[:, i]) for i in range(LS.N_CLASSES)}

    res["pooled_baseline"] = {kind: pooled_report({"baseline": LS.SENSORS}, kind)["baseline"]
                              for kind in ("logreg", "rf", "xgb")}
    res["distance_decay"] = distance_decay(kind="rf")
    res["calibration"] = {kind: summarize(calib_sp[kind]) for kind in ("logreg", "rf", "xgb")}
    res["hazard"] = {"auc": summarize(haz_sp["auc"]), "recall": summarize(haz_sp["recall"])}
    best_kind = min(("logreg", "rf", "xgb"),
                    key=lambda k: -np.mean(res["split_comparison"][k]["spatial_macro_f1"]["mean"]))
    best_kind = max(("logreg", "rf", "xgb"),
                    key=lambda k: res["split_comparison"][k]["spatial_macro_f1"]["mean"])
    res["best_classifier"] = best_kind

    # ---- 2. sensor ablation (best classifier, spatial split)
    ablation_sets = {"all": LS.SENSORS}
    for s in LS.SENSORS:
        ablation_sets["all_minus_" + s] = [x for x in LS.SENSORS if x != s]
    ab, _, _ = run_repeats(ablation_sets, split="spatial", classifiers=(best_kind,))
    res["ablation"] = {name: {"macro_f1": summarize(v[best_kind]["macro_f1"])}
                       for name, v in ab.items()}
    res["ablation_pooled"] = pooled_report(ablation_sets, best_kind)

    # ---- 3. NIR augmentation
    nir_sets = {"baseline": LS.SENSORS, "plus_nir": LS.SENSORS + ["optical"]}
    nir, _, _ = run_repeats(nir_sets, split="spatial", classifiers=(best_kind,))
    res["nir_augmentation"] = {}
    for name in nir_sets:
        pc = np.array(nir[name][best_kind]["per_class_f1"])
        res["nir_augmentation"][name] = {
            "macro_f1": summarize(nir[name][best_kind]["macro_f1"]),
            "per_class_f1": {LS.CLASSES[i]: summarize(pc[:, i]) for i in range(LS.N_CLASSES)}}
    res["nir_augmentation_pooled"] = pooled_report(nir_sets, best_kind)

    json.dump(res, open("landmap_model_results.json", "w"), indent=1)
    print("best classifier:", best_kind)
    for kind in ("logreg", "rf", "xgb"):
        print(kind, "spatial", res["split_comparison"][kind]["spatial_macro_f1"],
              "random", res["split_comparison"][kind]["random_macro_f1"])
    print("hazard auc", res["hazard"]["auc"], "recall", res["hazard"]["recall"])
    print("calibration (Brier)", res["calibration"])
    print("\nablation:")
    for k, v in res["ablation"].items():
        print(" ", k, v["macro_f1"])
    print("\nNIR:")
    for k, v in res["nir_augmentation"].items():
        print(" ", k, v["macro_f1"], "plastic", v["per_class_f1"]["plastic"], "glass", v["per_class_f1"]["glass"])


if __name__ == "__main__":
    main()
