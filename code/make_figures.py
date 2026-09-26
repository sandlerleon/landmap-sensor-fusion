# -*- coding: utf-8 -*-
"""Figures 1-9 for the Environmental Monitoring and Assessment manuscript. 300 dpi PNG,
600 dpi TIFF for separate upload.

    python make_figures.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

import landmap_synth as LS

R = json.load(open("landmap_model_results.json"))
E = json.load(open("landmap_econ_results.json"))
try:
    RB = json.load(open("landmap_robustness_results.json"))
except FileNotFoundError:
    RB = None
OUT = "figures"
os.makedirs(OUT, exist_ok=True)
DPI = 300
plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9.5,
                     "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "font.family": "DejaVu Sans", "axes.linewidth": 0.8})
BLUE, RED, GREEN, GREY, ORANGE, PURPLE = "#1b6ca8", "#c1553b", "#3f8f4a", "#7a7a7a", "#c98a1b", "#6a4c93"
CLASS_COLORS = {"metal": "#8a8a8a", "plastic": "#4fc3f7", "glass": "#66bb6a", "organic": "#8d6e63",
                "cd": "#c98a1b", "inert": "#e0e0e0"}


def save(fig, name):
    png = os.path.join(OUT, name + ".png")
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    tif = os.path.join(OUT, name + ".tif")
    fig.savefig(tif, dpi=600, bbox_inches="tight")
    plt.close(fig)
    Image.open(tif).convert("RGB").save(tif, dpi=(600, 600), compression="tiff_lzw")
    print(name)


def box(ax, x, y, w, h, text, col, fs=7.2, ls="-"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.04",
                                lw=1.1, ls=ls, edgecolor=col, facecolor=col + "18"))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs)


def arrow(ax, a, b, col="#555555", ls="-", rad=0.0, lw=1.0, style="-|>"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=9, lw=lw, color=col,
                                 ls=ls, shrinkA=3, shrinkB=3, connectionstyle="arc3,rad=%.2f" % rad))


# ------------------------------------------------------------------ Figure 1: architecture
fig, ax = plt.subplots(figsize=(7.6, 4.4))
ax.set_xlim(0, 12.5); ax.set_ylim(0, 6.6); ax.axis("off")
box(ax, 1.8, 5.4, 3.0, 0.7, "LiDAR/RGB + LWIR thermal\n(surface, topography)", BLUE)
box(ax, 1.8, 4.4, 3.0, 0.7, "Magnetometer", BLUE)
box(ax, 1.8, 3.4, 3.0, 0.7, "EMI (conductivity, moisture)", BLUE)
box(ax, 1.8, 2.4, 3.0, 0.7, "GPR (density, depth)", BLUE)
box(ax, 1.8, 1.4, 3.0, 0.7, "Methane/VOC gas array", BLUE)
box(ax, 1.8, 0.4, 3.0, 0.6, "RTK-GNSS + IMU georeferencing", GREY, fs=6.6)
box(ax, 6.2, 3.4, 3.0, 1.6, "Per-cell fusion classifier\n(logistic regression, random forest,\nXGBoost)\nspatial train/test protocol", PURPLE)
box(ax, 10.5, 4.6, 3.4, 1.1, "Material probabilities\n(metal, plastic, glass,\norganic, C&D, inert)", GREEN)
box(ax, 10.5, 2.2, 3.4, 1.1, "Gas-hazard probability\n(safety-oriented\noperating point)", RED)
box(ax, 6.2, 0.6, 3.0, 0.7, "Economic decision layer\n(Resource Value Score, VOI)", ORANGE)
box(ax, 10.5, 0.6, 3.4, 0.7, "Excavation Priority Map\n+ uncertainty", ORANGE)
for y in (5.4, 4.4, 3.4, 2.4, 1.4):
    arrow(ax, (3.3, y), (4.7, 3.4))
arrow(ax, (7.7, 3.9), (8.8, 4.6))
arrow(ax, (7.7, 2.9), (8.8, 2.2))
arrow(ax, (10.5, 4.05), (6.2, 0.95), rad=-0.15, col=ORANGE)
arrow(ax, (10.5, 1.65), (6.2, 0.95), rad=0.15, col=ORANGE)
arrow(ax, (7.7, 0.6), (8.8, 0.6), col=ORANGE)
ax.text(6.2, 5.9, "LAND-MAP: multimodal sensing to excavation-decision pipeline",
        ha="center", fontsize=10.5, fontweight="bold")
save(fig, "figure1_architecture")

# ------------------------------------------------------------------ Figure 2: example survey maps
surv = LS.generate(seed=20260926)
fig, axs = plt.subplots(1, 3, figsize=(9.2, 3.2))
cmap_mat = matplotlib.colors.ListedColormap([CLASS_COLORS[c] for c in LS.CLASSES])
im0 = axs[0].imshow(surv["dominant"], cmap=cmap_mat, vmin=-0.5, vmax=5.5)
axs[0].set_title("A   True near-surface material", loc="left", fontweight="bold", fontsize=8.6)
cb = fig.colorbar(im0, ax=axs[0], ticks=range(6), fraction=0.046)
cb.ax.set_yticklabels(LS.CLASSES, fontsize=6.5)
im1 = axs[1].imshow(surv["X"]["mag"], cmap="magma")
axs[1].set_title("B   Magnetometer reading (raw)", loc="left", fontweight="bold", fontsize=8.6)
fig.colorbar(im1, ax=axs[1], fraction=0.046)
im2 = axs[2].imshow(surv["hazard"].astype(float), cmap="Reds", vmin=0, vmax=1)
axs[2].set_title("C   Gas-hazard truth", loc="left", fontweight="bold", fontsize=8.6)
fig.colorbar(im2, ax=axs[2], fraction=0.046)
for a in axs:
    a.set_xticks([]); a.set_yticks([])
fig.tight_layout()
save(fig, "figure2_synthetic_survey")

# ------------------------------------------------------------------ Figure 3: per-class F1, three classifiers
fig, ax = plt.subplots(figsize=(6.6, 3.6))
kinds = ["logreg", "rf", "xgb"]
klabels = {"logreg": "Logistic regression", "rf": "Random forest", "xgb": "XGBoost"}
x = np.arange(LS.N_CLASSES)
w = 0.25
for i, k in enumerate(kinds):
    vals = [R["pooled_baseline"][k]["per_class_f1"][c] for c in LS.CLASSES]
    ax.bar(x + (i - 1) * w, vals, w, label=klabels[k], color=[BLUE, GREEN, ORANGE][i])
ax.set_xticks(x); ax.set_xticklabels(LS.CLASSES)
ax.set_ylabel("F1 score (pooled, spatial holdout)")
ax.set_ylim(0, 1.05)
ax.legend(frameon=False)
ax.set_title("Per-class classification performance, six-sensor baseline", loc="left", fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
save(fig, "figure3_per_class_f1")

# ------------------------------------------------------------------ Figure 4: spatial vs random split + distance decay
fig, axs = plt.subplots(1, 2, figsize=(8.8, 3.4))
kinds_lab = ["Logistic\nregression", "Random\nforest", "XGBoost"]
sp_means = [R["split_comparison"][k]["spatial_macro_f1"]["mean"] for k in kinds]
sp_sd = [R["split_comparison"][k]["spatial_macro_f1"]["sd"] for k in kinds]
rd_means = [R["split_comparison"][k]["random_macro_f1"]["mean"] for k in kinds]
rd_sd = [R["split_comparison"][k]["random_macro_f1"]["sd"] for k in kinds]
xp = np.arange(3)
axs[0].bar(xp - 0.18, sp_means, 0.32, yerr=sp_sd, label="Spatial block holdout", color=BLUE, capsize=3)
axs[0].bar(xp + 0.18, rd_means, 0.32, yerr=rd_sd, label="Random per-cell holdout", color=RED, capsize=3)
axs[0].set_xticks(xp); axs[0].set_xticklabels(kinds_lab, fontsize=7.4)
axs[0].set_ylabel("Macro-F1 (mean over %d repeats)" % R["n_repeats"])
axs[0].set_title("A   Spatial vs. random validation", loc="left", fontweight="bold", fontsize=8.8)
axs[0].legend(frameon=False, fontsize=7)
axs[0].set_ylim(0, 0.85)
dd = R["distance_decay"]["curve"]
xs = [(row["dist_lo"] + (row["dist_hi"] if row["dist_hi"] else row["dist_lo"] + 3)) / 2 for row in dd]
acc = [row["accuracy"] for row in dd]
ns = [row["n"] for row in dd]
axs[1].plot(xs, acc, "o-", color=PURPLE)
for xx, yy, nn in zip(xs, acc, ns):
    axs[1].annotate("n=%d" % nn, (xx, yy), textcoords="offset points", xytext=(0, 6), fontsize=6, ha="center")
axs[1].axhline(R["distance_decay"]["overall_accuracy"], color=GREY, ls="--", lw=0.9)
axs[1].set_xlabel("distance to nearest training cell (m)")
axs[1].set_ylabel("held-out accuracy")
axs[1].set_title("B   Accuracy vs. distance to training data", loc="left", fontweight="bold", fontsize=8.8)
axs[1].set_ylim(0.5, 0.85)
for a in axs:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
save(fig, "figure4_spatial_validation")

# ------------------------------------------------------------------ Figure 5: sensor ablation + NIR
fig, axs = plt.subplots(1, 2, figsize=(8.8, 3.4))
names = ["all"] + ["all_minus_" + s for s in LS.SENSORS]
labs = ["all 5\nsensors"] + ["−" + s for s in LS.SENSORS]
vals = [R["ablation_pooled"][n]["macro_f1"] for n in names]
cols = [GREEN] + [RED if vals[0] - v > 0.02 else ORANGE for v in vals[1:]]
axs[0].bar(range(len(names)), vals, color=cols)
axs[0].set_xticks(range(len(names))); axs[0].set_xticklabels(labs, fontsize=7)
axs[0].set_ylabel("pooled macro-F1")
axs[0].set_ylim(0, 0.75)
axs[0].set_title("A   Sensor ablation (classification)", loc="left", fontweight="bold", fontsize=8.8)
sv = R2 = E["sensor_value"]
names2 = LS.SENSORS
incr = [sv[s]["incremental_voi_mean"] for s in names2]
sds = [sv[s]["incremental_voi_sd"] for s in names2]
pp = [sv[s]["p_positive"] for s in names2]
cols2 = [BLUE if p >= 0.9 else (GREY if 0.3 < p < 0.9 else RED) for p in pp]
axs[1].bar(range(len(names2)), incr, yerr=sds, color=cols2, capsize=3)
axs[1].axhline(0, color="#333333", lw=0.8)
axs[1].set_xticks(range(len(names2))); axs[1].set_xticklabels(names2)
for i, p in enumerate(pp):
    axs[1].annotate("P>0=%.2f" % p, (i, incr[i] + sds[i] * np.sign(incr[i] + 1e-9) + 8),
                    ha="center", fontsize=6.2)
axs[1].set_ylabel("incremental VOI (USD, mean ± SD over repeats)")
axs[1].set_title("B   Sensor value (economic)", loc="left", fontweight="bold", fontsize=8.8)
for a in axs:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
save(fig, "figure5_ablation_sensor_value")

# ------------------------------------------------------------------ Figure 6: NIR augmentation
fig, ax = plt.subplots(figsize=(6.0, 3.6))
n = R["nir_augmentation_pooled"]
classes_show = ["plastic", "glass", "metal", "organic", "cd", "inert"]
xp = np.arange(len(classes_show))
base_v = [n["baseline"]["per_class_f1"][c] for c in classes_show]
nir_v = [n["plus_nir"]["per_class_f1"][c] for c in classes_show]
ax.bar(xp - 0.18, base_v, 0.32, label="6-sensor baseline", color=BLUE)
ax.bar(xp + 0.18, nir_v, 0.32, label="+ NIR/optical channel", color=GREEN)
ax.set_xticks(xp); ax.set_xticklabels(classes_show)
ax.set_ylabel("pooled F1")
ax.legend(frameon=False)
ax.set_title("Effect of adding an NIR/optical channel", loc="left", fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
save(fig, "figure6_nir_augmentation")

# ------------------------------------------------------------------ Figure 7: calibration + hazard ROC-style
fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.4))
briers = [R["calibration"][k]["mean"] for k in kinds]
briers_sd = [R["calibration"][k]["sd"] for k in kinds]
axs[0].bar(range(3), briers, yerr=briers_sd, color=[BLUE, GREEN, ORANGE], capsize=3)
axs[0].set_xticks(range(3)); axs[0].set_xticklabels(kinds_lab, fontsize=7.4)
axs[0].set_ylabel("mean one-vs-rest Brier score (lower is better)")
axs[0].set_title("A   Probability calibration", loc="left", fontweight="bold", fontsize=8.8)
haz = R["hazard"]
axs[1].bar([0, 1], [haz["auc"]["mean"], haz["recall"]["mean"]],
          yerr=[haz["auc"]["sd"], haz["recall"]["sd"]], color=[PURPLE, RED], capsize=4, width=0.5)
axs[1].set_xticks([0, 1]); axs[1].set_xticklabels(["ROC-AUC", "Recall at safety\noperating point"])
axs[1].set_ylim(0, 1.05)
axs[1].set_title("B   Gas-hazard classifier", loc="left", fontweight="bold", fontsize=8.8)
for a in axs:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
save(fig, "figure7_calibration_hazard")

# ------------------------------------------------------------------ Figure 8: VOI distribution + breakeven
fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.4))
rng = np.random.default_rng(1)
voi_ex = E["voi_example"]
# reconstruct an illustrative distribution shape from summary stats for the plot (kde-like via samples)
import landmap_econ as LE
proba, y_true, s2, mask = LE.get_holdout_proba(kind="rf", seed=LE.SEED)
voi, ev_s, ev_b, _ = LE.monte_carlo_voi(proba, y_true, n_mc=8000, seed=LE.SEED)
axs[0].hist(voi, bins=60, color=BLUE, alpha=0.85)
axs[0].axvline(0, color=RED, lw=1.3, ls="--")
axs[0].set_xlabel("Value of Information (USD)")
axs[0].set_ylabel("Monte Carlo draws")
axs[0].set_title("A   VOI distribution (example survey)", loc="left", fontweight="bold", fontsize=8.8)
axs[0].text(0.98, 0.92, "P(VOI>0) = %.2f" % voi_ex["p_positive"], transform=axs[0].transAxes,
           ha="right", fontsize=7.6)
be = LE.breakeven_survey_cost(proba, y_true, n_mc=8000, seed=LE.SEED)
axs[1].hist(be, bins=60, color=GREEN, alpha=0.85)
axs[1].axvline(np.median(be), color="#333333", lw=1.1, ls=":")
axs[1].set_xlabel("break-even survey cost, C_s* (USD)")
axs[1].set_ylabel("Monte Carlo draws")
axs[1].set_title("B   Maximum justified survey cost", loc="left", fontweight="bold", fontsize=8.8)
for a in axs:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
save(fig, "figure8_voi_breakeven")

# ------------------------------------------------------------------ Figure 9: robustness (if available)
if RB is not None:
    fig, axs = plt.subplots(1, 2, figsize=(8.8, 3.4))
    cd = RB["combined_degradation"]
    lvl = [r["level"] for r in cd]
    f1v = [r["macro_f1_mean"] for r in cd]
    voiv = [r["voi_mean"] for r in cd]
    ax2 = axs[0].twinx()
    axs[0].plot(lvl, f1v, "o-", color=BLUE, label="macro-F1")
    ax2.plot(lvl, voiv, "s-", color=RED, label="mean VOI (USD)")
    ax2.axhline(0, color=RED, lw=0.8, ls=":")
    axs[0].set_xlabel("combined degradation level (0 = baseline)")
    axs[0].set_ylabel("macro-F1", color=BLUE)
    ax2.set_ylabel("mean VOI (USD)", color=RED)
    axs[0].set_title("A   Combined synthetic-to-real degradation", loc="left", fontweight="bold", fontsize=8.4)
    ov = RB["sweeps"]["overlap_shrink"]
    xs = [r["value"] for r in ov]
    f1o = [r["macro_f1_mean"] for r in ov]
    voio = [r["voi_mean"] for r in ov]
    ax3 = axs[1].twinx()
    axs[1].plot(xs, f1o, "o-", color=BLUE)
    ax3.plot(xs, voio, "s-", color=RED)
    ax3.axhline(0, color=RED, lw=0.8, ls=":")
    axs[1].set_xlabel("material-signature overlap shrinkage")
    axs[1].set_ylabel("macro-F1", color=BLUE)
    ax3.set_ylabel("mean VOI (USD)", color=RED)
    axs[1].set_title("B   Signature-overlap sensitivity", loc="left", fontweight="bold", fontsize=8.4)
    fig.tight_layout()
    save(fig, "figure9_robustness")
else:
    print("robustness results not yet available; figure9 skipped")
