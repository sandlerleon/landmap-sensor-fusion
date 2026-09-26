# -*- coding: utf-8 -*-
"""Synthetic landfill generator: a physically motivated latent-material field and five
material-response sensor channels (magnetometer, EMI, GPR, LWIR thermal, gas), each
with material-dependent signal, depth attenuation, moisture effects, sensor noise and
random dropout. A sixth, optional NIR/optical channel is added separately as the
proposed remedy for the plastic/glass confusion the source proposal reports.

Nothing here is measured. Every material-response coefficient is either taken from the
cited geophysical literature where a source gives an order of magnitude, or stated as an
assumption; the ASSUMPTIONS dict is the single place they are set, so the sensitivity and
robustness studies can perturb them systematically.

    python landmap_synth.py
"""
import json

import numpy as np

SEED = 20260926
GRID = 32                        # 32 x 32 cells at 1 m^2 each = 1024 m^2 (matches the proposal)
N_DEPTH = 3                       # coarse depth layers per cell (0-1, 1-2, 2-3 m)
CLASSES = ["metal", "plastic", "glass", "organic", "cd", "inert"]     # cd = construction & demolition
N_CLASSES = len(CLASSES)

# ------------------------------------------------------------------ material response table
# Each material's mean response per sensor channel, in arbitrary standardised units, before
# noise, depth attenuation and moisture effects are applied. Rows are ASSUMPTIONS motivated
# by the qualitative physics cited in the manuscript (magnetometer/EMI respond to metal;
# GPR responds to density/dielectric contrast; thermal and gas respond to organic
# decomposition; plastic and glass are close to inert in every channel except an optical one).
RESPONSE = {
    #              mag   emi   gpr  therm  gas  optical(NIR)
    "metal":    [ 3.0,  2.6,  2.2,  0.1,  0.05,  0.3],
    "plastic":  [ 0.05, 0.15, 0.7,  0.1,  0.05,  1.8],
    "glass":    [ 0.05, 0.10, 0.9,  0.1,  0.05,  1.2],
    "organic":  [ 0.05, 0.9,  0.4,  1.6,  2.4,   0.9],
    "cd":       [ 0.55, 0.75, 2.9,  0.15, 0.05,  0.6],   # rebar/metal fragments give cd a mag/EMI signature
    "inert":    [ 0.05, 0.2,  0.3,  0.1,  0.05,  0.4],
}
ASSUMPTIONS = {
    "depth_atten_per_m": {"mag": 0.35, "emi": 0.25, "gpr": 0.55, "therm": 0.7, "gas": 0.3, "optical": 5.0},
    # optical/NIR cannot see through even 0.1 m of cover -- it is a surface/near-surface channel
    "moisture_gpr_atten": 1.5,        # extra GPR attenuation per unit moisture (models conductive loss)
    "moisture_emi_gain": 1.3,         # EMI responds positively to moisture (conductivity)
    "noise_sd": {"mag": 0.30, "emi": 0.35, "gpr": 0.32, "therm": 0.30, "gas": 0.35, "optical": 0.25},
    # fraction of each sensor's noise variance that is spatially correlated over ~1.5 m
    # (instrument drift, footprint overlap, local calibration) rather than independent per cell;
    # this is what makes randomly split (as opposed to spatially blocked) test cells leak
    # information from their near neighbours in the training set
    "noise_spatial_corr_frac": 0.6,
    "noise_corr_scale_m": 1.6,
    "dropout_p": {"mag": 0.03, "emi": 0.03, "gpr": 0.08, "therm": 0.05, "gas": 0.10, "optical": 0.05},
    # dropout probabilities are higher for gas (drift/saturation) and GPR (siting/coupling issues)
    "class_prior": [0.10, 0.16, 0.10, 0.28, 0.16, 0.20],   # organic and inert dominate a landfill
    "patch_scale_m": 8.0,             # spatial correlation length of the material field
    "gas_hazard_base_rate": 0.12,     # baseline hazard-cell probability, before organic coupling
    "gas_hazard_organic_boost": 0.55,
}
SENSORS = ["mag", "emi", "gpr", "therm", "gas"]                  # the six-channel baseline (+ optical held out)
ALL_SENSORS = SENSORS + ["optical"]


def _random_field(rng, shape, scale):
    """A smooth 2-D Gaussian random field via spectral synthesis, used to give the material
    field, moisture field and depth field spatial correlation instead of i.i.d. noise."""
    h, w = shape
    kh = np.fft.fftfreq(h)[:, None]
    kw = np.fft.fftfreq(w)[None, :]
    k = np.sqrt(kh ** 2 + kw ** 2)
    k[0, 0] = 1e-6
    spec = rng.standard_normal((h, w)) + 1j * rng.standard_normal((h, w))
    spec *= np.exp(-(k * scale) ** 2 * 8.0)
    field = np.fft.ifft2(spec).real
    return (field - field.mean()) / (field.std() + 1e-9)


def generate(seed=SEED, grid=GRID, response=None, assumptions=None):
    rng = np.random.default_rng(seed)
    R = response or RESPONSE
    A = assumptions or ASSUMPTIONS
    n = grid * grid

    # ---- latent material field: correlated per depth layer, materials assigned by
    # thresholding a smooth random field against the cumulative class prior (gives patchy,
    # not salt-and-pepper, material zones -- landfill cells are dumped and compacted in lots)
    material = np.empty((N_DEPTH, grid, grid), dtype=int)
    for d in range(N_DEPTH):
        f = _random_field(rng, (grid, grid), A["patch_scale_m"] / grid)
        ranks = f.argsort(axis=None).argsort(axis=None).reshape(grid, grid) / (n - 1)
        cum = np.cumsum(A["class_prior"])
        material[d] = np.searchsorted(cum, ranks)
    dominant = material[0]                      # the classification target: near-surface material

    depth_to_dominant = 0.15 + 0.5 * (_random_field(rng, (grid, grid), 6.0 / grid) * 0.5 + 0.5)  # 0.15-0.65 m cover
    moisture = np.clip(0.5 + 0.4 * _random_field(rng, (grid, grid), 5.0 / grid), 0.0, 1.0)

    # ---- sensor responses: depth-attenuated material signal + moisture cross-terms + noise + dropout
    X, base_signal = {}, {}
    for s in ALL_SENSORS:
        idx = ALL_SENSORS.index(s)
        base = np.zeros((grid, grid))
        for d in range(N_DEPTH):
            depth_m = depth_to_dominant + d * 1.0
            atten = np.exp(-A["depth_atten_per_m"][s] * depth_m)
            resp = np.array([R[CLASSES[m]][idx] for m in range(N_CLASSES)])
            layer_signal = resp[material[d]] * atten
            weight = 1.0 if d == 0 else 0.22 ** d     # near-surface layer dominates the reading
            base += weight * layer_signal
        if s == "gpr":
            base = base * np.exp(-A["moisture_gpr_atten"] * moisture)
        if s == "emi":
            base = base * (1.0 + A["moisture_emi_gain"] * moisture)
        base_signal[s] = base
        sd = A["noise_sd"][s]
        cf = A.get("noise_spatial_corr_frac", 0.0)
        indep = rng.normal(0, sd * np.sqrt(1 - cf), (grid, grid))
        corr_field = _random_field(rng, (grid, grid), A.get("noise_corr_scale_m", 1.5) / grid)
        corr = corr_field * sd * np.sqrt(cf)
        noise = indep + corr
        signal = base + noise
        drop = rng.random((grid, grid)) < A["dropout_p"][s]
        signal = np.where(drop, np.nan, signal)
        X[s] = signal

    # ---- gas hazard label: a noisy, logistic-thresholded function of the TRUE (pre-noise)
    # decomposition-driven gas level, so hazard is by construction what an ideal gas sensor
    # would detect; the measured 'gas' channel differs from this truth only by its own
    # instrument noise and dropout, exactly as a real hazard label (independent gas testing)
    # would relate to a deployed sensor's reading.
    g = base_signal["gas"]
    haz_z = (g - np.quantile(g, 1 - A["gas_hazard_base_rate"] - 0.10)) / (g.std() + 1e-9) * 3.0
    p_hazard = 1.0 / (1.0 + np.exp(-haz_z))
    hazard = rng.random((grid, grid)) < p_hazard

    rows, cols = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
    return {
        "grid": grid, "rows": rows, "cols": cols, "dominant": dominant, "material_layers": material,
        "depth_to_dominant": depth_to_dominant, "moisture": moisture, "X": X, "hazard": hazard,
        "p_hazard": p_hazard, "classes": CLASSES, "sensors": ALL_SENSORS,
    }


def to_table(surv, sensors=None):
    """Flatten to a (n_cells, n_features) design matrix plus labels, for the given sensor
    subset (default: the six-channel baseline, excluding optical)."""
    sensors = sensors or SENSORS
    g = surv["grid"]
    cols = []
    for s in sensors:
        cols.append(surv["X"][s].reshape(-1))
    cols.append(surv["rows"].reshape(-1).astype(float))
    cols.append(surv["cols"].reshape(-1).astype(float))
    Xf = np.column_stack(cols)
    y = surv["dominant"].reshape(-1)
    haz = surv["hazard"].reshape(-1).astype(int)
    return Xf, y, haz, sensors + ["row", "col"]


if __name__ == "__main__":
    s = generate()
    Xf, y, haz, feat = to_table(s)
    print("grid: %d x %d = %d cells" % (s["grid"], s["grid"], s["grid"] ** 2))
    print("feature columns:", feat)
    print("class balance:", {c: int((y == i).sum()) for i, c in enumerate(CLASSES)})
    print("hazard cells:", int(haz.sum()), "/", len(haz))
    print("NaN fraction per sensor:", {k: round(float(np.isnan(v).mean()), 3) for k, v in s["X"].items()})
    json.dump({"assumptions": ASSUMPTIONS, "response": RESPONSE, "classes": CLASSES,
              "sensors_baseline": SENSORS, "sensors_all": ALL_SENSORS},
             open("landmap_synth_config.json", "w"), indent=1)
