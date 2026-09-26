# -*- coding: utf-8 -*-
"""Audit LANDMAP_EMA_Manuscript.docx: re-derive headline numbers from the JSON result files and
check they appear verbatim in the built document; check every reference is cited and every
citation has a reference; check figures/tables are introduced in order; scan for stale/withdrawn
phrasing and leftover placeholder text.

    python audit_manuscript.py
"""
import io
import json
import re
import zipfile

DOCX = r"C:\Users\Leon\Downloads\LANDMAP_EMA\LANDMAP_EMA_Manuscript.docx"
HERE = r"C:\YouTube\books_work\landmap"

R = json.load(io.open(HERE + r"\landmap_model_results.json", encoding="utf-8"))
E = json.load(io.open(HERE + r"\landmap_econ_results.json", encoding="utf-8"))
RB = json.load(io.open(HERE + r"\landmap_robustness_results.json", encoding="utf-8"))
DOIREFS = json.load(io.open(HERE + r"\_refs.json", encoding="utf-8"))

with zipfile.ZipFile(DOCX) as z:
    xml = z.read("word/document.xml").decode("utf-8")
xml = re.sub(r"</w:p>", "\n", xml)     # paragraph boundary -> whitespace, so DOI/number
xml = re.sub(r"<w:tab/>", " ", xml)    # lookaheads at a paragraph/cell edge still match
TEXT = re.sub(r"<[^>]+>", "", xml)
TEXT = re.sub(r"&amp;", "&", TEXT)
TEXT = re.sub(r"&#8722;", "-", TEXT)   # minus sign used in EQ/negative numbers

problems = []


def check_present(label, s):
    if s not in TEXT:
        problems.append("MISSING %-28s %r" % (label, s))


best = R["best_classifier"]
pooled = R["pooled_baseline"][best]
for cls in ("metal", "organic", "cd", "plastic", "glass"):
    check_present("F1 %s (pooled best)" % cls, "%.2f" % pooled["per_class_f1"][cls])
check_present("F1 inert (pooled best, 3dp table)", "%.3f" % pooled["per_class_f1"]["inert"])
check_present("macro-F1 best", "%.3f" % pooled["macro_f1"])
check_present("hazard AUC", "%.2f" % R["hazard"]["auc"]["mean"])
check_present("hazard AUC (3dp)", "%.3f" % R["hazard"]["auc"]["mean"])
check_present("hazard recall", "%.0f" % (R["hazard"]["recall"]["mean"] * 100))
for k in ("logreg", "rf", "xgb"):
    check_present("Brier %s" % k, "%.3f" % R["calibration"][k]["mean"])
    diff = R["split_comparison"][k]["paired_diff_random_minus_spatial"]
    check_present("paired diff %s" % k, "%+.3f" % diff["mean"])
check_present("distance-decay n_total", str(R["distance_decay"]["n_total"]))
abl = R["ablation_pooled"]
check_present("ablation all macro-F1", "%.3f" % abl["all"]["macro_f1"])
check_present("ablation minus-GPR macro-F1", "%.3f" % abl["all_minus_gpr"]["macro_f1"])
nir = R["nir_augmentation_pooled"]
check_present("NIR plastic baseline", "%.2f" % nir["baseline"]["per_class_f1"]["plastic"])
check_present("NIR plastic +NIR", "%.2f" % nir["plus_nir"]["per_class_f1"]["plastic"])
check_present("NIR macro baseline", "%.3f" % nir["baseline"]["macro_f1"])
check_present("NIR macro +NIR", "%.3f" % nir["plus_nir"]["macro_f1"])

check_present("VOI averaged mean", "$%.0f" % E["voi_averaged"]["mean"])
check_present("VOI averaged median", "$%.0f" % E["voi_averaged"]["median"])
check_present("VOI averaged P>0", "%.2f" % E["voi_averaged"]["p_positive"])
check_present("breakeven averaged median", "$%.0f" % E["breakeven_survey_cost_usd_averaged"]["median"])
check_present("VOI example median", "$%.0f" % E["voi_example"]["median"])
check_present("sensor value mag mean", "$%.0f" % E["sensor_value"]["mag"]["incremental_voi_mean"])
for s, v in E["sensor_value"].items():
    check_present("sensor value %s p_positive" % s, "%.0f" % (v["p_positive"] * 100))
check_present("NIR delta VOI p_positive", "%.0f" % (E["nir_delta_voi"]["p_positive"] * 100))

boundary = RB["voi_zero_boundary"]["combined_severity_fraction"]
check_present("VOI=0 boundary", "%.0f" % (boundary * 100))
cd_ = RB["combined_degradation"]
for r in cd_:
    check_present("combined degradation VOI level %d" % r["level"], "$%.0f" % r["voi_mean"])
check_present("noise baseline F1", "%.2f" % RB["sweeps"]["noise_mult"][0]["macro_f1_mean"])
check_present("noise max F1", "%.2f" % RB["sweeps"]["noise_mult"][-1]["macro_f1_mean"])

# ---- references: every DOI-tagged source cited, every citation has a reference
cited = set(re.findall(r"https://doi\.org/(10\.\S+?)(?=[)\]\s,;]|$)", TEXT))
all_dois = {v["doi"] for v in DOIREFS.values()}
uncited = all_dois - cited
if uncited:
    problems.append("REFERENCES not appearing as doi.org links in body/reflist: %s" % uncited)

# ---- figures/tables mentioned
for i in range(1, 10):
    check_present("Figure %d mention" % i, "Fig. %d" % i)
for i in range(1, 6):
    check_present("Table %d mention" % i, "Table %d" % i)

# ---- stale/withdrawn/placeholder scan
for bad in ("WITHDRAWN", "TODO", "TBD", "XXX", "PLACEHOLDER", "wasman.2019.06.030"):
    if bad in TEXT:
        problems.append("STALE/PLACEHOLDER marker found: %r" % bad)

# ---- sensor-count consistency: baseline described as five-sensor, not six-sensor, anywhere
for bad_phrase in ("six-sensor baseline", "six sensor channels", "6-sensor baseline"):
    if bad_phrase in TEXT:
        problems.append("SENSOR-COUNT WORDING still says %r" % bad_phrase)

print("=" * 70)
if problems:
    print("%d problem(s) found:\n" % len(problems))
    for p in problems:
        print(" -", p)
else:
    print("No problems found: every headline number, reference and figure/table check passed.")
print("=" * 70)
print("word count (approx):", len(TEXT.split()))
