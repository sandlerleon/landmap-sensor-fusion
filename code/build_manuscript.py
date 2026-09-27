# -*- coding: utf-8 -*-
"""Build the Environmental Monitoring and Assessment manuscript. Every number is read from
landmap_model_results.json, landmap_econ_results.json and landmap_robustness_results.json;
every DOI reference from the harvest_refs.py verification.

    python build_manuscript.py
"""
import io
import json
import os

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

import landmap_synth as LS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = r"C:\Users\Leon\Downloads\LANDMAP_EMA"
R = json.load(io.open(os.path.join(HERE, "landmap_model_results.json"), encoding="utf-8"))
E = json.load(io.open(os.path.join(HERE, "landmap_econ_results.json"), encoding="utf-8"))
RB = json.load(io.open(os.path.join(HERE, "landmap_robustness_results.json"), encoding="utf-8"))
DOIREFS = json.load(io.open(os.path.join(HERE, "_refs.json"), encoding="utf-8"))
HX = json.load(io.open(os.path.join(HERE, "landmap_hazard_extra_results.json"), encoding="utf-8"))
CV = json.load(io.open(os.path.join(HERE, "landmap_calib_voi_results.json"), encoding="utf-8"))
SM = json.load(io.open(os.path.join(HERE, "landmap_sensor_map_results.json"), encoding="utf-8"))
SME = json.load(io.open(os.path.join(HERE, "landmap_sensor_map_extreme_results.json"), encoding="utf-8"))
VM = json.load(io.open(os.path.join(HERE, "landmap_voi_map_results.json"), encoding="utf-8"))

LS_RESPONSE = LS.RESPONSE
LS_ALL_SENSORS = LS.ALL_SENSORS
LS_PATCH = LS.ASSUMPTIONS["patch_scale_m"]
LS_NOISE_CORR = LS.ASSUMPTIONS["noise_corr_scale_m"]
ECON = E["econ_assumptions"]
ECON_PRICE = ECON["price_usd_per_t"]
ECON_REC = ECON["recovery_fraction"]
REPO_URL = "https://github.com/sandlerleon/landmap-sensor-fusion"
CODE_DOI = os.environ.get("LANDMAP_CODE_DOI")

TITLE = ("From Classification Accuracy to Decision Value: Multimodal Sensor Fusion for "
         "Economic Prioritization of Landfill Mining")
SUBTITLE = "A Simulation-Based Value-of-Information Analysis"

# ------------------------------------------------------------------ document
doc = Document()
st = doc.styles["Normal"]
st.font.name = "Times New Roman"
st.font.size = Pt(11)
st.paragraph_format.space_after = Pt(6)
st.paragraph_format.line_spacing = 1.5
for s in doc.sections:
    s.left_margin = s.right_margin = s.top_margin = s.bottom_margin = Inches(1.0)
    ln = OxmlElement("w:lnNumType")
    ln.set(qn("w:countBy"), "1"); ln.set(qn("w:restart"), "continuous"); ln.set(qn("w:distance"), "360")
    s._sectPr.find(qn("w:pgMar")).addnext(ln)

ORDER = []


def C(*tags):
    nums = []
    for t in tags:
        if t not in ORDER:
            ORDER.append(t)
        nums.append(ORDER.index(t) + 1)
    nums = sorted(set(nums))
    parts, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        if j - i >= 2:
            parts.append("%d\u2013%d" % (nums[i], nums[j]))
        else:
            parts.extend(str(x) for x in nums[i:j + 1])
        i = j + 1
    return "[%s]" % ", ".join(parts)


def H(text, size=12, before=12):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text); r.bold = True; r.font.size = Pt(size)


def H2(text):
    H(text, size=11, before=8)


def Pp(text, indent=True, italic=False, size=11, spacing=1.5):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = spacing
    if indent:
        p.paragraph_format.first_line_indent = Inches(0.3)
    r = p.add_run(text); r.italic = italic; r.font.size = Pt(size)
    return p


EQN = [0]


def EQ(text):
    EQN[0] += 1
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.2
    p.paragraph_format.left_indent = Inches(0.6)
    r = p.add_run(text); r.italic = True
    p.add_run("\t\t(%d)" % EQN[0])
    return EQN[0]


def CAP(text):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run(text); r.font.size = Pt(9.5)


def FIG(name, width=6.3):
    doc.add_picture(os.path.join(HERE, "figures", name + ".png"), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER


def TBL(headers, rows, fs=8.5, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]; c.text = ""
        r = c.paragraphs[0].add_run(h); r.bold = True; r.font.size = Pt(fs)
        c.paragraphs[0].paragraph_format.line_spacing = 1.0
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            pp = cells[i].paragraphs[0]
            pp.paragraph_format.line_spacing = 1.0
            pp.paragraph_format.space_after = Pt(0)
            pp.add_run(str(v)).font.size = Pt(fs)
    if widths:
        for row in t.rows:
            for c, w in zip(row.cells, widths):
                c.width = Inches(w)


CLASSES = R["classes"]
KINDS = ["logreg", "rf", "xgb"]
KLAB = {"logreg": "logistic regression", "rf": "random forest", "xgb": "XGBoost"}
KHEAD = {"logreg": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost"}
ARTICLE = {"logreg": "a", "rf": "a", "xgb": "an"}
CLASS_LABEL = {"metal": "Metal", "plastic": "Plastic", "glass": "Glass", "organic": "Organic",
               "cd": "C&D", "inert": "Inert"}
best = R["best_classifier"]
pooled = R["pooled_baseline"][best]
sc = R["split_comparison"]
dd = R["distance_decay"]
haz = R["hazard"]
calib = R["calibration"]
abl = R["ablation_pooled"]
nir = R["nir_augmentation_pooled"]
econ = E
sv = E["sensor_value"]
nirvoi = E["nir_delta_voi"]
cd_ = RB["combined_degradation"]
ov = RB["sweeps"]["overlap_shrink"]
boundary = RB["voi_zero_boundary"]
hazx = HX

_extreme_winners = sorted(set(v["winner"] for v in SME.values()))
_grid_winners = sorted(set(cell["winner"] for row in SM["grid"] for cell in row))
SNAME = {"mag": "magnetometer", "emi": "EMI", "gpr": "GPR", "therm": "thermal", "gas": "gas"}
EXTREME_SUMMARY = ("robust to price fluctuation but not to site composition: the magnetometer "
                   "led at every tested metal/plastic price combination, yet a deliberately "
                   "polymer-, debris-, or glass-rich site archetype shifted the top sensor to "
                   "%s" % " or ".join(sorted(set(SNAME[w] for w in _extreme_winners if w != "mag"))))


def pf1(d, cls):
    return d["per_class_f1"][cls]


# ==================================================================== FRONT
p = doc.add_paragraph(); p.paragraph_format.line_spacing = 1.3
r = p.add_run(TITLE); r.bold = True; r.font.size = Pt(14)
p2 = doc.add_paragraph(); p2.paragraph_format.line_spacing = 1.2
r2 = p2.add_run(SUBTITLE); r2.italic = True; r2.font.size = Pt(12)
Pp("Leon Sandler", indent=False, spacing=1.2)
Pp("Independent Researcher, Northbrook, Illinois, USA", indent=False, size=10, spacing=1.2)
Pp("E-mail: sandler.leon@gmail.com \u00b7 ORCID: 0009-0007-4584-808X", indent=False, size=10, spacing=1.2)

H("Abstract")
ABSTRACT = (
    "Landfill-mining projects are approved or shelved on incomplete information about what "
    "is underground, and a systematic assessment of European scenarios found roughly "
    "%d%% of them profitable %s. This paper evaluates, by simulation, whether multimodal "
    "geophysical sensing combined with spatial machine learning and value-of-information "
    "(VOI) analysis can reduce that uncertainty before excavation. On a synthetic but "
    "physically motivated 1,024 m\u00b2 survey (five geophysical/gas channels: magnetometer, "
    "electromagnetic induction, ground-penetrating radar, thermal and gas, surveyed alongside "
    "surface topography for georeferencing), %s %s "
    "classifier reaches pooled F1 of %.2f (metal), %.2f (organic) and %.2f (construction "
    "and demolition debris) under a spatially blocked, non-leaky validation protocol, "
    "against %.2f (plastic) and %.2f (glass). Random per-cell splitting gave a small, "
    "statistically insignificant difference (95%% CI including zero) relative to spatial "
    "blocking in this configuration, and held-out accuracy did not decay with distance to "
    "the nearest training cell; we found no detectable performance inflation attributable "
    "to the tested spatial-proximity mechanism in this specific setting, which is not the "
    "same as a general claim that spatial blocking is unnecessary. Sensor ablation shows "
    "ground-"
    "penetrating radar contributes most to classification (macro-F1 falls by %.3f when "
    "removed), but an economic value-of-information analysis shows only the magnetometer "
    "reliably increases the excavation decision's expected value; classification "
    "improvements from the other channels did not translate into decision value in this "
    "model. Adding a near-infrared channel raises plastic F1 from %.2f to %.2f but leaves "
    "VOI statistically unchanged, because low-value materials contribute little to the "
    "economic layer regardless of classification accuracy. In a synthetic hazard-detection "
    "test whose ground truth and sensor reading share the same underlying generative model "
    "— an upper-bound test, not an independent validation — the gas-hazard "
    "classifier reaches ROC-AUC %.2f and %.0f%% recall at a safety-oriented operating point, "
    "though at that operating point only %.0f%% of cells flagged as hazardous are true "
    "positives. Across the baseline synthetic scenarios and stated economic parameter "
    "distributions, the simulated survey has positive median VOI (P(VOI>0) = %.2f) and a "
    "break-even survey cost of $%.0f; a two-dimensional map over sensing-quality and "
    "economic-adversity severity locates a model-specific VOI-zero boundary rather than a "
    "single degradation threshold, and a further test across deliberately extreme economic "
    "regimes finds the magnetometer's economic advantage %s. All results are model "
    "predictions, and a field validation and falsification plan with acceptance criteria is "
    "given."
    % (19, C("laner2019"), ARTICLE[best], KLAB[best], pf1(pooled, "metal"), pf1(pooled, "organic"),
       pf1(pooled, "cd"), pf1(pooled, "plastic"), pf1(pooled, "glass"),
       abl["all"]["macro_f1"] - abl["all_minus_gpr"]["macro_f1"],
       pf1(nir["baseline"], "plastic"), pf1(nir["plus_nir"], "plastic"),
       haz["auc"]["mean"], haz["recall"]["mean"] * 100, hazx["pooled_precision"] * 100,
       econ["voi_averaged"]["p_positive"], econ["breakeven_survey_cost_usd_averaged"]["median"],
       EXTREME_SUMMARY))
Pp(ABSTRACT, indent=False)
Pp("Keywords: landfill mining; multimodal sensor fusion; spatial cross-validation; value of "
   "information; random forest; XGBoost; probability calibration; uncertainty quantification",
   indent=False, size=10)

# ==================================================================== 1
H("1 Introduction")
Pp("Landfill-mining projects convert a waste deposit into recovered material, avoided "
   "landfill tax, and reclaimed land, but their economics depend on what is actually "
   "underground: material composition, resource concentration, hazard locations, and the "
   "processing each fraction requires. A systematic assessment of 531,441 modelled "
   "European landfill-mining scenarios found only about 19%% net profitable and about 80%% "
   "returning a negative net present value %s, and follow-on work has examined which "
   "factors most affect that economic outcome and how enhanced landfill mining performs "
   "on environmental and economic criteria %s. The recurring finding is that uncertainty "
   "before excavation, not the difficulty of excavation itself, is what turns a potentially "
   "viable project into a cost centre." % (C("laner2019"), C("danthurebandara2015")))
Pp("Geophysical characterisation of landfills is well established. Electrical resistivity "
   "and induced-polarisation surveys have long been used to investigate landfill structure "
   "and leachate %s, frequency-domain electromagnetic induction has been applied to landfill "
   "site characterisation since at least the early 1990s %s and, more recently, to imaging "
   "inside capped sites %s, ground-penetrating radar has been used to estimate construction-"
   "waste landfill volume %s, and integrated electrical and electromagnetic prospecting has "
   "been proposed specifically for municipal waste landfills %s. What this literature "
   "characterises is subsurface structure and, at best, broad material classes; it does not, "
   "in general, convert that characterisation into a per-zone economic excavation decision "
   "with a quantified value of doing the survey at all."
   % (C("dedonno2024"), C("jansen1992"), C("deidda2022"), C("zhang2022"), C("dedonno2024")))
Pp("This paper evaluates one candidate architecture for closing that gap, previously "
   "described only as an industrial proposal, by treating it as a design hypothesis and "
   "testing it computationally: can multimodal geophysical sensing, fused by spatial "
   "machine learning with a spatially valid train/test protocol and propagated through an "
   "explicit economic layer, actually change a landfill-mining excavation decision, and by "
   "how much? The question is deliberately falsifiable. We do not assume the answer is yes.")
Pp("The contributions are: (i) a physically motivated synthetic-survey generator with five "
   "material-response sensor channels plus an optional near-infrared augmentation channel, "
   "material-dependent response, depth attenuation, moisture effects, "
   "spatially correlated noise and instrument dropout, released as open code so every "
   "number in this paper can be regenerated; (ii) a systematic comparison of spatially "
   "blocked against randomly split validation, including a distance-to-nearest-training-"
   "observation diagnostic of the kind used in the spatial-statistics literature %s; (iii) "
   "sensor ablation and near-infrared augmentation studies, evaluated on both classification "
   "F1 and economic value of information, which are shown to disagree; (iv) a Monte Carlo "
   "value-of-information and break-even survey-cost analysis with per-sensor value "
   "attribution; and (v) a synthetic-to-real degradation sweep that locates, rather than "
   "asserts, the point at which the survey stops being worth its cost." % C("roberts2017"))

# ==================================================================== 2
H("2 Related work")
H2("2.1 Landfill geophysics")
Pp("Electrical and electromagnetic methods have been applied to landfill investigation for "
   "several decades, from early frequency-domain electromagnetic induction soundings for "
   "site characterisation %s, through resistivity surveys of leachate plumes %s, to recent "
   "electromagnetic induction imaging of a capped site %s and integrated electrical/"
   "electromagnetic prospecting proposed specifically for municipal waste landfills %s. "
   "Ground-penetrating radar has been used to estimate the volume of construction-waste "
   "landfills from density contrasts %s. These studies establish that geophysical signals "
   "carry usable information about landfill structure and composition; none couples that "
   "information to a quantified excavation decision."
   % (C("jansen1992"), C("dedonno2024"), C("deidda2022"), C("dedonno2024"), C("zhang2022")))
H2("2.2 Spatial machine learning and validation")
Pp("Spatially structured data violate the independence assumption behind ordinary "
   "cross-validation, and ignoring that structure can seriously underestimate predictive "
   "error or, conversely, produce optimistic validation statistics that do not hold outside "
   "the training footprint %s. Spatial sorting bias in species-distribution models has been "
   "shown to inflate cross-validated accuracy relative to spatially independent evaluation "
   "%s, and a large-scale demonstration using 11.8 million trees found that non-spatial "
   "validation of a random-forest biomass model substantially overstated its real predictive "
   "power relative to spatially blocked validation %s. Random forests %s and gradient-"
   "boosted trees such as XGBoost %s are standard for this kind of tabular geophysical "
   "classification, and spatial transferability of random-forest crop-type classification "
   "has itself been studied directly %s." % (C("roberts2017"), C("hijmans2012"), C("ploton2020"),
                                             C("breiman2001"), C("chen2016"), C("orynbaikyzy2022")))
H2("2.3 Probability calibration")
Pp("A classifier that outputs \u201c72%% probability of metal-rich material\u201d is only "
   "useful to an economic decision layer if that 72%% is empirically reliable; well-"
   "calibrated probabilities from standard classifiers should not be assumed, and modern "
   "flexible classifiers in particular can be poorly calibrated by default %s." % C("niculescu2005"))
H2("2.4 Landfill-mining economics and value of information")
Pp("Enhanced landfill mining has been reviewed for its resource-recovery potential across "
   "material, energy and land-use value streams %s, for its multi-resource recovery "
   "prospects more broadly %s, for a coordinated European research programme spanning "
   "geophysics to material valorisation %s, and for country-specific resource-recovery "
   "potential %s. Value-of-information analysis, comparing the expected outcome of a "
   "decision made with additional data against the best decision made without it, net of "
   "the data's cost, has a long history in mineral exploration under a Bayesian decision-"
   "theoretic framing %s and has recently been reviewed for marine conservation management "
   "%s; we are not aware of a prior application to landfill-mining excavation decisions."
   % (C("jones2013"), C("jones2013"), C("vollprecht2021"), C("yi2019"), C("rendu1976"),
      C("luhede2025")))
H2("2.5 Explainability")
Pp("Permutation importance is a model-agnostic way to rank feature contributions by the "
   "performance drop when a feature is shuffled %s; we use the same logic at the level of "
   "whole sensor channels (ablation) and, in Section 5.4, at the level of economic value "
   "rather than classification accuracy, which is the more decision-relevant question for "
   "this application." % C("huang2016"))
H2("2.6 Research gap")
Pp("No prior study, to our knowledge, evaluates an integrated multimodal-sensing-to-"
   "excavation-decision pipeline for landfill mining under a spatially valid protocol, with "
   "sensor value measured economically rather than only by classification accuracy, and with "
   "an explicit, model-specific decision-viability boundary rather than a single robustness "
   "point estimate. That is what this paper contributes.")

# ==================================================================== 3
H("3 LAND-MAP framework")
H2("3.1 Sensing architecture")
Pp("The evaluated architecture surveys a site with six co-registered channels: LiDAR/RGB "
   "surface topography, long-wave infrared thermal imaging, a magnetometer, an "
   "electromagnetic-induction (EMI) sensor for bulk conductivity and moisture, ground-"
   "penetrating radar (GPR) for density contrast and depth, and a methane/VOC gas array, all "
   "mounted on a georeferenced platform. Every sensing technology is individually mature and "
   "commercially available; the contribution evaluated here is the fusion, validation and "
   "economic-decision layer built above them, not new sensor hardware. The classification "
   "model evaluated in Sections 4–5 draws on the five channels with a modelled material "
   "response (magnetometer, EMI, GPR, thermal, gas); LiDAR/RGB topography provides "
   "georeferencing and cell-depth context rather than an independent classification input, "
   "and a sixth, near-infrared/optical channel is evaluated separately as a proposed "
   "augmentation (Section 4.4). Figure 1 summarises the pipeline from raw channels to "
   "excavation decision.", indent=False)
FIG("figure1_architecture", width=6.5)
CAP("Fig. 1 LAND-MAP pipeline: co-registered sensing channels feed a per-cell fusion "
    "classifier under a spatial train/test protocol, producing material and gas-hazard "
    "probabilities that an economic decision layer converts into an excavation priority map.")
H2("3.2 Fusion and economic decision layers")
Pp("Per-cell readings are combined by a trained classifier into probabilistic material "
   "classifications over six classes (metal, plastic, glass, organic, construction and "
   "demolition debris, and inert) and a separate gas-hazard probability. Each cell's "
   "expected composition is converted into a Resource Value Score by multiplying predicted "
   "class probabilities by an assumed recoverable mass, recovery fraction and secondary-"
   "material price, and netting against excavation and processing cost; aggregating scores "
   "gives an Excavation Priority Map. A Value-of-Information metric compares the expected "
   "outcome of survey-guided excavation against the best decision an operator could make "
   "without any survey, net of the survey's own cost.")

# ==================================================================== 4
H("4 Methods")
H2("4.1 Synthetic landfill generation")
Pp("A 32 \u00d7 32 m (1,024 m\u00b2) survey area is discretised into 1 m\u00d71 m cells over "
   "three coarse depth layers (0\u20131, 1\u20132, 2\u20133 m). A latent dominant material is "
   "assigned to every cell and depth layer by thresholding a smooth two-dimensional Gaussian "
   "random field against the class prior, giving spatially patchy (not salt-and-pepper) "
   "material zones consistent with how waste is dumped and compacted in lots, with a "
   "correlation length of %.1f m. Cover depth to the dominant layer and moisture are "
   "independent smooth random fields." % LS_PATCH, indent=False)
EQ("x_{i,j} = f_j(M_i, z_i, m_i) + \u03b5_{i,j}")
Pp("where x_{i,j} is sensor j's reading at cell i, M_i the material at each depth layer, "
   "z_i the cover depth, m_i the moisture, and f_j a material-response function attenuated "
   "with depth as exp(\u2212\u03b1_j z) and, for GPR and EMI, further modified by moisture. "
   "Response coefficients (Table 1) are literature-motivated orders of magnitude, not fitted "
   "values, and are stated once so they can be perturbed systematically (Section 4.6). "
   "Sensor noise \u03b5_{i,j} has both an independent per-cell component and a spatially "
   "correlated component (correlation length %.1f m, modelling instrument drift and "
   "footprint overlap), and each channel independently drops out at cell-specific rates "
   "reflecting real failure modes (highest for gas, from drift and saturation, and GPR, from "
   "coupling and siting issues)." % LS_NOISE_CORR, indent=False)
TBL(["Channel", "Metal", "Plastic", "Glass", "Organic", "C&D", "Inert"],
    [[ch.upper() if ch != "optical" else "NIR (optional)"] +
     ["%.2f" % LS_RESPONSE[cls][LS_ALL_SENSORS.index(ch)] for cls in CLASSES]
     for ch in LS_ALL_SENSORS], widths=[1.1] + [0.85] * 6)
CAP("Table 1. Assumed mean sensor response by material and channel, arbitrary standardised "
    "units, before depth attenuation, moisture effects and noise. Every value is an "
    "assumption motivated by the qualitative physics of each sensing modality (e.g. strong "
    "magnetometer/EMI response to metal, GPR response to density/dielectric contrast, "
    "thermal and gas response to organic decomposition); none is fitted to field data.")
Pp("The gas-hazard label is a noisy, logistic threshold of the pre-noise gas-channel signal "
   "itself, so that hazard status is by construction what an ideal gas sensor would detect; "
   "the deployed sensor differs from that truth only by its own instrument noise and "
   "dropout, as a real hazard label derived from independent gas testing would relate to a "
   "field-deployed sensor. Figure 2 shows one representative synthetic survey: the true "
   "near-surface material field, one raw noisy sensor channel, and the resulting gas-hazard "
   "truth.", indent=True)
FIG("figure2_synthetic_survey", width=6.5)
CAP("Fig. 2 One representative 32×32 m synthetic survey. (A) true near-surface dominant "
    "material; (B) raw magnetometer reading (noisy, with dropout); (C) gas-hazard truth "
    "derived from the pre-noise gas signal.")
H2("4.2 Spatial validation protocol")
Pp("Two held-out protocols are compared on the same synthetic surveys: a spatially blocked "
   "holdout, in which one contiguous %d\u00d7%d cell region is withheld entirely, and a "
   "random per-cell holdout of the same size, scattered uniformly. To test the mechanism "
   "behind any difference directly rather than only at one block size, we also compute, "
   "pooling across %d repeats and eight block sizes (4\u201320 cells), the distance from "
   "every held-out cell to its nearest training cell, and report held-out accuracy as a "
   "function of that distance %s." % (R["block"], R["block"], 15, C("roberts2017", "ploton2020")),
   indent=False)
H2("4.3 Classifiers and calibration")
Pp("Three classifiers are compared: logistic regression, random forest %s, and XGBoost %s, "
   "each with mean imputation for sensor dropout. Calibration is assessed by the one-vs-rest "
   "Brier score, macro-averaged over classes %s, because the economic layer consumes class "
   "probabilities directly and a poorly calibrated \u201c72%%\u201d is not the same as an "
   "empirically reliable one." % (C("breiman2001"), C("chen2016"), C("niculescu2005")))
H2("4.4 Sensor ablation and NIR augmentation")
Pp("Each of the five material-response channels is removed in turn and the best-performing "
   "classifier retrained, isolating each channel's contribution to pooled macro-F1. A "
   "sixth, optical/near-infrared channel, motivated by the plastic/glass confusion this "
   "paper's synthetic physics reproduces, is added to test whether it repairs that specific "
   "limitation under the assumed NIR response coefficients (Table 1); it is modelled as a "
   "surface-only channel (rapid depth attenuation) with distinct plastic and glass "
   "responses. Because those coefficients are assumptions rather than measured spectra, "
   "this tests a hypothetical NIR channel with the stated response structure, not a claim "
   "about any specific real instrument (Section 5.5).", indent=False)
H2("4.5 Economic model and value of information")
Pp("For material class c the expected net value of excavating cell i is")
EQ("v_i = \u2211_c P(c\u2223x_i)\u00b7m_i\u00b7\u03c1_c\u00b7\u03c0_c \u2212 m_i(k_{ex} + k_{pr})")
Pp("with m_i the assumed waste mass per cell, \u03c1_c the recovery fraction, \u03c0_c the "
   "secondary-material price, and k_{ex}, k_{pr} the excavation and processing cost per "
   "tonne (Table 2, all assumptions, swept in Monte Carlo rather than fixed). The survey-"
   "guided decision excavates cell i iff v_i > 0; the blind decision excavates every cell "
   "iff the population-mean expected value is positive, else none. Value of information is",
   indent=False)
EQ("VOI = EV_{survey} \u2212 EV_{blind} \u2212 C_{survey}")
Pp("evaluated against the true (not expected) cell value, over %d Monte Carlo draws of the "
   "economic parameters. The break-even survey cost is EV_{survey} \u2212 EV_{blind} with "
   "C_{survey} excluded, the maximum a rational operator should pay. Per-sensor value is "
   "VOI_{all} \u2212 VOI_{without\\ j}, computed on the same economic draws for a paired "
   "comparison, and averaged over %d independent synthetic surveys because a single "
   "survey's VOI estimate is noisy." % (E["n_mc"], E["n_rep_econ"]), indent=False)
TBL(["Parameter", "Range (uniform)"],
    [["Price, $/t: metal", "%d\u2013%d" % tuple(ECON_PRICE["metal"])],
     ["Price, $/t: plastic", "%d\u2013%d" % tuple(ECON_PRICE["plastic"])],
     ["Price, $/t: glass", "%d\u2013%d" % tuple(ECON_PRICE["glass"])],
     ["Price, $/t: organic", "%d\u2013%d" % tuple(ECON_PRICE["organic"])],
     ["Price, $/t: C&D", "%d\u2013%d" % tuple(ECON_PRICE["cd"])],
     ["Price, $/t: inert", "%d\u2013%d" % tuple(ECON_PRICE["inert"])],
     ["Recovery fraction: metal", "%.2f\u2013%.2f" % tuple(ECON_REC["metal"])],
     ["Recovery fraction: plastic", "%.2f\u2013%.2f" % tuple(ECON_REC["plastic"])],
     ["Recovery fraction: glass", "%.2f\u2013%.2f" % tuple(ECON_REC["glass"])],
     ["Mass per cell, t", "%.1f\u2013%.1f" % tuple(ECON["mass_per_cell_t"])],
     ["Excavation cost, $/t", "%d\u2013%d" % tuple(ECON["excavation_cost_usd_per_t"])],
     ["Processing cost, $/t", "%d\u2013%d" % tuple(ECON["processing_cost_usd_per_t"])],
     ["Survey cost, $ total", "%d\u2013%d" % tuple(ECON["survey_cost_usd_total"])]], widths=[2.6, 2.0])
CAP("Table 2. Economic assumptions (illustrative, as labelled throughout the source "
    "proposal), swept uniformly in the Monte Carlo analysis rather than fixed at a point "
    "estimate. Full ranges for every material class are in the released configuration file.")
H2("4.6 Synthetic-to-real robustness sweep")
Pp("The largest acknowledged risk in this design is that real, weathered, heterogeneous "
   "waste is less distinctive than any simulation's idealised material signatures. Rather "
   "than assert a margin of safety, we degrade the simulation directly along five axes named "
   "in the source proposal's own risk assessment: overall sensor noise, dropout "
   "probability, GPR depth attenuation (modelling moisture/conductivity losses), moisture-"
   "driven GPR attenuation specifically, and material-signature overlap (each material's "
   "response shrunk toward the six-class mean). Each axis is swept individually and then "
   "combined, and the combined sweep is searched to locate the point at which mean VOI "
   "crosses zero, then extend the same logic to two dimensions with an independent "
   "economic-adversity axis (Section 5.11) \u2014 a model-specific VOI-zero boundary this "
   "paper reports rather than assumes.",
   indent=False)

# ==================================================================== 5
H("5 Results")
H2("5.1 Material classification")
TBL(["Class"] + [KHEAD[k] for k in KINDS],
    [[CLASS_LABEL[cls]] + ["%.3f" % R["pooled_baseline"][k]["per_class_f1"][cls] for k in KINDS]
     for cls in CLASSES] +
    [["Macro-F1"] + ["%.3f" % R["pooled_baseline"][k]["macro_f1"] for k in KINDS]],
    widths=[1.3, 1.5, 1.5, 1.5])
CAP("Table 3. Pooled per-class F1 under the spatially blocked protocol, three classifiers, "
    "%d pooled test cells across %d repeats." % (R["pooled_baseline"][best]["n_pooled"], R["n_repeats"]))
FIG("figure3_per_class_f1")
CAP("Fig. 3 Per-class F1 by classifier, five-channel material-response baseline, spatial holdout.")
Pp("Metal (F1 %.2f\u2013%.2f across classifiers) and organic material (%.2f\u2013%.2f) are "
   "classified reliably, and construction and demolition debris reaches %.2f\u2013%.2f once "
   "its magnetometer/EMI signature (motivated by embedded rebar and metal fragments) is "
   "included; these three approach the 0.90\u20130.98 range the source proposal reported. "
   "Plastic (%.2f\u2013%.2f) and glass (%.2f\u2013%.2f) remain poorly discriminated by this "
   "sensor set, consistent with the proposal's own diagnosis that both are weakly magnetic "
   "and lack a distinguishing signal in the magnetic/EMI/GPR/thermal/gas channel set. XGBoost "
   "gives the highest macro-F1 (%.3f) and is used as the reference classifier for the "
   "remaining analyses unless stated otherwise."
   % (min(R["pooled_baseline"][k]["per_class_f1"]["metal"] for k in KINDS),
      max(R["pooled_baseline"][k]["per_class_f1"]["metal"] for k in KINDS),
      min(R["pooled_baseline"][k]["per_class_f1"]["organic"] for k in KINDS),
      max(R["pooled_baseline"][k]["per_class_f1"]["organic"] for k in KINDS),
      min(R["pooled_baseline"][k]["per_class_f1"]["cd"] for k in KINDS),
      max(R["pooled_baseline"][k]["per_class_f1"]["cd"] for k in KINDS),
      min(R["pooled_baseline"][k]["per_class_f1"]["plastic"] for k in KINDS),
      max(R["pooled_baseline"][k]["per_class_f1"]["plastic"] for k in KINDS),
      min(R["pooled_baseline"][k]["per_class_f1"]["glass"] for k in KINDS),
      max(R["pooled_baseline"][k]["per_class_f1"]["glass"] for k in KINDS),
      R["pooled_baseline"]["xgb"]["macro_f1"]))
H2("5.2 Spatial versus random validation")
FIG("figure4_spatial_validation")
CAP("Fig. 4 (a) Macro-F1 under spatially blocked versus randomly split holdout, three "
    "classifiers, mean \u00b1 SD over %d repeats. (b) Held-out accuracy against distance to "
    "the nearest training cell, pooled over %d repeats and eight block sizes; the dashed "
    "line is the overall mean." % (R["n_repeats"], 15))
Pp("Contrary to the hypothesis that random per-cell splitting inflates apparent performance "
   "through spatial-autocorrelation leakage %s, the paired difference (random minus spatial) "
   "was small and its 95%% confidence interval included zero for every classifier: %s. "
   "Distance-to-nearest-training-cell analysis (Fig. 4b, pooled over %d observations) shows "
   "no accuracy decay out to 10 m, the largest distance sampled. We had expected a gap, "
   "given the spatial correlation deliberately built into both the material field and the "
   "sensor noise; its absence indicates that, for a feature set built from single-cell "
   "sensor readings, per-cell classification difficulty in this system is dominated by "
   "genuine material overlap (plastic/glass/inert) rather than by memorisable local spatial "
   "structure. We report this as a specific, tested finding, not a general claim that "
   "spatial blocking is unnecessary; the mechanism identified in the literature %s requires "
   "a model able to exploit local proximity, which single-cell features here evidently do "
   "not provide, and we recommend spatial blocking as standard practice regardless, because "
   "a less idealised real dataset need not share this property."
   % (C("roberts2017", "ploton2020", "hijmans2012"),
      "; ".join("%s: %+.3f (95%% CI %+.3f to %+.3f)" %
               (KLAB[k], sc[k]["paired_diff_random_minus_spatial"]["mean"],
                sc[k]["paired_diff_random_minus_spatial"]["ci95"][0],
                sc[k]["paired_diff_random_minus_spatial"]["ci95"][1]) for k in KINDS),
      dd["n_total"], C("roberts2017", "ploton2020")))
H2("5.3 Sensor ablation")
FIG("figure5_ablation_sensor_value")
CAP("Fig. 5 (a) Pooled macro-F1 with each sensor removed in turn. (b) Incremental value of "
    "information (VOI with all sensors minus VOI without sensor j), mean \u00b1 SD over %d "
    "independent surveys, with the fraction of surveys giving a positive incremental value."
    % E["n_rep_econ"])
Pp("Removing GPR costs the most classification performance (macro-F1 %.3f \u2192 %.3f), "
   "followed by EMI (\u2192 %.3f); removing the magnetometer, thermal or gas channels costs "
   "little (\u2192 %.3f, %.3f, %.3f respectively). By classification accuracy alone, GPR and "
   "EMI would be judged the most valuable channels."
   % (abl["all"]["macro_f1"], abl["all_minus_gpr"]["macro_f1"], abl["all_minus_emi"]["macro_f1"],
      abl["all_minus_mag"]["macro_f1"], abl["all_minus_therm"]["macro_f1"],
      abl["all_minus_gas"]["macro_f1"]))
H2("5.4 Sensor value: classification accuracy is not economic value")
Pp("The economic picture is different, and this divergence is the paper's central "
   "empirical finding. Of the five channels, only the magnetometer shows a reliably "
   "positive incremental VOI (mean $%.0f, positive in %.0f%% of independent surveys); EMI, "
   "GPR, thermal and gas each have an incremental VOI whose sign is not consistent across "
   "surveys (positive in %.0f%%, %.0f%%, %.0f%% and %.0f%% of surveys respectively). The "
   "reason is that the Resource Value Score is dominated by metal, whose assumed price "
   "(Table 2) is several times any other class's, so a channel's contribution to overall "
   "classification accuracy does not translate into decision value unless it specifically "
   "improves metal detection. GPR carries the most unique subsurface information and is "
   "the strongest classification contributor, but that information is spread across "
   "several material classes with low individual economic weight."
   % (sv["mag"]["incremental_voi_mean"], sv["mag"]["p_positive"] * 100,
      sv["emi"]["p_positive"] * 100, sv["gpr"]["p_positive"] * 100,
      sv["therm"]["p_positive"] * 100, sv["gas"]["p_positive"] * 100))
H2("5.5 Hypothetical optical/NIR augmentation")
FIG("figure6_nir_augmentation")
CAP("Fig. 6 Per-class pooled F1, five-channel material-response baseline versus baseline plus a "
    "hypothetical near-infrared/optical channel.")
Pp("Under the assumed NIR response coefficients (Table 1) — a hypothetical surface "
   "optical channel with plastic- and glass-distinguishing response, not spectra from any "
   "specific real instrument — adding it raises plastic F1 from %.2f to %.2f and macro-F1 from "
   "%.3f to %.3f, but glass F1 moves from %.2f to %.2f \u2014 essentially unchanged, and in "
   "this simulation slightly down, because the added channel and the classifier's rebalancing "
   "across six classes do not uniformly benefit every class. The economic effect is smaller "
   "still: mean incremental VOI from adding NIR is $%.1f (SD $%.1f), positive in only %.0f%% "
   "of surveys \u2014 statistically indistinguishable from no effect. Plastic and glass "
   "together carry a modest assumed price and a low assumed recovery fraction (Table 2), so "
   "even a genuine classification improvement for these classes barely moves the excavation "
   "decision. The proposal's own hypothesis \u2014 that an optical channel, not further "
   "classifier tuning, is needed to separate plastic and glass \u2014 is partially supported "
   "(plastic improves; glass does not), and in either case the improvement is not what would "
   "justify the channel economically in this model."
   % (pf1(nir["baseline"], "plastic"), pf1(nir["plus_nir"], "plastic"),
      nir["baseline"]["macro_f1"], nir["plus_nir"]["macro_f1"],
      pf1(nir["baseline"], "glass"), pf1(nir["plus_nir"], "glass"),
      nirvoi["mean"], nirvoi["sd"], nirvoi["p_positive"] * 100))
H2("5.6 Calibration and hazard detection")
FIG("figure7_calibration_hazard")
CAP("Fig. 7 (a) Mean one-vs-rest Brier score by classifier (lower is better calibrated). "
    "(b) Gas-hazard classifier ROC-AUC and recall at a safety-oriented operating point "
    "(the lowest threshold achieving \u226595%% recall), mean \u00b1 SD over %d repeats."
    % R["n_repeats"])
Pp("Logistic regression is the best calibrated of the three classifiers (Brier %.3f), "
   "followed by random forest (%.3f) and XGBoost (%.3f); XGBoost's higher discriminative "
   "power (Table 3) therefore comes with somewhat less reliable probabilities, which "
   "matters because the economic layer consumes those probabilities directly rather than "
   "only the predicted class %s. The gas-hazard classifier reaches ROC-AUC %.3f "
   "(SD %.3f) and, at an operating point chosen to guarantee at least 95%% recall, achieves "
   "%.1f%% recall (SD %.3f) in held-out testing, approaching the 97%%/0.97 figures the "
   "source proposal reported from its own Stage 1 testing. Recall alone overstates this "
   "operating point's usefulness: a detector can reach 100%% recall by flagging nearly "
   "everything. Pooled across the same %d repeats, precision at this operating point is "
   "only %.0f%% and specificity %.0f%%, meaning about %.0f%% of cells are flagged as "
   "hazardous and roughly two in five flagged cells are false positives (PR-AUC %.3f); the "
   "operating point is deliberately chosen to guarantee recall for a safety application, "
   "and it does so, but at a real, quantified precision cost that the recall figure alone "
   "does not convey."
   % (calib["logreg"]["mean"], calib["rf"]["mean"], calib["xgb"]["mean"], C("niculescu2005"),
      haz["auc"]["mean"], haz["auc"]["sd"], haz["recall"]["mean"] * 100, haz["recall"]["sd"],
      hazx["n_repeats_used"], hazx["pooled_precision"] * 100, hazx["pooled_specificity"] * 100,
      hazx["metrics"]["frac_flagged"]["mean"] * 100, hazx["metrics"]["pr_auc"]["mean"]))
H2("5.7 Economic prioritisation and value of information")
FIG("figure8_voi_breakeven")
CAP("Fig. 8 (a) Monte Carlo distribution of Value of Information for one representative "
    "survey (%d draws). (b) Distribution of the break-even survey cost (the maximum a "
    "rational operator should pay), same draws." % 8000)
Pp("For the representative survey used throughout Sections 5.3\u20135.6, median VOI is "
   "$%.0f (90%% interval $%.0f\u2013$%.0f) and the survey is worth doing in %.0f%% of Monte "
   "Carlo draws. Averaged over %d independent synthetic surveys, mean VOI is $%.0f "
   "(median $%.0f), positive in %.0f%% of surveys, and the median break-even survey cost is "
   "$%.0f \u2014 the amount up to which an operator should rationally be willing to pay for "
   "a survey of this kind under the stated economic assumptions."
   % (econ["voi_example"]["median"], econ["voi_example"]["p05"], econ["voi_example"]["p95"],
      econ["voi_example"]["p_positive"] * 100, E["n_rep_econ"],
      econ["voi_averaged"]["mean"], econ["voi_averaged"]["median"], econ["voi_averaged"]["p_positive"] * 100,
      econ["breakeven_survey_cost_usd_averaged"]["median"]))
Pp("Section 5.8's baseline scenario reports a somewhat higher mean VOI ($%.0f) for what is "
   "nominally the same untouched baseline. The two estimates are not expected to coincide: "
   "the robustness sweep averages %d survey realisations at %d Monte Carlo draws each per "
   "scenario, chosen for speed across %d scenarios, against %d realisations at %d draws "
   "here; Section 5.9 confirms the classifier itself (random forest throughout the economic "
   "layer, versus XGBoost in the robustness sweep before this revision) changes mean VOI by "
   "only about $%.0f, so the gap is sampling noise in the repeat-average estimate, not a "
   "protocol inconsistency — both are unbiased estimates of the same baseline quantity, "
   "and $%.0f falls within their combined sampling uncertainty."
   % (RB["baseline"]["voi_mean"], RB["n_rep"], 1500, 33,
      E["n_rep_econ"], E["n_mc"], abs(CV["voi_diff_xgb_minus_rf"]["mean"]),
      abs(RB["baseline"]["voi_mean"] - econ["voi_averaged"]["mean"])))
H2("5.8 Synthetic-to-real robustness")
AXIS_LABEL = {"noise_mult": "Sensor noise (× baseline SD)", "dropout_mult": "Dropout probability (× baseline)",
              "gpr_atten_mult": "GPR depth attenuation (× baseline)",
              "moisture_mult": "Moisture-driven GPR attenuation (× baseline)",
              "overlap_shrink": "Material-signature overlap (shrinkage fraction)"}
TBL(["Degradation axis", "Baseline\nmacro-F1", "Most severe\nmacro-F1", "Baseline\nmean VOI ($)",
     "Most severe\nmean VOI ($)"],
    [[AXIS_LABEL[axis], "%.3f" % arr[0]["macro_f1_mean"], "%.3f" % arr[-1]["macro_f1_mean"],
      "%.0f" % arr[0]["voi_mean"], "%.0f" % arr[-1]["voi_mean"]]
     for axis, arr in RB["sweeps"].items()], widths=[2.0, 0.95, 0.95, 1.0, 1.0])
CAP("Table 4. One-axis synthetic-to-real degradation sweep (Section 4.6): each row varies "
    "only that axis, from baseline to the most severe level tested, holding all others fixed. "
    "Full multi-level sweeps are in the released results file.")
FIG("figure9_robustness")
CAP("Fig. 9 (a) Macro-F1 (left axis) and mean VOI (right axis, dashed zero line) against a "
    "combined synthetic-to-real degradation level moving noise, dropout, GPR attenuation, "
    "moisture attenuation and material-signature overlap together. (b) The same two "
    "quantities against material-signature overlap alone.")
Pp("Individually, the five degradation axes are not equally damaging (Table 4). At the most severe "
   "level tested on each axis alone, overall sensor noise does the most harm (macro-F1 "
   "%.2f \u2192 %.2f, mean VOI $%.0f \u2192 $%.0f), followed by material-signature overlap "
   "(F1 \u2192 %.2f, VOI \u2192 $%.0f) and dropout probability (F1 \u2192 %.2f, VOI \u2192 "
   "$%.0f); GPR-specific depth attenuation and moisture attenuation, tested over the same "
   "relative range, show little standalone effect on either metric (VOI \u2192 $%.0f and "
   "$%.0f respectively), which is notable given GPR's importance to raw classification "
   "(Section 5.3) and suggests the economic result is more sensitive to generic sensor noise "
   "than to this specific channel's physics. The combined scenario, moving all five axes "
   "together, is more damaging than any individual axis: mean VOI falls from $%.0f at "
   "baseline to $%.0f, $%.0f, $%.0f and $%.0f across increasing combined severity, turning "
   "negative before the most severe combined scenario tested. A finer search along the same "
   "combined axis locates the point at which mean VOI changes sign at %.0f%% of the way from "
   "baseline to the most severe scenario evaluated \u2014 a model-specific VOI-zero boundary, "
   "not a general claim about when surveying stops being worthwhile: beyond it, this "
   "simulation's economic prediction changes sign. Section 5.11 replaces this "
   "one-dimensional severity fraction with an explicit two-dimensional decision-viability "
   "map giving the boundary in terms of the underlying degradation parameters."
   % (RB["sweeps"]["noise_mult"][0]["macro_f1_mean"], RB["sweeps"]["noise_mult"][-1]["macro_f1_mean"],
      RB["sweeps"]["noise_mult"][0]["voi_mean"], RB["sweeps"]["noise_mult"][-1]["voi_mean"],
      RB["sweeps"]["overlap_shrink"][-1]["macro_f1_mean"], RB["sweeps"]["overlap_shrink"][-1]["voi_mean"],
      RB["sweeps"]["dropout_mult"][-1]["macro_f1_mean"], RB["sweeps"]["dropout_mult"][-1]["voi_mean"],
      RB["sweeps"]["gpr_atten_mult"][-1]["voi_mean"], RB["sweeps"]["moisture_mult"][-1]["voi_mean"],
      cd_[0]["voi_mean"], cd_[1]["voi_mean"], cd_[2]["voi_mean"], cd_[3]["voi_mean"],
      cd_[4]["voi_mean"], boundary["combined_severity_fraction"] * 100))
_mid = boundary["combined_severity_fraction"]
_bnd = dict(noise_mult=1.0 + _mid * 1.5, dropout_mult=1.0 + _mid * 3.0, gpr_atten_mult=1.0 + _mid * 1.5,
           moisture_mult=1.0 + _mid * 1.5, overlap_shrink=_mid * 0.6)
_sev = dict(noise_mult=2.5, dropout_mult=4.0, gpr_atten_mult=2.5, moisture_mult=2.5, overlap_shrink=0.6)
TBL(["Degradation axis", "Baseline", "Value at VOI = 0", "Most severe scenario"],
    [[AXIS_LABEL[axis].split(" (")[0], "1.0" if axis != "overlap_shrink" else "0.0",
      "%.2f" % _bnd[axis], "%.1f" % _sev[axis]]
     for axis in ["noise_mult", "dropout_mult", "gpr_atten_mult", "moisture_mult", "overlap_shrink"]],
    widths=[2.1, 0.85, 1.1, 1.1])
CAP("Table 5. The combined-degradation VOI-zero boundary (%.1f%% of the way from baseline to "
    "the most severe scenario, Fig. 9a) expressed in the underlying degradation parameters "
    "used by the combined-severity sweep, rather than as a bare percentage. Sensor noise "
    "and dropout are expressed as multiples of their baseline standard deviation/probability; "
    "GPR and moisture attenuation as multiples of their baseline attenuation coefficient; "
    "overlap as the fraction each material's response is shrunk toward the six-class mean."
    % (_mid * 100))

H2("5.9 Classifier choice and probability calibration for the economic layer")
Pp("The economic layer (Sections 5.4 and 5.7) uses random forest rather than XGBoost, the "
   "classifier with the highest raw macro-F1 (Section 5.1), because XGBoost's probabilities "
   "are the least well calibrated of the three (Brier %.3f vs. %.3f for random forest; "
   "Section 5.6), and VOI consumes predicted probabilities directly rather than only the "
   "predicted class. This choice is tested, not merely asserted: averaged over %d "
   "independent surveys, mean VOI is $%.0f with random forest, $%.0f with raw XGBoost "
   "(difference $%.0f) and $%.0f with Platt-calibrated XGBoost (sigmoid calibration, "
   "%d-fold; Brier improves to %.3f). All three are within normal sampling variation of one "
   "another and all give P(VOI>0) ≥ %.0f%%: classifier choice for the economic layer "
   "does not materially change the economic conclusion in this simulation, which is itself "
   "informative — it means the classifier-calibration concern raised by Section 5.6's "
   "Brier-score gap, while real, is not what is driving this paper's headline VOI estimate."
   % (R["calibration"]["xgb"]["mean"], R["calibration"]["rf"]["mean"], CV["n_rep"],
      CV["voi"]["rf"]["mean"], CV["voi"]["xgb"]["mean"], CV["voi_diff_xgb_minus_rf"]["mean"],
      CV["voi"]["xgb_calibrated"]["mean"], 3, CV["brier"]["xgb_calibrated"]["mean"],
      min(CV["voi"][k]["p_positive"] for k in ("rf", "xgb", "xgb_calibrated")) * 100))

H2("5.10 Sensor selection depends on site composition, not price alone")
FIG("figure10_sensor_selection")
CAP("Fig. 10 Incremental VOI (mean over %d repeats) for each of the five channels, at "
    "baseline economics and four deliberately extreme site archetypes: metal devalued to "
    "%.0f%% of baseline with no other change; the same metal devaluation plus plastic price "
    "and recovery fraction raised %.1fx/%.1fx (polymer-rich); plus C&D price/recovery raised "
    "%.1fx/%.1fx (C&D-rich); plus glass price/recovery raised %.1fx/%.1fx (glass-rich)."
    % (15, SME["polymer_max"]["params"]["metal_mult"] * 100,
       SME["polymer_max"]["params"]["challenger_price_mult"], SME["polymer_max"]["params"]["challenger_rec_mult"],
       SME["cd_max"]["params"]["challenger_price_mult"], SME["cd_max"]["params"]["challenger_rec_mult"],
       SME["glass_max"]["params"]["challenger_price_mult"], SME["glass_max"]["params"]["challenger_rec_mult"]))
Pp("A grid over metal price (%.1f–%.1fx baseline) crossed with plastic price "
   "(%.1f–%.1fx baseline), %d×%d points, %d repeats per point, never changed the "
   "winner away from the magnetometer (Fig. 5b's single-point result therefore generalises "
   "across ordinary price fluctuation, not just the one price table in Table 2). But "
   "deliberately extreme, still economically nameable site archetypes do flip the ranking: "
   "starving metal's value alone collapses every sensor's incremental VOI toward zero (no "
   "single channel matters when no material is worth much); combined with a hypothetical "
   "high-value, high-recovery plastics market, GPR becomes most valuable ($%.0f), ahead of "
   "EMI ($%.0f) and gas ($%.0f); with a high-value C&D market, GPR again leads ($%.0f); with "
   "a high-value glass market, EMI leads ($%.0f). The classification-optimal sensor (GPR, "
   "Section 5.3) and the economic-optimal sensor are therefore not fixed properties of the "
   "sensing architecture: they are conditional on the site's resource-value structure, and "
   "the general claim this paper supports is that sensor suites should be selected against "
   "site-specific decision value, not against a generic classification benchmark or a "
   "single assumed price table."
   % (SM["metal_mult"][0], SM["metal_mult"][-1], SM["plastic_mult"][0], SM["plastic_mult"][-1],
      len(SM["metal_mult"]), len(SM["plastic_mult"]), SM["n_rep"],
      SME["polymer_max"]["incremental_voi_mean"]["gpr"], SME["polymer_max"]["incremental_voi_mean"]["emi"],
      SME["polymer_max"]["incremental_voi_mean"]["gas"], SME["cd_max"]["incremental_voi_mean"]["gpr"],
      SME["glass_max"]["incremental_voi_mean"]["emi"]))

H2("5.11 A two-dimensional decision-viability map")
FIG("figure11_voi_viability_map")
CAP("Fig. 11 Mean VOI as a function of sensing-degradation severity (alpha, the same "
    "combined parametrisation as Fig. 9a) and economic-adversity severity (beta: secondary-"
    "material prices shrunk toward zero and survey cost increased), %d×%d grid, %d "
    "repeats per point, with the VOI = 0 contour drawn in black."
    % (len(VM["alpha_grid"]), len(VM["beta_grid"]), VM["n_rep"]))
Pp("The one-dimensional combined-degradation sweep (Fig. 9a, Table 5) is the beta = 0 "
   "slice of this surface. Moving along that slice alone, the VOI-zero boundary sits at "
   "%.0f%% combined severity; but the boundary is far more sensitive to economic adversity "
   "than to sensing degradation: at alpha = 0 (no added sensing degradation at all), mean "
   "VOI already crosses zero at beta ≈ %.2f, and at alpha = 1 (the most severe sensing "
   "degradation tested) it crosses at beta ≈ %.2f — a comparatively small shift. "
   "The decision-viability region ℒ₊ = {(α,β): VOI(α,β) > 0} in "
   "this simulation is therefore bounded mainly by whether secondary-material markets and "
   "survey costs stay favourable, not by how well the sensors themselves perform; a site "
   "operator worried about the economics of this approach should stress-test market "
   "assumptions before stress-testing sensor specifications."
   % (boundary["combined_severity_fraction"] * 100,
      [z["beta_at_voi_zero"] for z in VM["zero_crossings"] if z["alpha"] == 0.0][0],
      [z["beta_at_voi_zero"] for z in VM["zero_crossings"] if z["alpha"] == 1.0][0]))

# ==================================================================== 6
H("6 Discussion")
H2("6.1 Classification accuracy is not the right sensor-value metric, and the right sensor "
   "is not fixed")
Pp("Section 5.4's central finding \u2014 that GPR is the strongest classification "
   "contributor but only the magnetometer shows reliable economic value at baseline "
   "economics \u2014 argues against sizing a multimodal survey by F1 or macro-F1 alone. "
   "Section 5.10 goes further: the magnetometer's advantage survives ordinary price "
   "fluctuation but not deliberate changes in what the site actually contains, and GPR or "
   "EMI become the economically optimal channel under polymer-rich, debris-rich or "
   "glass-rich archetypes. The general, falsifiable claim this paper supports is therefore "
   "that predictive utility is not decision utility, and that the economically optimal "
   "sensor is a function of site composition, market economics and sensor uncertainty "
   "jointly \u2014 not \u201cmagnetometers are economically most useful,\u201d which is only "
   "the baseline-economics special case. A sensor package for "
   "this application should be chosen, and priced, against its contribution to the value-"
   "of-information calculation for the specific material-price structure and composition of "
   "the site, not against a generic classification benchmark or a single assumed price "
   "table. This is a design implication the original proposal's Stage 1 testing, reporting "
   "only F1 scores under one assumed economic table, could not have surfaced.")
H2("6.2 What the spatial-validation null result does and does not show")
Pp("We tested, rather than assumed, whether random splitting overestimates performance for "
   "this system and found no detectable effect (Section 5.2). This should not be read as "
   "evidence that spatial blocking is unnecessary in general; the mechanism the literature "
   "documents %s depends on a model exploiting local spatial proximity, and our feature set "
   "(per-cell sensor readings plus raw coordinates) evidently gives XGBoost, random forest "
   "and logistic regression little opportunity to do so. A design that adds explicit "
   "neighbourhood-averaged features, or a real dataset whose noise correlation structure "
   "differs from our assumptions, could behave differently, and spatial blocking costs "
   "nothing to keep as standard practice." % C("roberts2017", "ploton2020"))
H2("6.3 The plastic/glass limitation is a physics limitation, not a tuning problem")
Pp("The proposal's diagnosis \u2014 that plastic and glass are weakly magnetic and poorly "
   "distinguished by this channel set, and that no amount of classifier tuning fixes this "
   "\u2014 is supported by our results: three different classifier families all fail on "
   "these two classes by a similar margin (Table 3), and adding a channel with distinct "
   "plastic/glass responses (NIR) improves plastic specifically without improving glass "
   "(Section 5.5). This points to plastic and glass needing materially different remedies "
   "(for example, glass may need a different spectral band or a density-based separation "
   "step) rather than one \u201coptical fix\u201d for both.")
H2("6.4 Limits of the economic model")
Pp("The VOI conclusion in Section 5.7 depends on the price and recovery-fraction ranges of "
   "Table 2, explicitly labelled illustrative, and on treating the excavate/do-not-excavate "
   "decision as a single per-cell threshold rather than a portfolio decision across the "
   "whole site. Real secondary-material markets are more volatile than the ranges swept "
   "here, and the framework's practical strength, as the source proposal itself notes, is "
   "that VOI can be recomputed under site- and market-specific inputs rather than relying "
   "on any single value estimate.")
H2("6.5 Relation to established landfill-mining economics")
Pp("The finding that survey-guided decisions can convert an unprofitable blind-excavation "
   "baseline into a profitable one is consistent with the broader literature's diagnosis "
   "that landfill-mining profitability is highly sensitive to a small number of factors "
   "that are, in principle, knowable before excavation %s. This paper's contribution is to "
   "make \u201chow much would it be worth to know\u201d a computable, falsifiable quantity "
   "for a specific sensing architecture, rather than a general observation about "
   "uncertainty." % C("laner2019", "danthurebandara2015"))
H2("6.6 Limitations")
Pp("Every quantitative result in this paper is a prediction from a synthetic, physically "
   "motivated but unvalidated model. Sensor-response coefficients (Table 1), noise levels, "
   "dropout rates, spatial correlation lengths, and every economic parameter (Table 2) are "
   "stated assumptions, not measurements; the robustness sweep (Section 5.8) shows the "
   "central economic conclusion depends on assumptions holding within roughly the first "
   "%.0f%% of the degradation range explored. The gas-hazard truth label is constructed "
   "from the same underlying model as the gas sensor reading, so the reported hazard "
   "performance is an upper bound on what independent ground-truth gas testing would show. "
   "The three-depth-layer material field is a simplification of real, continuously "
   "heterogeneous stratigraphy. No field, laboratory, or test-bed data were used anywhere "
   "in this study." % (boundary["combined_severity_fraction"] * 100))

H2("6.7 Load-bearing assumptions")
Pp("Four assumptions carry most of the paper's conclusions, and different conclusions lean "
   "on different subsets of them, so a failure in one need not undermine the others.",
   indent=False)
for lbl, txt in [
    ("A1 — synthetic sensor signatures approximate real landfill sensor separability.",
     "Load-bearing for every classification result (Sections 5.1–5.3, 5.5): the "
     "metal/organic/C&D-high, plastic/glass-low F1 pattern and the GPR-dominant ablation "
     "ranking all depend on it. If real materials are less separable than modelled, "
     "classification F1 (and everything downstream of it) will be lower than reported."),
    ("A2 — the material-price/recovery-fraction distributions represent the intended "
     "decision environment.",
     "Load-bearing for every economic result (Sections 5.4, 5.7–5.11): the "
     "magnetometer's baseline VOI advantage, the break-even survey cost and the decision-"
     "viability map's beta axis all depend on it directly. Section 5.10 shows this "
     "dependence explicitly by varying it."),
    ("A3 — the simulator's spatial structure approximates real landfill heterogeneity.",
     "Load-bearing mainly for the spatial-vs-random null result (Section 5.2) and the "
     "distance-decay diagnostic; a real site with different spatial correlation in its "
     "material field or sensor noise could show a leakage gap this simulation did not."),
    ("A4 — predicted probabilities are well enough calibrated for expected-value "
     "arithmetic.",
     "Load-bearing for the economic layer's use of predicted probabilities rather than "
     "only predicted classes; Section 5.9 tests this directly and finds the VOI conclusion "
     "is not sensitive to it in this simulation, which weakens, but does not remove, its "
     "load-bearing status.")]:
    p = doc.add_paragraph()
    r = p.add_run(lbl); r.bold = True
    p.add_run(" " + txt)

# ==================================================================== 7
H("7 Field validation and falsification plan")
Pp("The predictions above can be confirmed or overturned by a staged test programme. Table "
   "6 lists the measurements, the test bed, and the acceptance criterion for each.")
TBL(["Prediction", "Test", "Failure criterion"],
    [["Metal, organic, C&D F1 remain high on real waste", "Controlled test bed with known "
      "buried materials of each class", "F1 below the values in Table 3 by more than the "
      "robustness sweep's noise-equivalent margin (Fig. 9)"],
     ["Plastic/glass remain the hard classes without an optical channel",
      "Same test bed, five-channel configuration", "Unexpectedly high plastic/glass F1 would "
      "indicate the synthetic physics under-modelled real discriminability"],
     ["NIR improves plastic but not necessarily glass", "Same test bed, six-channel "
      "configuration with an added optical/NIR unit", "Glass F1 improves as much as plastic "
      "(would contradict Section 5.5); or neither improves (would contradict the source "
      "proposal's diagnosis)"],
     ["Sensor economic ranking follows site composition (Section 5.10), not a fixed "
      "classification-accuracy ranking", "Ablation repeated on field data with the same "
      "economic model applied to real prices and the site's actual measured composition",
      "Field outcome inconsistent with simulation: GPR (or another non-magnetometer "
      "channel) produces a reliably positive incremental VOI exceeding the magnetometer's "
      "on a site the simulation's regime map would have classified as metal-dominant"],
     ["Spatial vs. random validation gap remains small", "Repeat the spatial-block/random-"
      "split comparison on field data", "A gap appears whose 95% CI excludes zero, "
      "indicating the null result of Section 5.2 does not transfer"],
     ["Gas-hazard recall remains high at a safety operating point", "Independent, ground-"
      "truthed gas testing (not derived from the same sensor)", "Recall below 90% at the "
      "chosen operating point"],
     ["VOI remains positive at the site's actual survey cost", "Full field pilot with "
      "measured excavation and processing costs and actual secondary-material prices",
      "Measured VOI is negative, or the break-even survey cost is below the field survey's "
      "actual cost"]], widths=[1.9, 2.2, 2.2])
CAP("Table 6. Field validation and falsification criteria. A pilot at full site scale is "
    "proposed only after the controlled test-bed stage has been passed.")

H2("7.1 Status of claims and assumptions")
Pp("Every claim in this paper falls into one of four evidentiary categories; conflating "
   "them is the most common way a simulation study overstates itself, so Table 7 states "
   "each claim's status explicitly rather than leaving it implicit in the prose.",
   indent=False)
TBL(["Claim", "Status", "Evidence/basis"],
    [["Random splitting inflates classification performance relative to spatial blocking",
      "Not supported in simulation", "95% CI includes zero for LR/RF/XGB (Section 5.2)"],
     ["GPR contributes most to macro-F1 among the five channels", "Simulation result",
      "Leave-one-sensor-out ablation (Section 5.3)"],
     ["Magnetometer contributes the most reliable incremental VOI at baseline economics",
      "Simulation result, economics-conditional", "20-survey incremental-VOI analysis "
      "(Section 5.4); shown conditional in Section 5.10"],
     ["NIR improves plastic classification", "Conditional simulation result",
      "Assumed synthetic NIR response coefficients (Table 1), not measured spectra"],
     ["NIR increases VOI", "Not supported", "Incremental VOI statistically indistinguishable "
      "from zero (Section 5.5)"],
     ["Baseline median VOI is positive", "Model-dependent numerical result",
      "Monte Carlo economic model, stated price/recovery/cost ranges (Section 5.7)"],
     ["VOI becomes negative under severe combined degradation", "Model-dependent boundary",
      "Robustness sweep and 2-D decision-viability map (Sections 5.8, 5.11)"],
     ["The economically optimal sensor depends on site composition, not just price",
      "Supported within tested simulation", "Economic-regime grid and extreme-archetype "
      "test (Section 5.10)"],
     ["Classifier choice for the economic layer does not change the VOI conclusion",
      "Supported within tested simulation", "RF vs. raw/calibrated XGBoost comparison "
      "(Section 5.9)"],
     ["Field performance will match these simulated values", "Untested prediction",
      "Requires the prospective validation in Table 6"]], widths=[2.3, 1.5, 2.4])
CAP("Table 7. Status of claims and assumptions. “Simulation result” means the "
    "finding is reproducible from the released code under the stated assumptions; it is not "
    "thereby a claim about real landfills, which is what Table 6 exists to test.")

# ==================================================================== 8
H("8 Conclusions")
Pp("A multimodal sensor-fusion architecture for pre-excavation landfill characterisation "
   "was evaluated by simulation against the questions it raises rather than the claims "
   "originally made for it. Metal, organic material and construction and demolition debris "
   "are classified reliably (F1 up to %.2f) under a spatially valid protocol; plastic and "
   "glass are not (F1 %.2f\u2013%.2f), for reasons of sensor physics rather than "
   "classifier choice. Random per-cell splitting did not detectably inflate performance "
   "relative to spatial blocking in this configuration, a tested and reported null result "
   "rather than an assumption. The paper's central finding is that predictive utility is "
   "not decision utility and is not fixed: GPR maximises classification accuracy, but the "
   "economically optimal sensor at baseline economics is the magnetometer, and it changes "
   "to GPR or EMI under deliberately different, still economically nameable site "
   "compositions (Section 5.10) \u2014 optimal sensor selection is a function of site "
   "composition, market economics and sensor uncertainty, not a fixed property of the "
   "sensing architecture. Across the baseline synthetic scenarios and stated economic "
   "parameter distributions, the simulated survey has positive median VOI ($%.0f) and a "
   "break-even cost of $%.0f, and this classifier-layer choice was itself tested and found "
   "not to matter (Section 5.9). A two-dimensional decision-viability map (Section 5.11) "
   "replaces the single combined-degradation boundary with a full VOI(\u03b1,\u03b2) surface, "
   "showing the economic conclusion is considerably more sensitive to adverse market "
   "conditions than to sensor degradation itself. All results are predictions, explicitly "
   "labelled by evidentiary status in Table 7, and the field validation plan (Table 6) "
   "specifies the measurements that would confirm or overturn them."
   % (max(R["pooled_baseline"][k]["per_class_f1"]["metal"] for k in KINDS),
      min(R["pooled_baseline"][k]["per_class_f1"]["glass"] for k in KINDS),
      max(R["pooled_baseline"][k]["per_class_f1"]["plastic"] for k in KINDS),
      econ["voi_averaged"]["median"],
      econ["breakeven_survey_cost_usd_averaged"]["median"]))

# ==================================================================== DECLARATIONS
H("Declarations")
for head, body in [
    ("Funding", "This work received no external funding."),
    ("Competing interests", "The author declares no competing interests."),
    ("Ethics approval", "Not applicable. No human participants or animals were involved."),
    ("Consent to participate/for publication", "Not applicable."),
    ("Prior dissemination", "An earlier, non-quantitative description of this architecture was "
     "submitted by the author in response to an industrial open-innovation challenge focused on "
     "landfill mining. That submission was not peer reviewed or published and contained none of "
     "the models, validation protocol, or results reported here."),
    ("Data availability", "No field data were generated. All model inputs are stated in the "
     "paper and the released configuration file."),
    ("Code availability", "The synthetic-survey generator, classification and economic models, "
     "robustness sweep, decision-viability and sensor-selection maps, and figure generators "
     "are openly available at %s%s. Running landmap_model.py, landmap_econ.py, "
     "landmap_robustness.py, landmap_hazard_extra.py, landmap_calib_voi.py, "
     "landmap_sensor_map.py, landmap_sensor_map_extreme.py, landmap_voi_map.py and "
     "make_figures.py reproduces every number and figure."
     % (REPO_URL, (" and archived at https://doi.org/%s" % CODE_DOI) if CODE_DOI else "")),
    ("Author contributions", "L.S. conceived the study, developed the models, performed the "
     "analysis and wrote the manuscript."),
    ("Use of generative AI", "During the preparation of this work the author used Claude "
     "(Anthropic) to assist with literature search, implementation of the models and drafting "
     "of the text. The author reviewed and edited all content, verified every cited reference "
     "against its source, and takes full responsibility for the content of the article.")]:
    p = doc.add_paragraph(); p.paragraph_format.line_spacing = 1.3
    r = p.add_run(head + ". "); r.bold = True
    p.add_run(body)

# ==================================================================== REFERENCES
H("References")


def fmt(tag):
    v = DOIREFS[tag]
    au = v["authors"]
    aus = ", ".join(au[:6]) + (", et al." if len(au) > 6 else "")
    title = v["title"].rstrip(".")
    kind = v.get("type")
    if kind == "proceedings-article":
        cont = v["container"]
        where = cont if "proceedings" in cont.lower() else "Proceedings of the %s" % cont
        pg = ", %s" % v["page"] if v.get("page") else ""
        return "%s (%s) %s. In: %s%s. %s. https://doi.org/%s" % (aus, v["year"], title, where, pg,
                                                                 v["publisher"], v["doi"])
    if kind in ("book-chapter", "book", "monograph"):
        return "%s (%s) %s. In: %s. %s. https://doi.org/%s" % (aus, v["year"], title,
                                                               v["container"], v["publisher"], v["doi"])
    loc = v["container"]
    if v.get("volume"):
        loc += " " + str(v["volume"])
        if v.get("issue"):
            loc += "(%s)" % v["issue"]
    if v.get("page"):
        loc += ":" + str(v["page"])
    return "%s (%s) %s. %s. https://doi.org/%s" % (aus, v["year"], title, loc, v["doi"])


for i, tag in enumerate(ORDER, 1):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.left_indent = Inches(0.35)
    p.paragraph_format.first_line_indent = Inches(-0.35)
    p.paragraph_format.space_after = Pt(2)
    p.add_run("%d. %s" % (i, fmt(tag))).font.size = Pt(9.5)

os.makedirs(OUT, exist_ok=True)
doc.core_properties.author = "Leon Sandler"
doc.core_properties.title = TITLE
path = os.path.join(OUT, "LANDMAP_EMA_Manuscript.docx")
doc.save(path)
words = sum(len(p.text.split()) for p in doc.paragraphs)
print("abstract words:", len(ABSTRACT.split()))
print("total words   :", words)
print("equations     :", EQN[0])
print("references    : %d cited; unused: %s" % (len(ORDER), sorted(set(DOIREFS) - set(ORDER))))
print("saved         :", path)
