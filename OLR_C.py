"""
Wordle Difficulty Model v5
===========================
Main idea
---------
Predict detrended mean attempts as a regression problem, then map the predicted
detrended score into quartile-based difficulty classes.

Why v5?
-------
* Stable: no OrderedModel / no Hessian / no constant-column crashes
* Interpretable: Ridge coefficients + RF feature importance
* Practical: class labels come from regression output, not from forcing noisy
  quartiles into an ordinal likelihood model
* Robust to missing phoneme data: phoneme features are skipped if unavailable

Target
------
emp_mean = expected attempts from response distribution
y_dtr    = detrended target = emp_mean - linear contest trend

Classes
-------
Easy / Medium / Hard / Very Hard are quartiles of y_dtr.
"""

import math
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# Optional phoneme support
# ═══════════════════════════════════════════════════════════════
try:
    import pronouncing
    HAS_PRONOUNCING = True
except ImportError:
    print("WARNING: 'pronouncing' not installed. Phoneme features will be skipped.\n")
    HAS_PRONOUNCING = False

# ═══════════════════════════════════════════════════════════════
# 1. LOAD & PARSE DATA
# ═══════════════════════════════════════════════════════════════
file_path = r"C:\Users\natas\Downloads\data.xlsx"

df_raw = pd.read_excel(file_path, header=None)
header_row = next(
    i for i, row in df_raw.iterrows()
    if any("word" in str(x).lower() for x in row)
)

df = pd.read_excel(file_path, header=header_row)
df.columns = df.columns.astype(str).str.strip().str.lower()

word_col = next(c for c in df.columns if c == "word")
date_col = next(c for c in df.columns if "date" in c)
contest_col = next(c for c in df.columns if "contest" in c)
N_col = next(c for c in df.columns if "reported" in c)
hardmode_col = next(c for c in df.columns if "hard" in c)

WORDS = df[word_col].astype(str).str.lower().str.strip().tolist()

ATTEMPT_COLS = [c for c in [
    "1 try", "2 tries", "3 tries", "4 tries", "5 tries", "6 tries",
    "7 or more tries (x)",
] if c in df.columns]

if not ATTEMPT_COLS:
    raise ValueError("No attempt columns were found in the workbook.")

df[ATTEMPT_COLS] = df[ATTEMPT_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)
df[N_col] = pd.to_numeric(df[N_col], errors="coerce")
df[contest_col] = pd.to_numeric(df[contest_col], errors="coerce")
df[hardmode_col] = pd.to_numeric(df[hardmode_col], errors="coerce")
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

raw = df[ATTEMPT_COLS].values / 100.0
denom = raw.sum(axis=1, keepdims=True)
denom[denom == 0] = 1.0
emp_probs = raw / denom
emp_mean = (emp_probs * np.arange(1, 8)).sum(axis=1)

df["hard_mode_frac"] = df[hardmode_col] / df[N_col].replace(0, np.nan)
df["is_weekend"] = (df[date_col].dt.dayofweek >= 5).astype(int)
c_min = df[contest_col].min()
c_max = df[contest_col].max()
df["contest_norm"] = (df[contest_col] - c_min) / (c_max - c_min)

# ═══════════════════════════════════════════════════════════════
# 2. TEMPORAL DETRENDING
# ═══════════════════════════════════════════════════════════════
cn_vals = df["contest_norm"].values
trend_fit = np.polyfit(cn_vals, emp_mean, deg=1)
trend_fn = np.poly1d(trend_fit)
trend_hat = trend_fn(cn_vals)
y_dtr = emp_mean - trend_hat
TREND_END = float(trend_fn(1.0))

print(f"Temporal trend:  mean_attempts = "
      f"{trend_fit[0]:+.4f}·contest_norm + {trend_fit[1]:+.4f}")
print(f"Trend at plateau (end-2022): {TREND_END:.3f} attempts\n")

# ═══════════════════════════════════════════════════════════════
# 3. FREQUENCY TABLES
# ═══════════════════════════════════════════════════════════════
LETTER_FREQ = {
    'e': 12.70, 't': 9.06, 'a': 8.17, 'o': 7.51, 'i': 6.97, 'n': 6.75, 's': 6.33,
    'h': 6.09, 'r': 5.99, 'd': 4.25, 'l': 4.03, 'c': 2.78, 'u': 2.76, 'm': 2.41,
    'w': 2.36, 'f': 2.23, 'g': 2.02, 'y': 1.97, 'p': 1.93, 'b': 1.49, 'v': 0.98,
    'k': 0.77, 'j': 0.15, 'x': 0.15, 'q': 0.10, 'z': 0.07,
}
POS_FREQ = {
    0: {'s': 15.8, 'c': 8.6, 'b': 7.7, 't': 7.3, 'p': 7.2, 'a': 6.8, 'f': 5.5},
    1: {'a': 15.0, 'o': 11.1, 'e': 10.0, 'i': 8.5, 'u': 6.7, 'r': 5.5, 'l': 5.3},
    2: {'a': 8.5, 'i': 8.5, 'o': 7.8, 'e': 7.6, 'u': 5.9, 'r': 5.5, 'n': 5.4},
    3: {'e': 13.6, 'n': 7.0, 's': 6.5, 'a': 6.3, 'i': 6.0, 'l': 5.4, 'r': 5.1},
    4: {'e': 19.3, 's': 17.0, 'y': 14.3, 't': 7.6, 'r': 7.1, 'l': 5.8, 'n': 4.6},
}
def pfreq(letter, position):
    return POS_FREQ.get(position, {}).get(letter, LETTER_FREQ.get(letter, 0.5) / 5)

# ═══════════════════════════════════════════════════════════════
# 4. CORPUS-LEVEL STATISTICS
# ═══════════════════════════════════════════════════════════════
bigram_counts = Counter()
for w in WORDS:
    for i in range(4):
        bigram_counts[(w[i], w[i + 1])] += 1

total_bigrams = sum(bigram_counts.values())
BIGRAM_PROB = {
    bg: (cnt + 1) / (total_bigrams + 26 * 26)
    for bg, cnt in bigram_counts.items()
}
BIGRAM_UNK = 1 / (total_bigrams + 26 * 26)

VOWELS = set("aeiou")
def cv_pattern(w):
    return "".join("V" if c in VOWELS else "C" for c in w)

cv_counts = Counter(cv_pattern(w) for w in WORDS)
total_cv = len(WORDS)
CV_PROB = {pat: cnt / total_cv for pat, cnt in cv_counts.items()}

# ═══════════════════════════════════════════════════════════════
# 5. PHONEME FEATURES
# ═══════════════════════════════════════════════════════════════
RARE_PHONEMES = {"ZH", "AW", "OY", "UH", "AO", "EH", "AH", "IH", "OW"}
PHONEME_FREQ = {
    "N": 7.1, "T": 6.9, "S": 5.8, "L": 5.5, "R": 5.0, "D": 4.8, "M": 4.2, "K": 3.7,
    "Z": 3.5, "IH1": 3.3, "IH0": 3.0, "EH1": 2.9, "AE1": 2.8, "AH0": 2.7, "AH1": 2.5,
    "ER0": 2.4, "ER1": 2.2, "IY1": 2.0, "IY0": 1.8, "AY1": 1.7, "AO1": 1.6, "OW1": 1.5,
    "EY1": 1.4, "UW1": 1.2, "AW1": 0.9, "OY1": 0.6, "UH1": 0.5,
}
_phoneme_cache = {}

def get_phonemes(word):
    if not HAS_PRONOUNCING:
        return None
    if word in _phoneme_cache:
        return _phoneme_cache[word]
    entries = pronouncing.phones_for_word(word)
    result = entries[0].split() if entries else None
    _phoneme_cache[word] = result
    return result

def phoneme_features(word):
    phones = get_phonemes(word)
    if phones is None:
        return {}
    is_vow = lambda p: p[-1].isdigit()
    base = [p.rstrip("012") for p in phones]
    fvals = [PHONEME_FREQ.get(p, 1.0) for p in base]
    n_cc = 0
    run = 0
    for p in phones:
        if not is_vow(p):
            run += 1
            n_cc += int(run >= 2)
        else:
            run = 0
    return {
        "n_phonemes": len(phones),
        "n_syllables": sum(1 for p in phones if is_vow(p)),
        "n_unique_phonemes": len(set(phones)),
        "phoneme_freq_mean": float(np.mean(fvals)),
        "phoneme_freq_min": float(min(fvals)),
        "n_rare_phonemes": sum(1 for p in base if p in RARE_PHONEMES),
        "n_consec_vowel_ph": sum(
            1 for i in range(len(phones) - 1)
            if is_vow(phones[i]) and is_vow(phones[i + 1])
        ),
        "n_consonant_clust": n_cc,
        "in_cmu_dict": 1,
    }

# ═══════════════════════════════════════════════════════════════
# 6. LETTER FEATURES
# ═══════════════════════════════════════════════════════════════
def letter_features(word, all_words):
    w = word.lower()
    counts = Counter(w)

    mean_freq_uniq = float(np.mean([LETTER_FREQ.get(c, 0.1) for c in set(w)]))
    min_freq_uniq = float(min(LETTER_FREQ.get(c, 0.1) for c in set(w)))
    n_unique = len(set(w))
    max_letter_count = int(max(counts.values()))
    repeat_info_loss = math.log2(
        math.factorial(5) / math.prod(math.factorial(v) for v in counts.values())
    )

    # positional surprise
    pos_surprise = []
    for i, c in enumerate(w):
        all_p = [pfreq(l, i) for l in "abcdefghijklmnopqrstuvwxyz"]
        rank = sum(1 for p in all_p if p < pfreq(c, i)) / 26.0
        pos_surprise.append(1.0 - rank)
    positional_surprise = float(np.mean(pos_surprise))

    # CV rarity
    pat = cv_pattern(w)
    cv_prob = CV_PROB.get(pat, 0.5 / total_cv)
    cv_pat_rarity = float(-math.log2(cv_prob))

    # bigram log prob
    bigram_log_prob = float(sum(
        math.log(BIGRAM_PROB.get((w[i], w[i + 1]), BIGRAM_UNK))
        for i in range(4)
    ))

    # vowel structure
    is_v = lambda c: c in "aeiou"
    n_vowels = sum(1 for c in w if is_v(c))
    vowel_cluster = 0
    run = 0
    for c in w:
        if is_v(c):
            run += 1
            vowel_cluster = max(vowel_cluster, run)
        else:
            run = 0
    starts_vowel = int(is_v(w[0]))
    ends_vowel = int(is_v(w[4]))

    # neighborhood
    same_pos = lambda other: sum(a == b for a, b in zip(w, other))
    n_neighbors_3 = sum(1 for ow in all_words if ow != w and same_pos(ow) >= 3)
    n_set_overlap = sum(
        1 for ow in all_words if ow != w and len(set(ow) & set(w)) >= 4
    )

    # entropy / surface
    probs = np.array(list(counts.values())) / 5.0
    letter_entropy = float(scipy_entropy(probs, base=2))

    common_endings = ["atch", "ound", "ight", "ould", "tion", "ally"]
    ends_ambiguous = int(any(w.endswith(e) for e in common_endings))

    n_consec = sum(1 for i in range(4) if w[i] == w[i + 1])

    return {
        "repeat_info_loss": repeat_info_loss,
        "positional_surprise": positional_surprise,
        "cv_pat_rarity": cv_pat_rarity,
        "bigram_log_prob": bigram_log_prob,
        "mean_freq_uniq": mean_freq_uniq,
        "min_freq_uniq": min_freq_uniq,
        "n_unique": n_unique,
        "max_letter_count": max_letter_count,
        "n_vowels": n_vowels,
        "vowel_cluster": vowel_cluster,
        "starts_vowel": starts_vowel,
        "ends_vowel": ends_vowel,
        "n_neighbors_3": n_neighbors_3,
        "n_set_overlap": n_set_overlap,
        "letter_entropy": letter_entropy,
        "ends_ambiguous": ends_ambiguous,
        "n_consec": n_consec,
    }

# ═══════════════════════════════════════════════════════════════
# 7. BUILD FEATURE MATRIX
# ═══════════════════════════════════════════════════════════════
print("Extracting features ...")
records = []
for i, w in enumerate(WORDS):
    feats = letter_features(w, WORDS)
    pf = phoneme_features(w)
    if pf:
        feats.update(pf)
    feats.update({
        "hard_mode_frac": float(df["hard_mode_frac"].iloc[i]),
        "is_weekend": int(df["is_weekend"].iloc[i]),
        "word": w,
    })
    records.append(feats)

feat_df = pd.DataFrame(records).set_index("word")
feat_df["emp_mean"] = emp_mean
feat_df["y_dtr"] = y_dtr

# raw feature columns
FEATURE_COLS_ALL = [c for c in feat_df.columns if c not in ("emp_mean", "y_dtr")]
X_all = feat_df[FEATURE_COLS_ALL].copy()

# remove constant columns globally
const_cols = X_all.columns[X_all.nunique(dropna=False) <= 1].tolist()
if const_cols:
    print("Dropping constant columns:", const_cols)
    X_all = X_all.drop(columns=const_cols)

# remove highly correlated columns globally for cleaner interpretation
def drop_high_corr_columns(df, threshold=0.95):
    if df.shape[1] <= 1:
        return df, []
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
    return df.drop(columns=to_drop), to_drop

X_all, corr_dropped = drop_high_corr_columns(X_all, threshold=0.95)
if corr_dropped:
    print("Dropping correlated columns:", corr_dropped)

FEATURE_COLS = X_all.columns.tolist()
X = X_all.values.astype(float)
y = feat_df["y_dtr"].values

print(f"Feature matrix   : {X.shape[0]} words × {X.shape[1]} features")
n_lf = len(letter_features("crane", WORDS))
n_pf = len(phoneme_features("crane"))
print(f"  Letter         : {n_lf}  |  Phoneme : {n_pf if n_pf else 0}  |  Context : 2\n")

# ═══════════════════════════════════════════════════════════════
# 8. LABELS
# ═══════════════════════════════════════════════════════════════
q1, q2, q3 = np.percentile(y, [25, 50, 75])
print(f"Detrended quartile thresholds:  Q1={q1:+.4f}  Q2={q2:+.4f}  Q3={q3:+.4f}")

LABELS = ["Easy", "Medium", "Hard", "Very Hard"]

def dtr_to_class(v):
    if v <= q1:
        return "Easy"
    if v <= q2:
        return "Medium"
    if v <= q3:
        return "Hard"
    return "Very Hard"

feat_df["true_class"] = [dtr_to_class(v) for v in y]
y_class = np.array([dtr_to_class(v) for v in y])

# ═══════════════════════════════════════════════════════════════
# 9. REGRESSION MODELS
# ═══════════════════════════════════════════════════════════════
kf = KFold(n_splits=5, shuffle=True, random_state=42)

ridge = Pipeline([
    ("sc", StandardScaler()),
    ("m", Ridge(alpha=1.0))
])

rf = RandomForestRegressor(
    n_estimators=500,
    min_samples_leaf=5,
    max_features=0.5,
    random_state=42,
    n_jobs=-1
)

ridge_r2 = cross_val_score(ridge, X, y, cv=kf, scoring="r2")
ridge_mae = -cross_val_score(ridge, X, y, cv=kf, scoring="neg_mean_absolute_error")
rf_r2 = cross_val_score(rf, X, y, cv=kf, scoring="r2")
rf_mae = -cross_val_score(rf, X, y, cv=kf, scoring="neg_mean_absolute_error")

print("\n--- Regression CV (detrended target) ---")
print(f"Ridge  R²: {ridge_r2.mean():.3f} ± {ridge_r2.std():.3f} | "
      f"MAE: {ridge_mae.mean():.3f} ± {ridge_mae.std():.3f}")
print(f"RF     R²: {rf_r2.mean():.3f} ± {rf_r2.std():.3f} | "
      f"MAE: {rf_mae.mean():.3f} ± {rf_mae.std():.3f}")

# ═══════════════════════════════════════════════════════════════
# 10. FIT FINAL MODELS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("Final regression models")
print("=" * 65)

ridge.fit(X, y)
rf.fit(X, y)

ridge_pred_dtr = ridge.predict(X)
rf_pred_dtr = rf.predict(X)

ridge_acc = np.mean(np.array([dtr_to_class(v) for v in ridge_pred_dtr]) == feat_df["true_class"].values)
rf_acc = np.mean(np.array([dtr_to_class(v) for v in rf_pred_dtr]) == feat_df["true_class"].values)

ridge_kappa = cohen_kappa_score(
    feat_df["true_class"],
    [dtr_to_class(v) for v in ridge_pred_dtr],
    weights="quadratic"
)
rf_kappa = cohen_kappa_score(
    feat_df["true_class"],
    [dtr_to_class(v) for v in rf_pred_dtr],
    weights="quadratic"
)

print(f"\nIn-sample class accuracy (Ridge) : {ridge_acc*100:.1f}%")
print(f"In-sample class κ (Ridge)        : {ridge_kappa:.3f}")
print(f"In-sample class accuracy (RF)     : {rf_acc*100:.1f}%")
print(f"In-sample class κ (RF)            : {rf_kappa:.3f}")

# ═══════════════════════════════════════════════════════════════
# 11. INTERPRETABILITY: RIDGE COEFFICIENTS
# ═══════════════════════════════════════════════════════════════
ridge_coef = ridge.named_steps["m"].coef_
coef_df = (
    pd.DataFrame({
        "feature": FEATURE_COLS,
        "coef": ridge_coef
    })
    .assign(abs_coef=lambda d: d["coef"].abs())
    .sort_values("abs_coef", ascending=False)
)

print("\nRidge coefficients on standardized features (+ harder, − easier):")
print(f"{'Feature':24s}  {'Coef':>8}  Direction")
print("-" * 54)
for _, row in coef_df.iterrows():
    direction = "harder" if row.coef > 0 else "easier"
    bar = "|" * min(int(abs(row.coef) * 8), 30)
    print(f"  {row.feature:22s}  {row.coef:+8.4f}  {direction:7s}  {bar}")

print("\nRandom forest feature importance:")
rf_imp = (
    pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": rf.feature_importances_
    })
    .sort_values("importance", ascending=False)
)
for _, row in rf_imp.head(12).iterrows():
    print(f"  {row.feature:22s}  {row.importance:.4f}")

# ═══════════════════════════════════════════════════════════════
# 12. PREDICT EERIE
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PREDICTION: EERIE  (March 1 2023 — Wednesday)")
print("=" * 65)

hm_mean = float(df["hard_mode_frac"].mean())
eerie_feats = letter_features("eerie", WORDS)
pf_eerie = phoneme_features("eerie")
if pf_eerie:
    eerie_feats.update(pf_eerie)
eerie_feats.update({
    "hard_mode_frac": hm_mean,
    "is_weekend": 0,
})

eerie_df = pd.DataFrame([eerie_feats]).reindex(columns=FEATURE_COLS, fill_value=0.0)

eerie_ridge_dtr = float(ridge.predict(eerie_df.values)[0])
eerie_rf_dtr = float(rf.predict(eerie_df.values)[0])

eerie_ridge_raw = eerie_ridge_dtr + TREND_END
eerie_rf_raw = eerie_rf_dtr + TREND_END

eerie_ridge_class = dtr_to_class(eerie_ridge_dtr)
eerie_rf_class = dtr_to_class(eerie_rf_dtr)

print("\nRidge prediction:")
print(f"  detrended score : {eerie_ridge_dtr:+.3f}")
print(f"  raw approx      : {eerie_ridge_raw:.2f}")
print(f"  class           : {eerie_ridge_class}")

print("\nRF prediction:")
print(f"  detrended score : {eerie_rf_dtr:+.3f}")
print(f"  raw approx      : {eerie_rf_raw:.2f}")
print(f"  class           : {eerie_rf_class}")

# simple interval from RF residuals
rf_train_pred = rf.predict(X)
residuals = feat_df["emp_mean"].values - (rf_train_pred + trend_hat)
ci_lo = eerie_rf_raw + np.percentile(residuals, 5)
ci_hi = eerie_rf_raw + np.percentile(residuals, 95)
print(f"RF 90% pred interval : [{ci_lo:.2f}, {ci_hi:.2f}]")

print("\nKey EERIE features:")
eerie_show = [
    ("repeat_info_loss", eerie_feats["repeat_info_loss"]),
    ("positional_surprise", eerie_feats["positional_surprise"]),
    ("cv_pat_rarity", eerie_feats["cv_pat_rarity"]),
    ("bigram_log_prob", eerie_feats["bigram_log_prob"]),
    ("mean_freq_uniq", eerie_feats["mean_freq_uniq"]),
    ("n_unique", eerie_feats["n_unique"]),
    ("n_vowels", eerie_feats["n_vowels"]),
    ("vowel_cluster", eerie_feats["vowel_cluster"]),
    ("letter_entropy", eerie_feats["letter_entropy"]),
    ("n_consec", eerie_feats["n_consec"]),
]
for name, val in eerie_show:
    print(f"  {name:20s} = {val:.4f}")

# ═══════════════════════════════════════════════════════════════
# 13. SANITY-CHECK TABLE
# ═══════════════════════════════════════════════════════════════
check_words = [
    "eerie", "jazzy", "queue", "crane", "slate", "stare",
    "audio", "fuzzy", "knoll", "tryst", "mummy", "parer"
]

print("\n--- Sanity-check: notable words ---")
print(f"{'Word':10s}  {'Actual':>8}  {'Ridge':>10}  {'RF':>10}  {'TrueClass':>10}")
print("-" * 58)

for w in check_words:
    lf = letter_features(w, WORDS)
    pf = phoneme_features(w)
    ff = {**lf, **pf, "hard_mode_frac": hm_mean, "is_weekend": 0}
    row = pd.DataFrame([ff]).reindex(columns=FEATURE_COLS, fill_value=0.0)

    ridge_dtr = float(ridge.predict(row.values)[0])
    rf_dtr = float(rf.predict(row.values)[0])

    ridge_cls = dtr_to_class(ridge_dtr)
    rf_cls = dtr_to_class(rf_dtr)

    actual_str = "N/A"
    true_cls = "(OOS)"
    if w in feat_df.index:
        actual_str = f"{feat_df.loc[w, 'emp_mean']:.2f}"
        true_cls = feat_df.loc[w, "true_class"]

    print(f"{w:10s}  {actual_str:>8}  {ridge_cls:>10}  {rf_cls:>10}  {true_cls:>10}")