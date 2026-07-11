"""
End-to-end EEG ADHD vs Control classification pipeline.

============================================================================
HIS-MATCHED BASELINE VARIANT
----------------------------------------------------------------------------
This is the Alim & Imtiaz linear-feature baseline (https://www.mdpi.com/2624-6120/4/1/10), re-instrumented so its
reported estimator is IDENTICAL IN KIND to the HMM-GMM stationary-RKHS (HIS)
nested-CV engine, so the two are directly comparable:

  * Headline metric = MEAN over the 25 outer folds (not subject-pooled).
  * Bootstrap 95% CI resamples the 25 fold values (n=2000), matching HIS's
    aggregate_results/bootstrap_ci -- NOT the 121 subjects.
  * Pooled confusion matrix = SUM of the 25 fold confusion matrices (each
    subject counted once per repeat), matching HIS.
  * Fold construction = manual repeat loop with StratifiedKFold(5, shuffle=True,
    random_state = outer_cv_base_seed + repeat), base seed 0 -- identical
    PROCEDURE to HIS (fold ids r{repeat}f{fold}). With canonical_sort and a
    matching subject_id, the 25 outer partitions become bit-identical to HIS,
    which is what makes a fold-PAIRED test valid (see compare_folds_paired.py).
  * Inner tuning = StratifiedKFold(3) (HIS uses N_INNER_SPLITS=3).
  * Permutation test = HIS protocol: fix hyperparameters at the modal fold
    selection, permute labels, re-run the SAME 25-fold CV, statistic = mean
    fold balanced accuracy, n=1000.
  * Per-classifier outputs mirror HIS's file layout (fold_results.csv,
    metric_summary.csv, subject_stability.csv, permutation_scores.csv,
    permutation_result.txt) so downstream pairing reads both the same way.

The MODEL is unchanged from the baseline (StandardScaler -> PCA -> {kNN|SVM});
only the evaluation/reporting harness is aligned. 
============================================================================

Reimplementation and leakage-corrected extension of:
    Alim, A.; Imtiaz, M.H. "Automatic Identification of Children with ADHD from
    EEG Brain Waves." Signals 2023, 4, 193-205. doi:10.3390/signals4010010

What is faithful to the paper
-----------------------------
  * Preprocessing: 4th-order Butterworth band-pass (0.5-63 Hz) + 50 Hz notch
    (49-51 Hz stop-band), applied zero-phase (the paper's "zero phase distortion").
  * Band decomposition into delta (0.5-4), theta (4-8), alpha (8-13), beta (13-30).
    Gamma is dropped, exactly as in the paper.
  * 2 s windows with 50% overlap.
  * 11 features per channel per band (std, RMS, skewness, kurtosis, Hjorth
    activity/mobility/complexity, Shannon entropy, spectral entropy, mean PSD,
    band power).
  * StandardScaler -> PCA (explained-variance) -> classifier, with PCA variance
    tuned inside the CV.

What is deliberately different
-------------------------------------------
  * EVALUATION IS SUBJECT-LEVEL. The paper splits at the *segment* level with
    plain random folds, so segments from one subject appear in both train and
    test. That leakage is almost certainly why it reports 94%. Here, segment
    features are pooled per subject (mean + std by default) into ONE vector per
    subject, so the 5x5 nested CV is 25 genuinely leakage-free evaluations and
    every confusion matrix is per-subject. (A segment-level + subject-grouped
    majority-vote variant is a natural extension via CONFIG.evaluation, but
    only the subject-level pooling path is implemented here.)
  * Channel subset is configurable; default is the 7-channel frontal montage
    (Fp1, F7, F3, Fz, F4, F8, Fp2).

Outputs: per-fold metrics (mean +/- std over 25 outer folds), pooled per-subject
confusion matrix, bootstrap 95% CIs (n=2000), and a permutation test (n=1000),
for both kNN and Gaussian-RBF SVM.

Designed for Kaggle: checkpoints intermediate artifacts and
flushes stdout for the buffered console.
"""

from __future__ import annotations

import os
import re
import sys
import time
import json
import pickle
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy import stats as sp_stats
from scipy.io import loadmat

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def log(msg: str) -> None:
    print(msg, flush=True)


try:
    # Force line buffering so progress shows up live on Kaggle.
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except Exception:
    pass


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# Canonical channel order of the IEEE DataPort ADHD/Control dataset (19 ch).
# The paper uses the older T3/T4/T5/T6 labels; those are the same electrodes as
# T7/T8/P7/P8.
CANONICAL_CHANNELS: List[str] = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz",
]

FRONTAL_CHANNELS: List[str] = ["Fp1", "F7", "F3", "Fz", "F4", "F8", "Fp2"]

# Sub-bands actually used (gamma dropped, per the paper).
BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

FEATURE_NAMES: List[str] = [
    "std", "rms", "skewness", "kurtosis",
    "hjorth_activity", "hjorth_mobility", "hjorth_complexity",
    "shannon_entropy", "spectral_entropy", "mean_psd", "band_power",
]


@dataclass
class Config:
    # Data ---------------------------------------------------------------- #
    adhd_dir: str = ".../adhd-control/ADHD"
    control_dir: str = ".../adhd-control/Control"
    fs: float = 128.0
    channel_selection: object = "frontal"          # 'frontal' | 'all' | list[str]
    canonical_channels: List[str] = field(default_factory=lambda: list(CANONICAL_CHANNELS))

    # Preprocessing ------------------------------------------------------- #
    bandpass: Tuple[float, float] = (0.5, 63.0)
    notch: Tuple[float, float] = (49.0, 51.0)
    filter_order: int = 4
    bands: Dict[str, Tuple[float, float]] = field(default_factory=lambda: dict(BANDS))
    window_sec: float = 2.0
    overlap: float = 0.5

    # Evaluation ---------------------------------------------------------- #
    evaluation: str = "subject"                    # 'subject' (pool) | 'segment' (grouped + vote)
    pooling: Tuple[str, ...] = ("mean", "std")     # subject-level pooling statistics
    outer_splits: int = 5
    outer_repeats: int = 5                         # 5 x 5 = 25 outer evaluations
    inner_splits: int = 3                          # HIS-matched (HIS uses N_INNER_SPLITS=3)
    outer_cv_base_seed: int = 0                    # HIS-matched: per-repeat seed = base + repeat
    positive_label: int = 1                        # ADHD = positive class
    inner_scoring: str = "balanced_accuracy"

  
    canonical_sort: bool = True
    subject_id_regex: Optional[str] = None         

    # Statistics ---------------------------------------------------------- #
    # HIS-matched reporting: headline is the MEAN over the 25 outer folds;
    # bootstrap CI resamples those 25 fold values (NOT the 121 subjects).
    permutation_n: int = 1000
    permutation_scoring: str = "balanced_accuracy"
    bootstrap_n: int = 2000
    ci: float = 0.95

    random_state: int = 42
    checkpoint_dir: str = ".../checkpoints_hismatched"
    n_jobs: int = -1

    def resolved_channels(self) -> List[str]:
        if isinstance(self.channel_selection, (list, tuple)):
            return list(self.channel_selection)
        if self.channel_selection == "all":
            return list(self.canonical_channels)
        if self.channel_selection == "frontal":
            return list(FRONTAL_CHANNELS)
        raise ValueError(f"Unknown channel_selection: {self.channel_selection!r}")


# --------------------------------------------------------------------------- #
# Signal processing
# --------------------------------------------------------------------------- #
def _sos_bandpass(low: float, high: float, fs: float, order: int) -> np.ndarray:
    nyq = fs / 2.0
    high = min(high, 0.99 * nyq)                   # guard against >= Nyquist
    low = max(low, 1e-3)
    return sp_signal.butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")


def _sos_bandstop(low: float, high: float, fs: float, order: int) -> np.ndarray:
    nyq = fs / 2.0
    return sp_signal.butter(order, [low / nyq, high / nyq], btype="bandstop", output="sos")


def preprocess_signal(x: np.ndarray, cfg: Config) -> np.ndarray:
    """Band-pass + notch, zero-phase. x: (n_samples, n_channels)."""
    sos_bp = _sos_bandpass(cfg.bandpass[0], cfg.bandpass[1], cfg.fs, cfg.filter_order)
    sos_bs = _sos_bandstop(cfg.notch[0], cfg.notch[1], cfg.fs, cfg.filter_order)
    x = sp_signal.sosfiltfilt(sos_bp, x, axis=0)
    x = sp_signal.sosfiltfilt(sos_bs, x, axis=0)
    return x


def band_decompose(x: np.ndarray, cfg: Config) -> Dict[str, np.ndarray]:
    """Return {band_name: (n_samples, n_channels)} zero-phase band-limited signals."""
    out = {}
    for name, (lo, hi) in cfg.bands.items():
        sos = _sos_bandpass(lo, hi, cfg.fs, cfg.filter_order)
        out[name] = sp_signal.sosfiltfilt(sos, x, axis=0)
    return out


def segment_indices(n_samples: int, win: int, step: int) -> List[Tuple[int, int]]:
    idx = []
    start = 0
    while start + win <= n_samples:
        idx.append((start, start + win))
        start += step
    return idx


# --------------------------------------------------------------------------- #
# Feature extraction (11 features on a single-channel, single-band segment)
# --------------------------------------------------------------------------- #
_EPS = 1e-12


def _hjorth(x: np.ndarray) -> Tuple[float, float, float]:
    dx = np.diff(x)
    ddx = np.diff(dx)
    var_x = np.var(x)
    var_dx = np.var(dx)
    var_ddx = np.var(ddx)
    activity = var_x
    mobility = np.sqrt(var_dx / (var_x + _EPS))
    mob_dx = np.sqrt(var_ddx / (var_dx + _EPS))
    complexity = mob_dx / (mobility + _EPS)
    return activity, mobility, complexity


def _shannon_entropy(x: np.ndarray, bins: int = 20) -> float:
    counts, _ = np.histogram(x, bins=bins)
    p = counts.astype(float)
    p = p[p > 0]
    p /= p.sum()
    return float(-np.sum(p * np.log(p)))


def _spectral_features(x: np.ndarray, fs: float, band: Tuple[float, float]) -> Tuple[float, float, float]:
    """Return (spectral_entropy, mean_psd, band_power) via one-sided FFT power."""
    n = len(x)
    P = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    total = P.sum()
    if total <= _EPS:
        return 0.0, 0.0, 0.0
    p = P / total
    p_nz = p[p > 0]
    spectral_entropy = float(-np.sum(p_nz * np.log2(p_nz)) / (np.log2(len(P)) + _EPS))
    mean_psd = float(P.mean())
    mask = (freqs >= band[0]) & (freqs <= band[1])
    band_power = float(P[mask].sum())
    return spectral_entropy, mean_psd, band_power


def segment_features(seg: np.ndarray, fs: float, band: Tuple[float, float]) -> List[float]:
    """11 features for one channel, one band, one 2 s window. seg: 1-D array."""
    std = float(np.std(seg, ddof=1))
    rms = float(np.sqrt(np.mean(seg ** 2)))
    skew = float(sp_stats.skew(seg))
    kurt = float(sp_stats.kurtosis(seg))            # Fisher: already subtracts 3
    activity, mobility, complexity = _hjorth(seg)
    shannon = _shannon_entropy(seg)
    spec_ent, mean_psd, band_power = _spectral_features(seg, fs, band)
    return [std, rms, skew, kurt, activity, mobility, complexity,
            shannon, spec_ent, mean_psd, band_power]


def build_feature_names(channels: Sequence[str], bands: Sequence[str]) -> List[str]:
    names = []
    for b in bands:
        for ch in channels:
            for f in FEATURE_NAMES:
                names.append(f"{b}__{ch}__{f}")
    return names


def subject_segment_matrix(raw: np.ndarray, cfg: Config) -> np.ndarray:
    """Full per-subject pipeline -> (n_segments, n_features_per_segment)."""
    x = preprocess_signal(raw, cfg)
    bands = band_decompose(x, cfg)
    win = int(round(cfg.window_sec * cfg.fs))
    step = int(round(win * (1.0 - cfg.overlap)))
    segs = segment_indices(x.shape[0], win, step)
    if not segs:
        return np.empty((0, len(cfg.bands) * x.shape[1] * len(FEATURE_NAMES)))

    n_ch = x.shape[1]
    rows = []
    for (a, b) in segs:
        feat = []
        for band_name, (lo, hi) in cfg.bands.items():
            band_sig = bands[band_name][a:b, :]
            for c in range(n_ch):
                feat.extend(segment_features(band_sig[:, c], cfg.fs, (lo, hi)))
        rows.append(feat)
    return np.asarray(rows, dtype=float)


def pool_subject(seg_mat: np.ndarray, pooling: Sequence[str]) -> np.ndarray:
    """Pool (n_segments, n_feat) -> 1-D subject vector using the requested stats."""
    parts = []
    for stat in pooling:
        if stat == "mean":
            parts.append(np.nanmean(seg_mat, axis=0))
        elif stat == "std":
            parts.append(np.nanstd(seg_mat, axis=0))
        elif stat == "median":
            parts.append(np.nanmedian(seg_mat, axis=0))
        elif stat == "min":
            parts.append(np.nanmin(seg_mat, axis=0))
        elif stat == "max":
            parts.append(np.nanmax(seg_mat, axis=0))
        else:
            raise ValueError(f"Unknown pooling stat: {stat}")
    return np.concatenate(parts)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _load_mat_eeg(path: str, n_expected: int) -> np.ndarray:
    """Load one .mat, return (n_samples, n_channels) with n_channels == n_expected."""
    mat = loadmat(path)
    candidates = [
        v for k, v in mat.items()
        if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 2
    ]
    if not candidates:
        raise ValueError(f"No 2-D numeric array found in {path}")
    # Prefer the array whose one dimension equals the channel count.
    arr = None
    for c in candidates:
        if n_expected in c.shape:
            arr = c
            break
    if arr is None:
        arr = max(candidates, key=lambda a: a.size)
    arr = np.asarray(arr, dtype=float)
    if arr.shape[1] == n_expected:
        return arr
    if arr.shape[0] == n_expected:
        return arr.T
    raise ValueError(f"{path}: shape {arr.shape} does not match {n_expected} channels")


def load_dataset(cfg: Config):
    """Return X_seg_list, y, subject_ids, feature_names, channel indices."""
    all_channels = cfg.canonical_channels
    sel = cfg.resolved_channels()
    missing = [c for c in sel if c not in all_channels]
    if missing:
        raise ValueError(f"Selected channels not in canonical list: {missing}")
    ch_idx = [all_channels.index(c) for c in sel]
    n_all = len(all_channels)

    seg_matrices: List[np.ndarray] = []
    labels: List[int] = []
    subject_ids: List[str] = []

    for label, folder in [(1, cfg.adhd_dir), (0, cfg.control_dir)]:
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Directory not found: {folder}")
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".mat"))
        log(f"[load] {folder}: {len(files)} .mat files (label={label})")
        for fn in files:
            path = os.path.join(folder, fn)
            raw_full = _load_mat_eeg(path, n_all)         # (n_samples, 19)
            raw = raw_full[:, ch_idx]                      # select channels
            seg_mat = subject_segment_matrix(raw, cfg)
            if seg_mat.shape[0] == 0:
                log(f"[load] WARNING: {fn} produced 0 segments (too short); skipped")
                continue
            seg_matrices.append(seg_mat)
            labels.append(label)
            subject_ids.append(os.path.splitext(fn)[0])

    feat_names = build_feature_names(sel, list(cfg.bands.keys()))
    log(f"[load] subjects={len(labels)}  segment-features/seg={len(feat_names)}  "
        f"ADHD={sum(labels)}  Control={len(labels) - sum(labels)}")
    return seg_matrices, np.asarray(labels), subject_ids, feat_names


def make_subject_matrix(seg_matrices, cfg: Config) -> np.ndarray:
    X = np.vstack([pool_subject(m, cfg.pooling) for m in seg_matrices])
    return X


# --------------------------------------------------------------------------- #
# Estimators and hyperparameter grids
# --------------------------------------------------------------------------- #
def build_estimators(cfg: Config) -> Dict[str, Tuple[Pipeline, dict]]:
    knn_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(random_state=cfg.random_state)),
        ("clf", KNeighborsClassifier()),
    ])
    knn_grid = {
        "pca__n_components": [0.85, 0.90, 0.95],
        "clf__n_neighbors": [1, 3, 5, 7, 9, 11],
        "clf__weights": ["uniform", "distance"],
        "clf__metric": ["euclidean", "manhattan"],
    }

    svm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(random_state=cfg.random_state)),
        ("clf", SVC(kernel="rbf", probability=True,
                    class_weight="balanced", random_state=cfg.random_state)),
    ])
    svm_grid = {
        "pca__n_components": [0.85, 0.90, 0.95],
        "clf__C": [0.1, 1, 10, 100, 1000],
        "clf__gamma": ["scale", 1e-2, 1e-3, 1e-4],
    }
    return {"kNN": (knn_pipe, knn_grid), "SVM": (svm_pipe, svm_grid)}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
METRIC_KEYS = [
    "accuracy", "balanced_accuracy", "sensitivity", "specificity",
    "precision", "recall", "f1", "auc_roc", "mcc",
]


def compute_metrics(y_true, y_pred, y_score, pos_label: int = 1) -> Dict[str, float]:
    labels = [1 - pos_label, pos_label]            # [neg, pos]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan   # recall of ADHD
    specificity = tn / (tn + fp) if (tn + fp) else np.nan   # recall of Control
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = np.nan
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision_score(y_true, y_pred, pos_label=pos_label,
                                            zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=pos_label,
                                     zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
        "auc_roc": float(auc),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": cm,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def _get_scores(estimator, X) -> np.ndarray:
    """Positive-class score. Prefer decision_function (consistent with predict
    for SVC); fall back to predict_proba for kNN (also consistent)."""
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(X)).ravel()
    return estimator.predict_proba(X)[:, 1]


# --------------------------------------------------------------------------- #
# Outer folds
# --------------------------------------------------------------------------- #
def his_outer_folds(y, cfg: Config):
    """Yield (repeat, fold, fold_id, train_idx, test_idx) exactly as HIS builds
    them: for each repeat, StratifiedKFold(shuffle=True, seed=base+repeat)."""
    idx = np.arange(len(y))
    for repeat in range(cfg.outer_repeats):
        seed = cfg.outer_cv_base_seed + repeat
        skf = StratifiedKFold(n_splits=cfg.outer_splits, shuffle=True,
                              random_state=seed)
        for fold, (tr, te) in enumerate(skf.split(idx, y)):
            yield repeat, fold, f"r{repeat + 1}f{fold + 1}", tr, te


# --------------------------------------------------------------------------- #
# nested CV
# --------------------------------------------------------------------------- #
def nested_cv_hismatched(name, pipe, grid, X, y, subject_ids, cfg: Config) -> dict:
    n_total = cfg.outer_splits * cfg.outer_repeats
    fold_records: List[dict] = []
    subj = {sid: {"n_test": 0, "n_correct": 0, "preds": [], "true": -1}
            for sid in subject_ids}

    log(f"\n[{name}] HIS-matched nested CV: {cfg.outer_splits}x{cfg.outer_repeats}"
        f" = {n_total} outer folds, inner={cfg.inner_splits}, "
        f"base_seed={cfg.outer_cv_base_seed}")
    t0 = time.time()

    for repeat, fold, fold_id, tr, te in his_outer_folds(y, cfg):
        inner_cv = StratifiedKFold(n_splits=cfg.inner_splits, shuffle=True,
                                   random_state=cfg.outer_cv_base_seed + repeat)
        gs = GridSearchCV(clone(pipe), grid, scoring=cfg.inner_scoring,
                          cv=inner_cv, n_jobs=cfg.n_jobs, refit=True)
        gs.fit(X[tr], y[tr])
        est = gs.best_estimator_
        pred = est.predict(X[te])
        score = _get_scores(est, X[te])

        m = compute_metrics(y[te], pred, score, pos_label=cfg.positive_label)
        m.update({
            "classifier": name, "fold_id": fold_id,
            "repeat": repeat, "fold": fold,
            "best_params": gs.best_params_,
            "inner_best_val": float(gs.best_score_),
            "n_train": int(len(tr)), "n_test": int(len(te)),
        })
        fold_records.append(m)

        for local_i, gi in enumerate(te):
            sid = subject_ids[gi]
            subj[sid]["n_test"] += 1
            subj[sid]["n_correct"] += int(pred[local_i] == y[te][local_i])
            subj[sid]["preds"].append(int(pred[local_i]))
            subj[sid]["true"] = int(y[te][local_i])

        if (len(fold_records)) % cfg.outer_splits == 0:
            log(f"[{name}] repeat {repeat + 1}/{cfg.outer_repeats} done "
                f"({time.time() - t0:.1f}s)")

    # Modal hyperparameter configuration across the 25 folds (for permutation).
    keyed = {}
    for m in fold_records:
        k = json.dumps(m["best_params"], sort_keys=True, default=str)
        keyed[k] = keyed.get(k, 0) + 1
    modal_key = max(keyed.items(), key=lambda kv: kv[1])
    modal_params = json.loads(modal_key[0])

    return {
        "name": name,
        "fold_records": fold_records,
        "subject_tracker": subj,
        "modal_params": modal_params,
        "modal_params_count": modal_key[1],
        "elapsed_sec": time.time() - t0,
    }


# --------------------------------------------------------------------------- #
# Aggregation over the 25 folds + bootstrap CI over fold values
# --------------------------------------------------------------------------- #
def _bootstrap_over_folds(values, n_boot, ci, seed):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    good = values[~np.isnan(values)]
    if good.size == 0:
        return np.nan, np.nan, np.nan
    boots = np.array([rng.choice(good, size=good.size, replace=True).mean()
                      for _ in range(n_boot)])
    alpha = (1.0 - ci) / 2.0
    return (float(good.mean()),
            float(np.percentile(boots, 100 * alpha)),
            float(np.percentile(boots, 100 * (1 - alpha))))


def aggregate_folds(fold_records, cfg: Config) -> dict:
    summary = {}
    for mk in METRIC_KEYS:
        vals = [r[mk] for r in fold_records]
        mean, lo, hi = _bootstrap_over_folds(vals, cfg.bootstrap_n, cfg.ci,
                                             cfg.random_state)
        arr = np.asarray(vals, dtype=float)
        summary[mk] = {
            "mean": mean, "std": float(np.nanstd(arr)),
            "ci_lower": lo, "ci_upper": hi,
            "min": float(np.nanmin(arr)), "max": float(np.nanmax(arr)),
        }
    pooled_cm = sum(r["confusion_matrix"] for r in fold_records)
    observed_bacc = float(np.mean([r["balanced_accuracy"] for r in fold_records]))
    return {"summary": summary, "pooled_cm": pooled_cm,
            "observed_bacc": observed_bacc}


def build_subject_df(subject_tracker, classifier) -> pd.DataFrame:
    rows = []
    for sid, info in subject_tracker.items():
        preds = info["preds"]
        nt = info["n_test"]
        rows.append({
            "classifier": classifier, "subject_id": sid,
            "true_label": info["true"],
            "true_class": "ADHD" if info["true"] == 1 else "Control",
            "n_folds_tested": nt, "n_correct": info["n_correct"],
            "accuracy_rate": round(info["n_correct"] / nt, 4) if nt else 0.0,
            "pred_adhd_rate": round(sum(preds) / len(preds), 4) if preds else 0.0,
            "consistently_correct": nt > 0 and info["n_correct"] == nt,
            "never_correct": nt > 0 and info["n_correct"] == 0,
        })
    return pd.DataFrame(rows).sort_values(
        ["true_class", "accuracy_rate"], ascending=[True, False]
    ).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Permutation test
# --------------------------------------------------------------------------- #
def permutation_test_hismatched(name, pipe, modal_params, X, y, observed_bacc,
                                cfg: Config, checkpoint_path=None) -> dict:
    rng = np.random.default_rng(cfg.random_state)
    perm_scores = np.zeros(cfg.permutation_n, dtype=float)
    start = 0

    if checkpoint_path and os.path.exists(checkpoint_path):
        st = load_pickle(checkpoint_path)
        perm_scores = st["perm_scores"]
        start = st["next_p"]
        rng = st["rng"]
        log(f"[{name}] resuming permutation from {start}/{cfg.permutation_n}")

    fixed = clone(pipe).set_params(**modal_params)
    if fixed.named_steps["clf"].__class__ is SVC:
        fixed.set_params(clf__probability=False)

    log(f"[{name}] permutation test (HIS protocol): n={cfg.permutation_n}, "
        f"fixed params={modal_params}")
    log(f"[{name}] observed mean-fold balanced accuracy = {observed_bacc:.4f}")
    t0 = time.time()

    for p in range(start, cfg.permutation_n):
        yp = rng.permutation(y)
        fold_baccs = []
        for _, _, _, tr, te in his_outer_folds(yp, cfg):
            est = clone(fixed)
            est.fit(X[tr], yp[tr])
            fold_baccs.append(balanced_accuracy_score(yp[te], est.predict(X[te])))
        perm_scores[p] = float(np.mean(fold_baccs))

        if checkpoint_path and (p + 1) % 100 == 0:
            save_pickle({"perm_scores": perm_scores, "next_p": p + 1, "rng": rng},
                        checkpoint_path)
        if (p + 1) % 200 == 0:
            running = float((perm_scores[:p + 1] >= observed_bacc).mean())
            log(f"[{name}]   perm {p + 1}/{cfg.permutation_n}  "
                f"running p={running:.4f}  ({time.time() - t0:.1f}s)")

    p_value = float((perm_scores >= observed_bacc).mean())
    if checkpoint_path and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    return {"observed_score": observed_bacc, "p_value": p_value,
            "null_mean": float(perm_scores.mean()),
            "null_std": float(perm_scores.std()),
            "perm_scores": perm_scores}


# --------------------------------------------------------------------------- #
# Checkpoint helpers
# --------------------------------------------------------------------------- #
def _ckpt_path(cfg: Config, name: str) -> str:
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    return os.path.join(cfg.checkpoint_dir, name)


def save_pickle(obj, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def report_and_export(name, cv_res, agg, perm_res, X, cfg: Config) -> None:
    out_dir = os.path.join(cfg.checkpoint_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    n_outer = cfg.outer_splits * cfg.outer_repeats
    ci_pct = int(round(cfg.ci * 100))

    log("\n" + "=" * 70)
    log(f"RESULTS (HIS-matched)  |  {name}   ({n_outer} outer folds)")
    log("=" * 70)
    log(f"Modal hyperparameters ({cv_res['modal_params_count']}/{n_outer} folds): "
        f"{cv_res['modal_params']}")
    log(f"\nMean over {n_outer} folds  |  bootstrap {ci_pct}% CI over folds "
        f"(n={cfg.bootstrap_n}):")
    for mk in METRIC_KEYS:
        s = agg["summary"][mk]
        log(f"  {mk:20s}: {s['mean']:.4f} +/- {s['std']:.4f}   "
            f"[{s['ci_lower']:.4f}, {s['ci_upper']:.4f}]")

    cm = agg["pooled_cm"]
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    log(f"\nPooled confusion matrix (sum of {n_outer} fold matrices, "
        f"total={cm.sum()}):")
    log(f"                 pred:Control   pred:ADHD")
    log(f"  true:Control      {tn:5d}        {fp:5d}")
    log(f"  true:ADHD         {fn:5d}        {tp:5d}")

    p = perm_res["p_value"]
    p_str = "<0.001" if p < 0.001 else f"{p:.4f}"
    log(f"\nPermutation (HIS protocol, n={cfg.permutation_n}): "
        f"observed={perm_res['observed_score']:.4f}  "
        f"null={perm_res['null_mean']:.4f}+/-{perm_res['null_std']:.4f}  "
        f"p={p_str}")

    # ---- CSV exports --------------- #
    rows = []
    for r in cv_res["fold_records"]:
        bp = r["best_params"]
        rows.append({
            "classifier": name, "fold_id": r["fold_id"],
            "repeat": r["repeat"], "fold": r["fold"],
            "pca_n_components": bp.get("pca__n_components"),
            "clf_n_neighbors": bp.get("clf__n_neighbors"),
            "clf_weights": bp.get("clf__weights"),
            "clf_metric": bp.get("clf__metric"),
            "clf_C": bp.get("clf__C"),
            "clf_gamma": bp.get("clf__gamma"),
            "inner_best_val": r["inner_best_val"],
            "n_train": r["n_train"], "n_test": r["n_test"],
            "tp": r["tp"], "tn": r["tn"], "fp": r["fp"], "fn": r["fn"],
            **{mk: r[mk] for mk in METRIC_KEYS},
        })
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "fold_results.csv"),
                              index=False)

    srows = []
    for mk in METRIC_KEYS:
        s = agg["summary"][mk]
        srows.append({"classifier": name, "metric": mk, **s})
    pd.DataFrame(srows).to_csv(os.path.join(out_dir, "metric_summary.csv"),
                               index=False)

    build_subject_df(cv_res["subject_tracker"], name).to_csv(
        os.path.join(out_dir, "subject_stability.csv"), index=False)

    pd.DataFrame({
        "classifier": name,
        "permutation_index": np.arange(cfg.permutation_n),
        "balanced_accuracy": perm_res["perm_scores"],
    }).to_csv(os.path.join(out_dir, "permutation_scores.csv"), index=False)

    with open(os.path.join(out_dir, "permutation_result.txt"), "w") as f:
        f.write(f"Classifier                 : {name}\n")
        f.write(f"Observed balanced accuracy : {perm_res['observed_score']:.6f}\n")
        f.write(f"n_permutations             : {cfg.permutation_n}\n")
        f.write(f"p-value                    : {perm_res['p_value']:.6f}\n")
        f.write(f"Significant at 0.05        : {perm_res['p_value'] < 0.05}\n")
        f.write(f"Modal params               : {cv_res['modal_params']}\n")
    log(f"\n[{name}] CSVs written to {out_dir}")


# --------------------------------------------------------------------------- #
# Subject-id canonicalisation
# --------------------------------------------------------------------------- #
def canonicalise(X, y, subject_ids, cfg: Config):
    ids = list(subject_ids)
    if cfg.subject_id_regex:
        rgx = re.compile(cfg.subject_id_regex)
        norm = []
        for s in ids:
            mobj = rgx.search(s)
            norm.append(mobj.group(1) if (mobj and mobj.groups()) else
                        (mobj.group(0) if mobj else s))
        ids = norm
    if cfg.canonical_sort:
        order = np.argsort(np.asarray(ids, dtype=object), kind="stable")
        return X[order], y[order], [ids[i] for i in order]
    return X, y, ids


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(cfg: Config) -> dict:
    if cfg.evaluation != "subject":
        raise NotImplementedError("Only evaluation='subject' is implemented.")
    log("Config (HIS-matched baseline):")
    for k, v in asdict(cfg).items():
        log(f"  {k} = {v}")

    feat_ckpt = _ckpt_path(cfg, "features_subject.pkl")
    if os.path.exists(feat_ckpt):
        log(f"\n[features] loading checkpoint {feat_ckpt}")
        blob = load_pickle(feat_ckpt)
        X, y, subject_ids = blob["X"], blob["y"], blob["ids"]
    else:
        seg_matrices, y, subject_ids, _ = load_dataset(cfg)
        X = make_subject_matrix(seg_matrices, cfg)
        save_pickle({"X": X, "y": y, "ids": subject_ids}, feat_ckpt)
        log(f"[features] saved checkpoint {feat_ckpt}")

    X, y, subject_ids = canonicalise(X, np.asarray(y), subject_ids, cfg)
    log(f"\n[features] X shape = {X.shape}  (subjects x pooled-features)")
    log(f"[features] canonical_sort={cfg.canonical_sort}  "
        f"first ids: {subject_ids[:5]}")
    log(f"[features] ADHD={int(y.sum())}  Control={int((y == 0).sum())}")

    estimators = build_estimators(cfg)
    all_results = {}
    for name, (pipe, grid) in estimators.items():
        res_ckpt = _ckpt_path(cfg, f"results_{name}.pkl")
        bundle = None
        if os.path.exists(res_ckpt):
            loaded = load_pickle(res_ckpt)
            if isinstance(loaded, dict) and {"cv", "agg", "perm"} <= set(loaded):
                log(f"\n[{name}] loading results checkpoint {res_ckpt}")
                bundle = loaded
            else:
                log(f"\n[{name}] checkpoint {res_ckpt} has an incompatible schema "
                    f"(keys={list(loaded) if isinstance(loaded, dict) else type(loaded)}) "
                    f"-- ignoring and recomputing.")
        if bundle is None:
            cv_res = nested_cv_hismatched(name, pipe, grid, X, y, subject_ids, cfg)
            agg = aggregate_folds(cv_res["fold_records"], cfg)
            perm_res = permutation_test_hismatched(
                name, pipe, cv_res["modal_params"], X, y,
                agg["observed_bacc"], cfg,
                checkpoint_path=_ckpt_path(cfg, f"perm_{name}.pkl"),
            )
            bundle = {"cv": cv_res, "agg": agg, "perm": perm_res}
            save_pickle(bundle, res_ckpt)
        report_and_export(name, bundle["cv"], bundle["agg"], bundle["perm"], X, cfg)
        all_results[name] = bundle

    summary = {}
    for name, b in all_results.items():
        summary[name] = {
            "fold_mean_ci": {mk: b["agg"]["summary"][mk] for mk in METRIC_KEYS},
            "permutation": {k: v for k, v in b["perm"].items()
                            if k != "perm_scores"},
            "modal_params": b["cv"]["modal_params"],
        }
    with open(_ckpt_path(cfg, "summary_hismatched.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"\n[done] summary -> {_ckpt_path(cfg, 'summary_hismatched.json')}")
    return all_results


if __name__ == "__main__":
    cfg = Config()
    main(cfg)