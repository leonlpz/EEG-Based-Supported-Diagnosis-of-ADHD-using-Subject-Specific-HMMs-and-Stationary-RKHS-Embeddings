# HMM-RKHS Classification Project

**Interpretable EEG-Based Supported Diagnosis of ADHD Using Subject-Specific HMMs and Stationary RKHS Embeddings.**

This repository contains the full pipeline for classifying ADHD versus control children from resting/task EEG. Each subject's frontal EEG is modeled with a dedicated Hidden Markov Model with Gaussian Mixture Model emissions (HMM-GMM). The fitted generative models are then embedded in a Reproducing Kernel Hilbert Space (RKHS), where inter-subject similarity is computed in closed form. These kernels feed K-Nearest Neighbors (KNN) and Support Vector Machine (SVM) classifiers, evaluated on both real and synthetic EEG under several validation protocols of increasing statistical rigor.

---

## Motivation

EEG-based ADHD classification is difficult because of strong inter-subject variability: a single global feature space rarely captures the dynamics shared across individuals. Rather than extracting hand-crafted features, this project represents each subject by their *own* temporal generative model (an HMM-GMM) and measures distances between subjects directly in the space of distributions. This yields interpretable state dynamics while remaining robust to subject heterogeneity.

---

## Method overview

The pipeline proceeds in four stages:

1. **Preprocessing** — bandpass and notch filtering, z-score normalization, and frontal-channel selection on the 10-20 montage.
2. **Generative modeling** — a subject-specific HMM-GMM is trained per subject via the Baum-Welch algorithm, across three topologies: `N3G3`, `N4G4`, and `N5G5` (number of hidden states x GMM components per state).
3. **Kernel embedding** — two kernel families compare subjects in RKHS:
   - **HIS-RKHS (MMD²)** — closed-form Gaussian-Gaussian RBF kernels yielding a Maximum Mean Discrepancy between embedded model distributions.
   - **PPK** — the Probability Product Kernel between subject HMMs.
4. **Classification** — KNN and SVM with precomputed kernel matrices.

---

## Evaluation protocols

The repository implements three evaluation regimes, ordered by statistical rigor. This ordering is deliberate: comparing them quantifies how much reported performance depends on the chosen protocol, a point of methodological heterogeneity in the ADHD EEG literature.

| Protocol | Description | Outputs |
|----------|-------------|---------|
| **Single hold-out** | One subject-level 80/20 split. | Optimistic, high-variance point estimates. |
| **Repeated K-fold CV** | 7 folds x 5 repeats (35 partitions), non-nested. | Mean ± std, fold-level min/max. |
| **Nested CV** *(primary)* | 5 outer folds x 5 repeats (25 partitions) with an inner loop for hyperparameter search only. | Bootstrap 95% CIs, pooled confusion matrices, label-permutation significance tests (1000 permutations). |

The nested protocol is the headline result, since hyperparameters are selected strictly within each outer training fold, preventing leakage into model selection.

---

## Key results

Under the nested cross-validation protocol, the strongest configurations reach a balanced accuracy of approximately **73.5%** (95% CI [69.8, 77.0]), with label-permutation tests confirming significance (*p* < 0.001). Single hold-out estimates run substantially higher, and the spread across protocols (roughly 22 percentage points) is itself reported as evidence of evaluation-rigor sensitivity. Detailed per-topology, per-kernel, and per-classifier tables are in `experiment2_section.pdf` (repeated K-fold CV) and `experiment3_section.pdf` (nested CV).

---

## Repository structure

```
HMM_RKHS_Classification_Project/
|
|-- Preprocessing/                       # EEG preprocessing utilities
|-- preprocessed/                        # Cached, model-ready preprocessed EEG
|-- preprocessed_sensitivity/            # Preprocessing variants for sensitivity analysis
|   |-- ADHD_prep/
|   |-- Control_prep/
|
|-- training-hmm.ipynb                   # Subject-level HMM-GMM training (Baum-Welch)
|-- Synthetic_EEG_data_generation.ipynb  # Synthetic EEG generation from fitted HMMs
|-- Synthetic EEG.rar                    # Generated synthetic EEG archive
|
|-- hmm_results/                         # Per-topology HMM outputs and summaries
|   |-- N3G3/  N4G4/  N5G5/              # Results per HMM topology
|   |-- plots/
|   |-- best_config_per_subject.csv
|   |-- combined_summary.csv
|
|-- single held out test/                # Single 80/20 hold-out evaluation
|   |-- rkhs-knn-svm-eeg-dataset.ipynb
|   |-- ppk_knn_svm_eeg_dataset.ipynb
|   |-- EEG_real_and_synthetic_results/
|
|-- repeated k-fold CV/                   # Repeated stratified K-fold evaluation
|   |-- repeatedcv_HIS_hmm_knn_svm.ipynb
|   |-- repeatedcv_PPK_knn_svm_Dataset.ipynb
|
|-- nested CV/                            # Nested cross-validation (primary protocol)
|   |-- rkhs_classifiers/
|   |-- ppk_classifiers/
|   |-- rkhs_vs_ppk_comparison/
|   |-- fold_analysis/
|   |-- subject_hmm_quality.csv
|   |-- subject_hard_vs_unlucky.csv
|
|-- subject_lengths_full.csv             # Per-subject recording lengths
|-- experiment2_section.pdf              # Repeated K-fold CV results (manuscript section)
|-- experiment3_section.pdf              # Nested CV results (manuscript section)
|-- README.md
```

---

## Data

Real EEG recordings from 121 children (ADHD and control groups) from the public IEEE DataPort dataset:

> https://ieee-dataport.org/open-access/eeg-data-adhd-control-children

Synthetic EEG, generated by sampling from the fitted subject HMMs, is included for augmentation and sanity-checking (`Synthetic EEG.rar`, `Synthetic_EEG_data_generation.ipynb`). The raw dataset is not redistributed here; download it from the link above and place it where the preprocessing notebooks expect it.

---

## Requirements

The notebooks were developed for Kaggle and Google Colab environments. Core dependencies:

- Python 3.10+
- `numpy`, `scipy`, `pandas`
- `scikit-learn`
- `matplotlib`
- `optuna` (hyperparameter search; nested CV additionally uses exhaustive grid search)

The HMM-GMM training, Baum-Welch routine, RKHS/MMD² computation, and PPK kernel are implemented within the notebooks.

---

## Reproducing the results

1. Download the EEG dataset and run the **Preprocessing** stage to populate `preprocessed/`.
2. Train subject-level models with `training-hmm.ipynb`; outputs land in `hmm_results/` for each topology.
3. Choose an evaluation protocol:
   - Single hold-out: notebooks under `single held out test/`
   - Repeated K-fold CV: notebooks under `repeated k-fold CV/`
   - Nested CV (recommended): notebooks under `nested CV/`
4. Compare kernels and inspect fold-level behavior via the outputs in `nested CV/rkhs_vs_ppk_comparison/` and `nested CV/fold_analysis/`.

---

## Citation

If you use this code, please cite the associated manuscript (in revision). A formal citation entry will be added here upon publication. https://www.preprints.org/manuscript/202605.1367

---

## Authors and affiliation

Developed within the **ACEMATE** research program, Maestria en Ingenieria Electrica, Universidad Tecnologica de Pereira (UTP), Colombia.

- Leonardo Lopez-Ortiz
- Cristhian K. Valencia-Marin
- Julian Gil-Gonzalez
- Paula M. Herrera-Gomez
- David Cardenas-Pena

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.