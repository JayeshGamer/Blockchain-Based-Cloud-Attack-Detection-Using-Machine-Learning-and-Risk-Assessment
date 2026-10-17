# Adaptive ML-DST: Implementation Plan

## Project Title
**Adaptive ML-DST: Online Machine Learning and Uncertainty-Aware Temporal Risk Assessment for Cloud Attack Tolerance**

---

## 1. Project Summary & Base Paper Alignment
- **Base Paper Reference:** Faiza Benmenzer, Rachid Beghdad, and Alaa Eddine Khalfoune, *"Reinforcing Cloud Attack Tolerance on a Large Scale: Advanced Blockchain Technologies With Hierarchical Risk and Smart Contracts,"* *IET Blockchain*, 2025. DOI: 10.1049/blc2.70023.
- **Identified Gap:** The base paper's hierarchical Dempster–Shafer Theory (DST) risk engine operates on static, rule-based attribute mappings and lacks adaptability to non-stationary distributions, concept drift, and temporal risk fluctuations.
- **Novel Contribution:** An online, uncertainty-aware risk engine combining:
  1. Prequential online machine learning (predict-then-update).
  2. Statistical concept-drift detection (ADWIN).
  3. Unsupervised anomaly detection on benign baseline traffic.
  4. Temporal sliding-window flow deviation metrics.
  5. Dempster–Shafer Theory (DST) evidence fusion across ML probability, anomaly score, and temporal deviation, with explicit epistemic uncertainty and conflict mitigation.
  6. Drift-aware dynamic threshold adjustment bounded within safe operational constraints.
  7. 3-tier hierarchical risk aggregation (Node $\to$ Cluster $\to$ Global).
  8. Staged simulated defensive countermeasures.
  9. Explainability via permutation importance and uncertainty decomposition.
  10. Optional blockchain audit ledger with mock fallback.

---

## 2. Research Questions (RQs) and Hypotheses (Hs)
- **RQ1:** Does adaptive online learning improve attack detection under temporally changing network traffic compared with a static classifier?
- **RQ2:** Does Dempster–Shafer fusion of classifier probability, anomaly score, and temporal deviation reduce false positives (FPR) and false negatives (FNR)?
- **RQ3:** Does concept-drift-aware threshold adaptation provide more stable performance than a fixed threshold?
- **RQ4:** What processing latency and memory overhead are introduced by the adaptive system?
- **RQ5:** Which proposed component contributes most to performance according to an ablation study?

- **H1:** The proposed adaptive ML-DST method will achieve higher macro-F1 and lower false-negative rate than a fixed-threshold baseline.
- **H2:** Concept-drift detection will reduce performance degradation when the traffic distribution changes.
- **H3:** DST evidence fusion will provide better-calibrated and more reliable risk scores than raw ML probabilities alone.

---

## 3. Directory Layout
```
adaptive-ml-dst/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env.example
├── configs/
│   ├── default.yaml
│   └── lightweight.yaml
├── data/
│   ├── README.md
│   └── processed/
├── reports/
│   ├── dataset_audit/
│   ├── experiments/
│   ├── figures/
│   └── tables/
├── models/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── drift/
│   ├── dst/
│   ├── risk/
│   ├── simulation/
│   ├── blockchain/
│   ├── api/
│   └── utils/
├── dashboard/
├── contracts/
├── scripts/
├── tests/
├── notebooks/
└── paper/
    ├── ieee_paper.tex
    ├── references.bib
    └── figures/
```

---

## 4. Phase-by-Phase Execution Plan

- **Phase 1: Base Paper Analysis**
  - Read `Base_Paper.pdf`, extract key equations, architecture, results, limitations, and future directions into `BASE_PAPER_ANALYSIS.md`. *(Completed)*

- **Phase 2: Dataset Audit and Sanitization**
  - Implement chunked audit tool `src/data/audit.py`.
  - Strip whitespace, repair Web Attack unicode corruptions, convert $\pm\infty$ to NaN, detect constant columns (bulk rate features), report duplicates, check class distributions and imbalance ratios.
  - Save reports: `reports/dataset_audit/audit_report.json`, `audit_report.csv`, and `audit_report.md`.

- **Phase 3: Core Architecture, Configuration & Preprocessing Pipeline**
  - Create configuration files (`configs/default.yaml`, `configs/lightweight.yaml`).
  - Implement `src/data/loader.py` for chunked streaming and stratified/ordered flow replay.
  - Implement `src/features/preprocessor.py` for zero-leakage scaling, imputation, and constant feature removal.

- **Phase 4: Baseline Models**
  - Implement Dummy Majority, Logistic Regression / SGD, Random Forest / HistGradientBoosting, Isolation Forest, and Static DST baseline in `src/models/baselines.py`.

- **Phase 5: Online Learning, Drift Detection & Anomaly Evidence**
  - Implement incremental online classifier (`src/models/online_classifier.py`) with prequential predict-then-update protocol and probability calibration.
  - Implement ADWIN drift detector (`src/drift/adwin.py`) monitoring online 0/1 loss and traffic variance.
  - Implement benign-trained anomaly detector (`src/models/anomaly_detector.py`) normalized to $[0, 1]$.
  - Implement temporal sliding-window deviation tracker (`src/features/temporal.py`).

- **Phase 6: DST Evidence Fusion, Adaptive Thresholding & Hierarchical Risk**
  - Implement DST engine (`src/dst/evidence_fusion.py`): Frame of discernment $\Omega = \{\text{Benign}, \text{Attack}\}$, mass assignments $m_1, m_2, m_3$, uncertainty $\Theta$, Dempster rule with conflict factor $k$ handling.
  - Implement adaptive thresholding (`src/risk/adaptive_threshold.py`) bounded in $[\theta_{\min}, \theta_{\max}]$.
  - Implement hierarchical risk aggregation (`src/risk/hierarchical_risk.py`) for Node $\to$ Cluster $\to$ Global risk.

- **Phase 7: Simulated Defense Activation & Explainability**
  - Implement staged defense actions (`src/simulation/defense.py`): Log $\to$ Rate Limit $\to$ Filter $\to$ Isolate $\to$ Reroute.
  - Implement explainability engine (`src/models/explainability.py`): Permutation importance and per-alert top features.

- **Phase 8: FastAPI Backend & Interactive Dashboard**
  - Build REST API (`src/api/main.py`) covering all 13 specified endpoints.
  - Build Streamlit dashboard (`dashboard/app.py`) for live stream simulation, risk gauges, drift alerts, and metric visualization.

- **Phase 9: Optional Blockchain Audit Ledger**
  - Write Solidity contract (`contracts/RiskAuditLedger.sol`) recording event hashes and defense actions.
  - Provide Hardhat test suite (`contracts/test/RiskAuditLedger.test.js`).
  - Provide Python Web3 adapter (`src/blockchain/adapter.py`) with local mock ledger fallback.

- **Phase 10: Unit and Integration Test Suite**
  - Comprehensive unit tests in `tests/` covering cleaning, preprocessing, DST combination, conflict handling, adaptive threshold bounds, hierarchical risk aggregation, defense activation, API endpoints, blockchain hashing, and end-to-end stream replay.

- **Phase 11 & 12: Experimental Evaluation & Statistical Analysis**
  - Run Task A (Binary) and Task B (Multiclass / Attack Families).
  - Prequential streaming evaluation on 5 fixed random seeds.
  - Comparative baseline evaluation, answering RQ1–RQ5.
  - Ablation study across 6 configurations (A through F).
  - Robustness analysis (varying attack ratios, noise, drift magnitude, batch sizes).
  - Statistical hypothesis testing (Wilcoxon signed-rank and McNemar's tests with effect sizes).

- **Phase 13: Automated Results, Tables & Figures Pipeline**
  - Generate publication-quality figures (`reports/figures/`) and tables (`reports/tables/`) directly from machine-readable JSON/CSV result files.

- **Phase 14 & 15: IEEE LaTeX Research Paper**
  - Author `paper/ieee_paper.tex` and `paper/references.bib` adhering strictly to IEEE transactions standards.
  - Dynamically populate tables and metrics from experimental result files.

- **Phase 16: Documentation & Reproducibility Package**
  - Complete `README.md`, setup guides for Windows and Linux, reproducibility checklist, viva questions & answers (`VIVA_QA.md`), and presentation outline (`PRESENTATION_OUTLINE.md`).
