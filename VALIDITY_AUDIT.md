# Validity and Rigor Audit: Adaptive ML-DST Architecture

**Project Title:** Adaptive ML-DST: Online Machine Learning and Uncertainty-Aware Temporal Risk Assessment for Cloud Attack Tolerance  
**Base Paper Reference:** Benmenzer et al., *"Reinforcing Cloud Attack Tolerance on a Large Scale: Advanced Blockchain Technologies With Hierarchical Risk and Smart Contracts,"* IET Blockchain, 2025 (DOI: 10.1049/blc2.70023).  
**Audit Date:** September 18, 2026  
**Auditor:** Senior Cybersecurity Researcher & Machine Learning Systems Architect  
**Audit Purpose:** Provide an unsparing, line-by-line inspection of the existing codebase to identify what is correctly implemented, partially implemented, invalid, misleading, unused, and what must be repaired prior to conducting formal experiments.

---

## 1. Executive Audit Summary

The preliminary codebase established initial modular components for online learning, Dempster–Shafer evidence combination, simulated hierarchical risk tiers, and an API/dashboard demo. However, a rigorous methodological audit reveals **critical architectural and experimental defects** that invalidate the preliminary reported metrics (such as the claimed 90.00% ROC-AUC, 2.03% FPR, and 4,940 FPS throughput):

1. **Catastrophic Recall & FNR (The "Low FPR" Illusion):** The preliminary adaptive threshold implementation computed operational cutoffs across a mixture of benign and attack risk scores. In an adversarial batch with high attack prevalence, this pushed the decision threshold up to $\sim 0.85$, causing the proposed system to suffer an unacceptable **75.59% False Negative Rate (24.41% Recall)**, missing more than three-quarters of attacks while presenting an artificially low FPR ($2.03\%$).
2. **Lack of Cryptographic Deduplication & Leakage-Disjoint Partitions:** Train/val/test splits used naive random stratified splitting without SHA-256 row-hash deduplication, allowing identical network flows to span across training and evaluation partitions.
3. **Local Sampling Distortion:** The sampling logic in `src/data/loader.py` evaluated attack rarity inside localized 30,000-row chunks, misclassifying common attacks (like DDoS and DoS Hulk) as rare classes and distorting class distributions.
4. **Synthetic Experimental Visualizations:** `scripts/generate_plots.py` generated streaming risk trajectories and ADWIN drift alarm markers using random Gaussian noise (`np.random.randn`) rather than streaming execution logs.
5. **Conflated Benchmarking & Unisolated Throughput:** Reported processing latency and memory metrics were measured across the entire multi-model benchmark suite rather than isolating the proposed ML-DST engine.
6. **Unused / Heuristic Parameters:** In `src/dst/evidence_fusion.py`, `source_weights` were defined in `__init__` but never used in evidence fusion; classical Shafer reliability discounting was omitted; and conflict was handled via an ad-hoc cutoff ($K=0.9999$) rather than justified conflict reallocation (such as Yager's rule).
7. **Temporal Baseline Contamination:** The sliding window in `src/features/temporal.py` unconditionally ingested attack bursts, allowing volumetric attacks to pollute the benign baseline within 200 flows.

---

## 2. File-by-File Technical Audit

### 2.1 `src/data/loader.py`
- **Status:** **Partially Implemented / Defective**
- **Correctly Implemented:**
  - Standard column whitespace trimming and label normalization mapping.
  - Chunked reading generator `stream_ordered_replay()` reflecting the natural day-to-day progression of CICIDS2017 files.
- **Partially Implemented:**
  - Chronological replay lacks explicit metadata tracking (`source_day`, `source_file`, `original_row_index`) cleanly isolated from feature matrices.
- **Invalid / Misleading:**
  - In `load_stratified_sample()` (lines 64–75), class rarity is computed locally per 30,000-row chunk (`cnt <= rare_threshold`). When a large attack family (such as DDoS or DoS Hulk) spans across chunk boundaries and appears in small counts in an edge chunk, it is erroneously appended to `rare_dfs` and 100% preserved, skewing the overall sample.
  - In `split_data()` (lines 126–141), `train_test_split` is applied directly without global exact deduplication or hash-disjoint isolation.
- **Must Be Repaired:**
  - Implement two distinct pipelines: `src/data/deduplication.py` for global SHA-256 row hashing and exact duplicate extraction, and `src/data/pipeline.py` supporting `configs/lightweight.yaml` and `configs/full.yaml`.
  - Enforce zero row-hash overlap across train, validation, and test partitions.
  - Determine class rarity using global dataset counts prior to sampling.

### 2.2 `src/features/preprocessor.py`
- **Status:** **Partially Implemented**
- **Correctly Implemented:**
  - Removal of zero-variance constant columns.
  - Removal of duplicate column `Fwd Header Length.1`.
  - Median computation strictly on training data with fallback to 0.0.
  - Outlier clipping to $[-50.0, 50.0]$ to protect online learners from numerical overflow.
- **Partially Implemented:**
  - Batch `RobustScaler` is fitted on training split, but no incremental scaler or streaming normalization mechanism exists for streaming flows where feature distributions drift over time.
- **Invalid / Misleading:**
  - If metadata columns (`source_file`, `source_day`, `original_row_index`) or raw `Label` variants are present in the incoming DataFrame, `preprocessor.transform()` does not systematically check for or strip them, risking feature contamination.
- **Must Be Repaired:**
  - Explicitly prune all non-numeric metadata and label columns in `extract_features_and_labels()`.
  - Provide an incremental scaling interface for prequential evaluation without future data leakage.

### 2.3 `src/features/temporal.py`
- **Status:** **Partially Implemented / Vulnerable to Contamination**
- **Correctly Implemented:**
  - Avoids artificial timestamps; relies strictly on ordered sequence indices.
  - Uses hyperbolic tangent mapping $\tanh(Z/3.0)$ to map weighted composite Z-scores to $[0, 1]$.
- **Partially Implemented:**
  - Tracks only 4 metrics: byte rate, packet rate, duration, and flow IAT mean.
- **Invalid / Defective:**
  - **Buffer Contamination:** In `compute_deviation()` (lines 95–96), `buf.append(val)` is executed unconditionally for every flow. During a sustained volumetric attack (e.g., DoS Hulk or PortScan), the 200-flow buffer fills with attack traffic in seconds, causing rolling means and standard deviations to adapt to the attack. Subsequent attack flows exhibit $Z \approx 0$ and deviation collapses to 0.
- **Must Be Repaired:**
  - Implement a protected benign baseline: update rolling statistics only from high-confidence benign flows.
  - Expand deviation metrics to include: Flow Bytes/s, Flow Packets/s, Flow Duration, Flow IAT Mean, Forward/Backward packet ratio, and Rolling Attack-Score deviation.
  - Ensure scores are computed strictly before window insertion.

### 2.4 `src/models/online_classifier.py`
- **Status:** **Partially Implemented / Uncalibrated**
- **Correctly Implemented:**
  - Incremental linear SGD using `partial_fit` with `log_loss` and fixed classes `[0, 1]`.
- **Partially Implemented:**
  - Warm start functionality is active, but lacks a non-linear online classifier alternative.
- **Invalid / Misleading:**
  - Lines 72–77 compute raw sigmoid scores from the decision function and designate them as "calibrated probabilities." Uncalibrated SGD scores are biased under class imbalance and non-stationary streams.
  - `adapt_to_drift()` (lines 84–91) modifies `self.model.alpha = max(1e-6, self.alpha / boost_factor)`. In scikit-learn, modifying `alpha` after initialization does not update the internal `optimal` learning rate schedule ($\eta^{(t)} = \frac{1}{\alpha(t_0 + t)}$).
- **Must Be Repaired:**
  - Integrate River `HoeffdingAdaptiveTreeClassifier` (or `AdaptiveRandomForestClassifier`) as the non-linear incremental learner alongside linear SGD.
  - Implement valid probability calibration on validation data or past stream windows; otherwise explicitly label the output as "attack scores."
  - Implement active drift responses: plasticity boost, background learner promotion, partial/full model reset, and ensemble replacement.

### 2.5 `src/models/anomaly_detector.py`
- **Status:** **Correctly Implemented (Baseline), Incomplete Reporting**
- **Correctly Implemented:**
  - `IsolationForest` is trained strictly on benign baseline data (`X_tr[y_tr == 0]`).
  - Score normalization anchors (`min_score_`, `max_score_`) are derived strictly from benign training percentiles.
- **Partially Implemented:**
  - Anomaly detector is evaluated only as a component of the fusion pipeline, not independently.
- **Must Be Repaired:**
  - Implement independent evaluation reporting AUROC, PR-AUC, Recall, and FPR on clean validation holdouts.
  - Support an online anomaly scoring mode (e.g., streaming half-space trees or sliding-window density).

### 2.6 `src/models/baselines.py`
- **Status:** **Partially Implemented / Inaccurate Base Paper Baseline**
- **Correctly Implemented:**
  - `DummyMajorityBaseline`, `LogisticRegressionBaseline`, `RandomForestBaseline`, `IsolationForestBaseline`.
- **Invalid / Misleading:**
  - `StaticDSTBaseline` (lines 133–197): The base paper (*Benmenzer et al., 2025*) uses a three-tier hierarchical architecture with multi-attribute trust factors and smart-contract consensus. The implementation in `baselines.py` splits the 78 features into two arbitrary halves and averages them into two heuristic indicators. This is an oversimplified heuristic and cannot be claimed as a faithful reproduction of the base paper.
- **Must Be Repaired:**
  - Implement `StaticSGD` and `HistGradientBoosting` baselines.
  - Accurately describe `StaticDSTBaseline` as a non-learning heuristic DST baseline with documented limitations.

### 2.7 `src/drift/adwin.py`
- **Status:** **Partially Implemented / Suboptimal Complexity**
- **Correctly Implemented:**
  - Implements the Hoeffding bound with Bonferroni correction: $\epsilon = \sqrt{\frac{1}{2m} \ln\frac{4}{\delta'}}$.
  - Drops older subwindow $W_0$ when a statistically significant shift occurs.
- **Partially Implemented:**
  - Only prediction error is monitored in the preliminary experiment; stream statistics and attack-score distributions are not monitored.
- **Inefficiencies / Defects:**
  - Lines 75–80 recompute `sum(self.window[:i])` on Python lists inside the candidate split loop, causing $O(n^2)$ complexity per update rather than $O(n)$ with cumulative sums.
- **Must Be Repaired:**
  - Provide a vectorized $O(n)$ implementation and compare against `river.drift.ADWIN`.
  - Add unit tests verifying behavior on stationary streams (zero false alarms), sudden step drift, and gradual drift.
  - Monitor multiple orthogonal signals with separate ADWIN instances: prediction error, attack score, and volumetric packet rates.

### 2.8 `src/dst/evidence_fusion.py`
- **Status:** **Partially Implemented / Unused Weights / Defective Conflict Handling**
- **Correctly Implemented:**
  - Frame of discernment $\Omega = \{\text{Benign}, \text{Attack}\}$.
  - Calculation of Belief $\text{Bel}(\text{Attack})$, Plausibility $\text{Pl}(\text{Attack})$, Uncertainty $m(\Theta)$, and Pignistic expectation $\text{BetP}(\text{Attack})$.
- **Unused Code:**
  - `source_weights` parameter passed into `__init__` is completely ignored in `fuse()`. Hardcoded budget factors (`0.8`, `1.2`, `1.0`) are used instead.
- **Invalid / Defective:**
  - **Absence of Classical Shafer Discounting:** Classical reliability discounting multiplies specific hypothesis masses by source reliability $r_i \in [0, 1]$ and reallocates the remainder to $\Theta$:
    $$m^{\alpha}(A) = r_i \cdot m(A), \quad m^{\alpha}(B) = r_i \cdot m(B), \quad m^{\alpha}(\Theta) = 1 - r_i + r_i \cdot m(\Theta)$$
  - **High-Conflict Clamping:** In `combine_two_bpas()` (lines 65–68), when $k \ge 0.9999$, the code sets $k_{\text{eff}} = 0.9999$ and divides by $1.0 - k_{\text{eff}} = 0.0001$. This forces standard Dempster combination near the singularity rather than properly applying Yager's rule (where conflicting mass is assigned to $\Theta$).
- **Must Be Repaired:**
  - Implement rigorous Shafer reliability discounting with reliabilities $r_{\text{ML}}, r_{\text{anom}}, r_{\text{temp}}$ calibrated on validation data.
  - Implement Yager's combination rule as an alternative when conflict $k > k_{\text{threshold}}$.
  - Add mathematical invariant tests: commutativity, associativity, mass conservation ($\sum m = 1$), and NaN rejection.

### 2.9 `src/risk/adaptive_threshold.py`
- **Status:** **CRITICAL DEFECT — ROOT CAUSE OF 75.59% FNR**
- **Defect Mechanism:**
  - In `update()` (lines 51–58), all incoming risk scores are appended into `self.risk_buffer`.
  - The threshold target is computed as `np.percentile(self.risk_buffer, (1.0 - target_fpr) * 100)`.
  - When the incoming batch contains attack traffic (e.g. 30% attack flows with scores near 0.8–0.9), the 99th percentile ($1 - 0.01$) is located at the top 1% of the **attack distribution**!
  - This pushes the threshold up to $\sim 0.85$. Any attack flow with a risk score of $0.75$ is classified as **BENIGN**.
  - **Consequence:** Attack recall dropped to **24.41%**, resulting in a **75.59% False Negative Rate**.
- **Must Be Repaired:**
  - Separate benign risk tracking from attack scores: update the benign background distribution only from high-confidence benign samples.
  - Implement two operational modes:
    - **Mode 1 (Recall-Constrained):** Target minimum validation recall of $\ge 90\%$, minimizing FPR subject to that constraint.
    - **Mode 2 ($F_2$-score Optimization):** Penalize missed attacks (false negatives) twice as heavily as false alarms.
  - Freeze threshold updates during active attack bursts and immediately following detected concept drift.

### 2.10 `src/risk/hierarchical_risk.py`
- **Status:** **Partially Implemented / Incomplete Aggregation**
- **Correctly Implemented:**
  - Simulated three-tier topology: Tier 1 (Nodes), Tier 2 (Clusters), Tier 3 (Global).
  - Deterministic load mapping via Knuth's multiplicative hashing on flow indices.
- **Partially Implemented:**
  - Only risk score is aggregated across the hierarchy; evidential belief, plausibility, and epistemic uncertainty are calculated at Tier 1 but dropped at Tiers 2 and 3.
  - Fixed heuristic aggregation: $0.6 \cdot \max + 0.4 \cdot \text{mean}$.
- **Must Be Repaired:**
  - Propagate uncertainty, belief, and plausibility through all tiers.
  - Implement and compare multiple aggregation operators: Mean, Maximum, Percentile (90th), and evidential DST combination.

### 2.11 `src/simulation/defense.py`
- **Status:** **Partially Implemented / Missing Specified Countermeasures**
- **Correctly Implemented:**
  - Safe simulation mode (does not touch host network or OS firewall).
  - Staged action evaluation based on risk tiers.
- **Partially Implemented:**
  - Implements only 4 actions (`LOG_MONITOR`, `RATE_LIMIT`, `TRAFFIC_FILTER_ISOLATE`, `GLOBAL_REROUTE_ALERT`), whereas Phase 9 specifies 7 distinct actions:
    1. Log
    2. Increased monitoring
    3. Rate limiting
    4. Traffic filtering
    5. Temporary node isolation
    6. Cluster traffic rerouting
    7. Global alert
- **Missing Telemetry:**
  - Defense event records lack detailed evidence attribution: `source_day`, `DST belief`, `plausibility`, `uncertainty`, `conflict`, and `model explanation`.
- **Must Be Repaired:**
  - Implement all 7 defensive actions with complete telemetry records.
  - Compute defense quality metrics: True Interventions, Benign Interruptions, Missed Attacks, and Response Delay.

### 2.12 `scripts/run_experiments.py`
- **Status:** **Methodologically Invalid / Conflated Protocols**
- **Invalid / Defective:**
  - **No Protocol Separation:** Conflates conventional static evaluation and streaming evaluation into a single random split loop.
  - **Throughput & Memory Conflation (lines 580–586):**
    ```python
    average_throughput_fps = round(len(df_sample) * len(seeds) / total_wall_time, 1)
    memory_overhead_mb = mem_after_mb - mem_before_mb
    ```
    `total_wall_time` and `mem_after_mb` encompass the execution of the entire experiment script (Random Forest training, Logistic Regression, etc.). This artificially distorts the throughput and memory claims of the proposed model.
  - **Underpowered Statistical Testing:** Running Wilcoxon signed-rank tests across $N=5$ random seeds on identical subsamples is statistically underpowered and yields uninterpretable $p$-values.
  - **No Persisted Prediction Logs:** Raw flow predictions, ground truth arrays, and risk trajectories were not serialized, preventing external verification of McNemar tests and calibration curves.
- **Must Be Repaired:**
  - Separate into Protocol A (Duplicate-Safe Conventional), Protocol B (Genuine Chronological Prequential), and Protocol C (Controlled Drift Benchmark).
  - Profile proposed model latency and peak memory independently using `psutil` or `tracemalloc`.
  - Persist predictions and ground truth in compressed format (`.parquet` or `.json.gz`).

### 2.13 `scripts/run_multiclass.py`
- **Status:** **Partially Implemented / Data Leakage in Encoder**
- **Correctly Implemented:**
  - Evaluation of high-level attack families (DoS, DDoS, PortScan, Brute Force, Web Attack, Botnet, Infiltration, Exploit).
- **Invalid / Leakage:**
  - Lines 49–51 fit `LabelEncoder` on `pd.concat([y_tr_fam, y_te_fam])`. Fitting encoders on test data represents an avoidable leakage risk.
  - Rare classes with negligible test support (e.g. Infiltration with $\le 10$ test flows) were reported alongside high-support classes without marking them as having insufficient support for statistical significance.
- **Must Be Repaired:**
  - Fit label encoding strictly on training data or use a static ontology dictionary.
  - Explicitly flag low-support classes ($N < 50$) as statistically uninformative.

### 2.14 `scripts/generate_plots.py`
- **Status:** **INVALID / SYNTHETIC EXPERIMENTAL ARTIFACTS**
- **Critical Violation:**
  - Lines 123–145 generate Figure 4 (`fig4_streaming_risk_threshold.png`) using pseudo-random Gaussian numbers (`np.random.randn`) and hardcoded drift indices (`drift_points = [82, 203]`) rather than reading saved streaming logs.
- **Must Be Repaired:**
  - Delete all synthetic plotting functions.
  - Rewrite plot generator to consume only validated, serialized experimental logs from `reports/experiments/`.

### 2.15 API (`src/api/main.py`) and Dashboard (`dashboard/app.py`)
- **Status:** **Functional Demo / Missing Endpoints & Controls**
- **Missing Endpoints:**
  - Missing `GET /drift/events`
  - Missing `GET /threshold/history`
  - Missing `GET /experiments` (listing saved experiment IDs)
- **Dashboard Deficiencies:**
  - Lacks dataset mode selection (`lightweight` vs `full`).
  - Lacks evaluation protocol selector (`Protocol A`, `B`, `C`).
  - Lacks model and drift-response strategy selector.
  - Does not highlight Recall and FNR or warn when recall drops below 90%.
- **Must Be Repaired:**
  - Implement missing REST endpoints in FastAPI backend.
  - Upgrade Streamlit dashboard to include protocol selection, drift-strategy controls, prominent recall/FNR alarms, and full result export.

### 2.16 Test Suite (`tests/`)
- **Status:** **23 Tests Passing, but Significant Coverage Gaps**
- **Passing:** Basic unit tests for preprocessor, threshold manager, DST, hierarchy, defense, blockchain, and API.
- **Missing Critical Coverage:**
  - No test verifying global SHA-256 deduplication.
  - No test verifying hash-disjoint partition isolation.
  - No test verifying absence of label leakage into feature matrices.
  - No test verifying absence of metadata columns in features.
  - No test comparing custom ADWIN against `river.drift.ADWIN`.
  - No test verifying ADWIN behavior on stationary vs drifting streams.
  - No test verifying DST associativity, commutativity, and high-conflict Yager handling.
  - No test verifying recall-constrained threshold optimization.
  - No test verifying predictable chronological replay ordering.

---

## 3. Unsupported Claims & Irreproducible Results in Preliminary Draft

| Preliminary Claim | Audited Reality | Remediation Required |
| :--- | :--- | :--- |
| **"ROC-AUC of $0.9000 \pm 0.0041$"** | Computed on a 63,130-row random split with potential cross-partition duplicate leakage. | Recompute on hash-disjoint Protocol A and chronological Protocol B. |
| **"FPR of $2.03\% \pm 0.0011\%$"** | Misleading artifact of an inflated threshold ($\approx 0.85$) that missed **75.59% of attacks** (Recall: 24.41%). | Re-calibrate threshold using Recall-constrained ($\ge 90\%$) or $F_2$ optimization. Report Recall & FNR prominently. |
| **"Throughput exceeding 4,940 FPS"** | Measured by dividing total dataset size by entire script execution time across all baseline models. | Measure isolated ML-DST latency ($50\text{th}, 95\text{th}, 99\text{th}$ percentiles) and throughput independently. |
| **"Memory overhead of 21.89 MB"** | Conflated across process lifecycle; did not isolate model memory footprint. | Profile peak resident memory using continuous process sampling / `tracemalloc`. |
| **"Concept Drift Trajectory" (Fig. 4)** | Generated via synthetic random numbers (`np.random.randn`) and fake drift points. | Generate plots strictly from persisted experimental streaming event logs. |
| **"Base Paper DST Baseline"** | Arbitrarily averaged half of feature columns; did not implement the hierarchical trust model of Benmenzer et al. | Accurately qualify the baseline and document its structural assumptions. |

---

## 4. Phase-by-Phase Remediation Roadmap

1. **Dataset Pipeline (Phase 2):** Build `src/data/deduplication.py` and `src/data/pipeline.py`. Perform global SHA-256 deduplication across the 2.83M flows. Create `configs/lightweight.yaml` and `configs/full.yaml`. Produce split integrity reports.
2. **Evaluation Protocols (Phase 3):** Implement Protocol A (Conventional Hash-Disjoint), Protocol B (Chronological Prequential Replay with 1-batch delayed feedback), and Protocol C (Controlled Drift Benchmark).
3. **Model & Evidential Engineering (Phases 4–7):**
   - Integrate `river.tree.HoeffdingAdaptiveTreeClassifier` and calibrated online SGD.
   - Vectorize ADWIN and verify against River ADWIN.
   - Fix temporal baseline buffer contamination with a protected benign baseline.
   - Implement Classical Shafer Reliability Discounting and Yager conflict reallocation in DST.
   - Implement Recall-Constrained ($\ge 90\%$) and $F_2$-optimized adaptive thresholds.
4. **Hierarchy & Defense Simulation (Phases 8–10):**
   - Propagate uncertainty through all three tiers.
   - Implement all 7 simulated defense actions with complete contextual telemetry.
5. **System Verification & Real Evaluation (Phases 11–20):**
   - Add comprehensive tests in `tests/` and generate `reports/verification/test_results.txt`.
   - Run lightweight and full experiments, serializing all prediction logs.
   - Generate genuine, traceable visualizations from saved logs.
   - Author final system documentation (`IMPLEMENTATION_RESULTS.md`, `LIMITATIONS.md`, `REPRODUCIBILITY.md`).
