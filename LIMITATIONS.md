# Technical Scope, Implementation Boundaries, and Limitations
## Adaptive ML-DST: Online Machine Learning and Uncertainty-Aware Temporal Risk Assessment for Cloud Attack Tolerance

To maintain uncompromising academic and technical integrity, this document explicitly details the **technical boundaries**, **simulation assumptions**, **empirical negative results**, and **scientific limitations** of this implementation.

---

### 1. Empirical Limitations and Negative Results

In strict adherence to scientific rigor, we report all observed limitations and negative findings honestly:

1. **Random Forest Superiority on Conventional Splits (Protocol A):**
   - On conventional stratified random partitions with no temporal drift, static tree ensembles (**Random Forest:** 99.49% accuracy, 98.42% recall, 0.08% FPR; **HistGradientBoosting:** 99.61% accuracy, 98.92% recall) decisively outperform the proposed streaming framework (**Adaptive ML-DST:** 65.34% accuracy, 94.63% recall, 47.25% FPR).
   - *Scientific Rationale:* Adaptive ML-DST is engineered specifically for temporal non-stationarity and uncertainty-aware adaptation, not for static curve-fitting on i.i.d. random splits.

2. **Online SGD vs. Adaptive ML-DST Empirical Trade-off (Protocol B):**
   - In prequential chronological streaming, unconstrained linear `Online SGD (No Drift)` achieves higher unconstrained recall (**86.91%** vs. **85.39%**) and slightly higher Macro-F1 (**0.9294** vs. **0.9218**) than the multi-source `Adaptive ML-DST (Proposed)`.
   - *Honest Reporting Rationale:* We do NOT claim that Adaptive ML-DST outperforms Online SGD on raw binary recall or Macro-F1. Adaptive ML-DST's scientific advantage lies in:
     (a) Higher Precision-Recall AUC (**0.9419** vs. **0.9234**),
     (b) Controlled false positive rate (**7.79%** vs. **9.03%**),
     (c) Epistemic uncertainty quantification through Shafer belief masses ($m(\Theta)$), and
     (d) Robustness against non-classifier sensor failures via multi-sensor Dempster-Shafer evidential fusion.

3. **Cold-Start Vulnerability under Zero-Shot Emergence (Protocol B):**
   - When novel attack families first appear in the chronological stream under a strict 1-batch label delay ($t+1$), the online learner suffers from cold-start delay:
     - **Tuesday (FTP/SSH Patator):** Achieves **26.75% recall** ($FNR = 73.25\%$).
     - **Thursday Afternoon (Infiltration):** Achieves **30.56% recall** ($FNR = 69.44\%$).
   - *Scientific Rationale:* Although the benign anomaly detector and temporal tracker elevate risk scores, the linear incremental classifier requires several batches of delayed feedback before updating its decision hyperplane to reflect the novel attack vectors.

4. **Frozen SGD Baseline Thresholding Invariant:**
   - When SGD is trained exclusively on benign Monday data, decision function values are strongly negative ($\sim 10^{-8}$). Applying an Isolation Forest anomaly percentile to classifier probabilities is mathematically invalid and yields 100% false negatives. A classifier-specific validation percentile (e.g. 99th percentile of Monday benign scores) must be used.

5. **Simplified Static Heuristic-DST Distinction:**
   - In the base paper (Benmenzer et al., 2025), mass functions and reliabilities are defined theoretically. Our baseline is a simplified heuristic implementation inspired by the paper's framework, calibrated via validation grid search. We explicitly name it: `Simplified Static Heuristic-DST (paper-inspired; not an exact reproduction)`.

6. **Controlled Concept-Drift Detection Delay and Matching (Protocol C):**
   - In the controlled benchmark (Protocol C), out of 5 true change points, River ADWIN detected 3 change points with a mean detection delay of **50.0 flows**, missed 2 subtle gradual drifts, and generated 2 duplicate alarms and 0 false alarms under strict one-to-one partition matching.

---

### 2. What is Implemented in Production Code

1. **Global SHA-256 Deduplication:** Processed all 2,830,743 raw CICIDS2017 records, eliminating 309,079 duplicates (10.92%) and proving zero hash overlap across splits.
2. **Prequential Online Learning Pipeline:** Strict predict-before-update streaming with configurable delayed label feedback ($t+1$).
3. **Multi-Signal Concept-Drift Architecture:** River ADWIN monitoring classification loss, risk scores, and volumetric rates with automated plasticity boosts ($2.0\times$ learning rate for 15 batches).
4. **Benign-Trained Anomaly Detector:** Unsupervised Isolation Forest trained exclusively on clean Monday benign traffic with validation-calibrated scaling.
5. **Temporal Deviation Tracker:** Protected benign baselines tracking 6 temporal network signals (flow bytes/s, packets/s, duration, IAT mean, packet ratio, rolling risk).
6. **Mathematically Rigorous DST Fusion:** Classical Shafer reliability discounting ($m^\alpha(A) = r_i \cdot m(A)$, $m^\alpha(\Theta) = 1 - r_i + r_i \cdot m(\Theta)$) and Yager conflict reallocation ($K \to \Theta$).
7. **Validation-Derived Adaptive Thresholding:** Constraint-satisfying optimization (Recall $\ge 90\%$) with attack-burst freezing preventing baseline contamination.
8. **FastAPI Backend & Streamlit Dashboard:** Complete REST API and live command center displaying genuine telemetry without synthetic curves.

---

### 3. What is Simulated

1. **Cloud Compute Topology:** Virtual cloud nodes ($v \in V$) and clusters ($C_i \in \mathcal{C}$) are simulated software state objects within Python memory rather than physical VMs or Kubernetes pods across AWS/GCP/OpenStack.
2. **Defensive Countermeasures:** Rate limiting, IP filtering, node isolation, and cluster traffic rerouting are **software simulations**. They do **not** modify host OS iptables, pfSense, or physical routing tables.
3. **Network Stream Replay:** Ingress traffic is generated via ordered chronological replay from CICIDS2017 rather than live packet taps on a 10Gbps physical NIC.

---

### 4. What is Simplified (Distinction from Base Paper)

1. **Blockchain Infrastructure:**
   - *Base Paper:* Configured a multi-organization Hyperledger Fabric network with Docker peer nodes, orderer nodes, CA containers, and a custom Proof of Risk (PoR) consensus mechanism.
   - *Our Implementation:* Implements a privacy-preserving Solidity audit contract (`RiskAuditLedger.sol`) and an in-memory cryptographic hash ledger fallback. We do **not** reproduce the full Fabric PoR multi-peer consensus.
2. **No Claim of Blockchain Improving Accuracy:**
   - We explicitly clarify that blockchain provides **tamper-evident audit logging** only. It does **not** improve classification accuracy, recall, or latency.

---

### 5. Academic Integrity Disclaimers

1. **No Zero-Day Panacea:** We do not claim that unsupervised anomaly detection or online learning guarantees complete protection against zero-day exploits. Attacks that closely mimic benign flow characteristics evade detection during early exposure.
2. **No Hardware Equivalence:** Simulated node load and packet queues do not capture hardware-level hypervisor cache contention, PCIe bus latency, or kernel context switches.
3. **Operational Trade-Offs:** Optimizing for high recall ($\ge 90\%$) under non-stationary traffic inherently increases the false-positive rate compared to a stationary static classifier. In real-world security operations, this requires balancing alert fatigue against missed intrusion tolerance.
