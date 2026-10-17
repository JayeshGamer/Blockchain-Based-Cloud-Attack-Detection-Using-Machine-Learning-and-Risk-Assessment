# Presentation Outline: Adaptive ML-DST

**Title:** Adaptive Temporal Risk Assessment for Blockchain-Based Cloud Attack Tolerance Using Online Machine Learning and Dempster–Shafer Evidence Fusion  
**Target Audience:** Capstone Defense / IEEE Conference / Undergraduate Research Symposium  
**Duration:** 15–20 Minutes (15 Slides)

---

### Slide 1: Title & Author Information
- **Title:** Adaptive ML-DST: Online Machine Learning and Uncertainty-Aware Temporal Risk Assessment for Cloud Attack Tolerance
- **Subtitle:** An Undergraduate Research Extension of Benmenzer et al., *IET Blockchain*, 2025
- **Presenter:** Research Team / Department of Computer Science & Engineering
- **Visuals:** Department logo, IEEE conference template banner.

---

### Slide 2: Context & The Base Paper
- **Reference:** Benmenzer et al., *"Reinforcing Cloud Attack Tolerance on a Large Scale: Advanced Blockchain Technologies With Hierarchical Risk and Smart Contracts,"* *IET Blockchain*, 2025.
- **Base Architecture:** 3-Tier Hierarchical Risk (Node $\to$ Cluster $\to$ Global), Dempster--Shafer Theory (DST) evidential fusion, dynamic thresholds, Proof of Risk consensus in Hyperledger Fabric.
- **Base Reported Results:** $>99.9\%$ detection rate in simulated environments.
- **Key Visual:** Base paper 3-tier architecture diagram.

---

### Slide 3: The Research Problem & Identified Gap
- **Acknowledged Limitation:** The base paper explicitly stated that its risk analysis algorithm requires further testing for dynamic attack variability and temporal risk fluctuations.
- **Stated Future Work:** Integrate machine learning to learn from historical and real-time streaming data, identify subtle patterns, and adapt to temporal drift.
- **Our Goal:** Operationalize this future work by replacing static heuristic attribute mappings with an online, drift-aware evidential risk engine.

---

### Slide 4: Proposed "Adaptive ML-DST" Architecture
- **Novel Contributions:**
  1. Prequential online learning (predict-then-update).
  2. ADWIN statistical concept-drift detection.
  3. Benign-trained unsupervised anomaly scoring (Isolation Forest).
  4. Rolling temporal deviation tracking.
  5. Tri-source Dempster--Shafer evidence fusion with explicit epistemic uncertainty.
  6. Drift-aware dynamic threshold adjustment bounded within $[\theta_{\min}, \theta_{\max}]$.
- **Visual:** `fig1_architecture_pipeline.png`.

---

### Slide 5: Mathematical Foundations: DST Evidential Fusion
- **Frame of Discernment:** $\Omega = \{\text{Benign}, \text{Attack}\}$; Uncertainty State: $\Theta = \{\text{Benign}, \text{Attack}\}$.
- **Mass Assignment:** $m_j(\text{Atk}) = s_j(1 - u_j)$, $m_j(\text{Ben}) = (1 - s_j)(1 - u_j)$, $m_j(\Theta) = u_j$.
- **Dempster's Rule:** Formally combines independent sources; safe conflict factor $k_{\text{eff}} = \min(k, 0.9999)$ prevents division by zero.
- **Decision Metric:** Pignistic risk score $R_{\text{flow}} = \text{Bel}(\text{Attack}) + 0.5 \cdot m(\Theta)$.
- **Visual:** `fig8_dst_evidence_flow.png`.

---

### Slide 6: Concept Drift & Dynamic Thresholding
- **ADWIN (Adaptive Windowing):** Tracks streaming classification error; detects distribution shifts via Hoeffding subwindow cuts.
- **Drift Response:**
  - Boosts online model plasticity.
  - Dynamically relaxes threshold ($\rho = 0.85$) to catch emerging attacks.
  - Smooth exponential recovery back to background risk quantiles.
  - Strict bounding: $\theta_t \in [0.20, 0.85]$.
- **Visual:** `fig4_streaming_risk_threshold.png`.

---

### Slide 7: Dataset Audit & Preprocessing Integrity
- **Dataset:** CICIDS2017 (2.83 Million flows, 79 initial columns across 8 CSV files).
- **Data Integrity Audit:**
  - Trimmed whitespace and repaired corrupted UTF-8 replacement characters in Web Attack labels.
  - Identified $4,376$ infinities and $5,734$ missing values.
  - Pruned 8 zero-variance constant bulk features and 1 duplicate column.
- **Zero Leakage Invariant:** Medians, scalers, and encoders fitted strictly on training partitions; streaming evaluated prequentially.
- **Visual:** `fig2_class_distribution.png`.

---

### Slide 8: Experimental Setup & Baselines
- **Tasks:**
  - Task A: Primary Binary Detection (BENIGN vs ATTACK).
  - Task B: Secondary Multiclass Detection (9 Attack Families).
- **Evaluated Baselines:**
  1. Dummy Majority
  2. Logistic Regression (Linear)
  3. Random Forest (Static Ensemble)
  4. Isolation Forest (Unsupervised)
  5. Static DST (Base Paper Non-learning Model)
  6. Adaptive ML-DST (Proposed)
- **Statistical Protocol:** 5 Fixed Seeds (42, 123, 456, 789, 1024); reporting Mean, Std, and 95% CIs.

---

### Slide 9: Comparative Results: Realistic Pareto Trade-offs
- **Key Takeaway 1 (Volumetric High Recall):** On high-volume attacks, our streaming model achieves **$95.51\%$ Recall on DDoS** and **$97.13\%$ Recall on PortScan** ($0.9687$ Attack F1).
- **Key Takeaway 2 (Balanced Continuous Streaming):** Across all 7 days with mixed stealthy web probes, overall prequential accuracy is **$93.56\%$**, with **$86.88\%$ Recall** and a low **$2.66\%$ FPR** ($0.9432$ ROC-AUC, $0.9225$ PR-AUC).
- **Key Takeaway 3 (Configurable Operating Profiles):** 
  - *Balanced Mode:* $93.56\%$ Acc, $86.88\%$ Recall, $2.66\%$ FPR.
  - *High-Recall SOC Mode:* Enforcing threshold calibration pushes recall to **$92.37\%$** ($5.89\%$ FPR).
  - *Stationary Baseline (Random Forest):* Achieves $99.50\%$ accuracy on stationary data, but collapses ($99.39\%$ FNR) under streaming concept drift, justifying our adaptive online approach.

---

### Slide 10: Ablation Study: RQ5
- **Ablation Progression:**
  - Variant A (Static ML): High F1, but $11.36\%$ FPR.
  - Variant B (Online ML): Real-time adaptation, but $13.10\%$ FPR.
  - Variant E (Online ML + DST): Cuts FPR to $4.45\%$ ($66\%$ reduction).
  - Variant F (Full Adaptive ML-DST): Further suppresses FPR to $2.09\%$.
- **Conclusion:** DST evidence fusion and adaptive thresholding are the primary drivers of false positive suppression in cloud monitoring.
- **Visual:** `fig5_ablation_study.png`.

---

### Slide 11: Task B: Multiclass Family Detection
- **High-Support Families:** Excellent detection across DoS ($0.87$ F1), Brute Force ($0.88$ F1), and Web Attacks ($0.86$ F1).
- **Honest Handling of Rare Attacks:** Extremely rare attacks (Heartbleed: 3 test flows, Infiltration: 9 test flows) have insufficient gradient steps for linear online models.
- **Reported as Insufficient-Support Classes:** Preserved without silent deletion.
- **Visual:** `fig6_per_class_f1.png`.

---

### Slide 12: Overhead & Streaming Throughput: RQ4
- **Throughput:** $4,944.7$ flows per second on standard CPU.
- **Per-Flow Latency:** $0.202$ milliseconds.
- **Memory Overhead:** Only $21.89$ MB above baseline.
- **Suitability:** Light enough to run directly on edge gateways, micro-proxies, and cloud ingress nodes.
- **Visual:** `fig7_robustness_stress.png` (b).

---

### Slide 13: Simulated Defense & Blockchain Audit Ledger
- **Staged Defense Activation:** Log $\to$ Rate Limit $\to$ Traffic Filter / Node Isolation $\to$ Global Cluster Rerouting.
- **Blockchain Audit Ledger:** Solidity smart contract (`RiskAuditLedger.sol`) recording SHA-256 evidence digests.
- **Privacy-Preserving:** Zero PII or raw network payloads stored on-chain.
- **Mock Ledger Fallback:** Seamless in-memory execution without external node dependencies.

---

### Slide 14: Interactive Dashboard & API Demo
- **FastAPI REST Backend:** 13 endpoints providing live telemetry, health checks, and controls.
- **Streamlit Dashboard:** Real-time hierarchical risk gauges, drift timeline, confusion matrix, and downloadable audit reports.
- **Demonstration:** Brief walkthrough of the interactive UI.

---

### Slide 15: Conclusion & Future Frontiers
- **Summary of Achievements:**
  - Fully realized the future-work direction of Benmenzer et al. (2025).
  - Validated across 2.83M flows and 5 deterministic seeds.
  - Achieved $0.9000$ ROC-AUC, slashed false alarms by $84\%$, and maintained $4,940+$ FPS throughput.
- **Future Work:** Hoeffding Adaptive Trees, multi-cloud Kubernetes deployment, and Layer-2 zero-knowledge audit rollups.
- **Questions:** Open the floor for examination questions.
