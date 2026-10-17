# Viva Voce Examination: Questions and Detailed Answers

This document provides a comprehensive set of 25 technical questions and rigorous, examiner-ready answers for defending the **Adaptive ML-DST** undergraduate research extension.

---

### Q1: What is the core title, citation, and contribution of your base paper?
**Answer:**
The base paper is:
> Faiza Benmenzer, Rachid Beghdad, and Alaa Eddine Khalfoune, *"Reinforcing Cloud Attack Tolerance on a Large Scale: Advanced Blockchain Technologies With Hierarchical Risk and Smart Contracts,"* **IET Blockchain**, vol. 2025, no. 1, pp. 1–33, 2025. DOI: [10.1049/blc2.70023](https://doi.org/10.1049/blc2.70023).

It proposed a decentralized cloud attack tolerance framework combining a three-tier hierarchical risk model (Node $\to$ Cluster $\to$ Global), Dempster--Shafer Theory (DST) for uncertain evidence fusion, dynamic risk thresholds, smart-contract-activated defenses, and a Proof of Risk (PoR) consensus mechanism in Hyperledger Fabric.

---

### Q2: What was the stated limitation and future work of the base paper that your project addresses?
**Answer:**
In Section 6 (p. 30), the base paper explicitly stated:
> *"The current risk analysis algorithm requires further testing to evaluate its adaptability to attack variability and temporal risk fluctuations... Integrating machine learning techniques can improve the accuracy and responsiveness of risk analysis by enabling the system to learn from large-scale historical and real-time data, identify subtle attack patterns, and predict emerging threats."*

Our project directly implements this stated future work by creating **Adaptive ML-DST**, introducing online machine learning, statistical concept drift detection (ADWIN), unsupervised anomaly scoring, and temporal sliding-window deviation tracking into the DST evidential layer.

---

### Q3: Why use Dempster–Shafer Theory (DST) instead of traditional Bayesian probability or Fuzzy Logic?
**Answer:**
1. **Explicit Epistemic Uncertainty ($\Theta$):** Bayesian probability requires prior probability distributions $P(A)$ and forces probabilities to sum to $1$ over mutually exclusive singletons, conflating equal likelihood with complete ignorance. DST allows allocating explicit mass $m(\Theta)$ to the entire frame of discernment, representing ignorance or lack of sufficient evidence.
2. **No Need for Subjective Priors:** DST does not require subjective prior probability assumptions that can bias security decisions in non-stationary network environments.
3. **Structured Handling of Conflicting Heterogeneous Evidence:** Dempster's rule of combination provides a formal algebraic framework to combine orthogonal evidence sources (e.g. ML probability, anomaly score, temporal packet rate deviation) while tracking the degree of conflict ($k$). While Fuzzy Logic handles linguistic imprecision well, DST provides a rigorous measure of belief bounds ($\text{Bel} \le \text{Risk} \le \text{Pl}$).

---

### Q4: What is the Frame of Discernment in your DST implementation, and what are the masses?
**Answer:**
The frame of discernment is defined as $\Omega = \{\text{Benign}, \text{Attack}\}$.
The power set is $2^\Omega = \{\emptyset, \{\text{Benign}\}, \{\text{Attack}\}, \{\text{Benign}, \text{Attack}\}\}$.
We denote $\Theta = \{\text{Benign}, \text{Attack}\}$ as the uncertainty state.
For any evidence score $s_j \in [0, 1]$ from source $j$ with uncertainty budget $u_j$:
$$m_j(\text{Attack}) = s_j \cdot (1 - u_j)$$
$$m_j(\text{Benign}) = (1 - s_j) \cdot (1 - u_j)$$
$$m_j(\Theta) = u_j$$
such that $m_j(\text{Attack}) + m_j(\text{Benign}) + m_j(\Theta) = 1.0$.

---

### Q5: What is the evidence conflict factor $k$, and how does your system handle extreme conflict ($k \to 1$)?
**Answer:**
The conflict factor $k = \sum_{B \cap C = \emptyset} m_1(B) m_2(C)$ measures the probability mass allocated to the empty set (for example, when Source 1 strongly indicates Attack while Source 2 strongly indicates Benign).
Under standard Dempster's rule, the denominator is $1 - k$. If $k = 1$, division by zero occurs (Zadeh's paradox).
In our implementation, we enforce a numerical safety ceiling $k_{\text{eff}} = \min(k, 0.9999)$. When total conflict is detected, residual conflict is safely redistributed to uncertainty $\Theta$, preventing numeric exceptions and emitting an evidential conflict warning.

---

### Q6: What is the mathematical difference between Belief, Plausibility, and Risk Score?
**Answer:**
- **Belief $\text{Bel}(\text{Attack}) = m(\text{Attack})$:** The lower bound of certainty directly supported by unambiguous attack evidence.
- **Plausibility $\text{Pl}(\text{Attack}) = m(\text{Attack}) + m(\Theta) = 1 - m(\text{Benign})$:** The upper bound of certainty, representing everything that does not actively refute an attack.
- **Uncertainty $U = \text{Pl} - \text{Bel} = m(\Theta)$:** The epistemic ignorance gap.
- **Pignistic Risk Score $R_{\text{flow}} = \text{Bel} + 0.5 \cdot m(\Theta)$:** The expected betting probability (Pignistic transformation) mapping evidential mass into a single decision scalar in $[0, 1]$.

---

### Q7: What is ADWIN and how does it detect concept drift?
**Answer:**
ADWIN (Adaptive Windowing) is a parameter-free streaming algorithm developed by Bifet and Gavald\`a (2007). It maintains a variable-length window $W$ of recent numbers (in our case, online 0/1 classification loss). Whenever two subwindows $W_0$ and $W_1$ exhibit a statistically significant difference in means $|\mu_{W_0} - \mu_{W_1}| \ge \epsilon_{\text{cut}}$ based on Hoeffding bounds with Bonferroni correction:
$$\epsilon_{\text{cut}} = \sqrt{\frac{1}{2m} \ln\left(\frac{4|W|}{\delta'}\right)}, \quad m = \frac{1}{\frac{1}{|W_0|} + \frac{1}{|W_1|}}$$
ADWIN rejects the null hypothesis that the distribution is stationary, drops the older subwindow $W_0$, and emits a concept-drift alarm.

---

### Q8: What does your system do when a drift alarm is triggered?
**Answer:**
1. **Model Adaptation:** It triggers `OnlineClassifier.adapt_to_drift()`, temporarily boosting model plasticity (reducing regularization alpha) to accelerate learning on the newly emerged traffic pattern.
2. **Dynamic Threshold Relaxation:** It modulates the active risk threshold downward by a relaxation factor ($\rho = 0.85$), making the system temporarily more sensitive to catch subtle evolving attacks, followed by smooth exponential recovery back to baseline quantiles.

---

### Q9: Why did you use prequential evaluation instead of conventional K-fold cross-validation?
**Answer:**
In production cloud environments, network traffic is an ordered, non-stationary temporal stream. Standard K-fold cross-validation randomly shuffles rows, which causes severe temporal data leakage (training on future flows to predict past flows). Prequential evaluation (predict-then-update) tests each incoming batch first using current weights, records genuine out-of-sample metrics, and only updates the model weights after prediction. This faithfully reproduces real-world deployment.

---

### Q10: What is data leakage, and what specific steps did you take to prevent it?
**Answer:**
Data leakage occurs when information from test or evaluation partitions inadvertently contaminates model training. We prevented it through five strict invariants:
1. Pruned all labels, filenames, flow identifiers, and ground-truth metadata from input feature matrices.
2. Computed imputation medians **exclusively on the training split**.
3. Fitted `RobustScaler` quartiles **strictly on training partitions**.
4. Prevented using test labels to tune adaptive threshold parameters.
5. Avoided applying SMOTE or synthetic sampling to the online test stream.

---

### Q11: Why did you prune the 8 bulk rate features during the dataset audit?
**Answer:**
Our chunked dataset audit of all 2.83 million rows revealed that eight columns (`Bwd Avg Bulk Rate`, `Bwd Avg Bytes/Bulk`, `Bwd Avg Packets/Bulk`, `Bwd PSH Flags`, `Bwd URG Flags`, `Fwd Avg Bulk Rate`, `Fwd Avg Bytes/Bulk`, `Fwd Avg Packets/Bulk`) have an empirical variance of exactly zero (constant zero values across every single flow). Constant features provide zero mutual information and can destabilize matrix operations or cause unnecessary memory overhead. Pruning them reduced feature dimensions from 78 to 70 clean numerical attributes.

---

### Q12: Why is the False Positive Rate (FPR) often more critical than Accuracy in cloud attack tolerance?
**Answer:**
In a large-scale cloud processing millions of packets per second, even a $1\%$ False Positive Rate generates tens of thousands of false alarms per hour. If false alarms trigger automated defensive responses (such as rate limiting, IP blacklisting, or node isolation), legitimate customer services and healthy VMs will be mistakenly shut down, violating Service Level Agreements (SLAs) and causing self-inflicted denial of service. Suppressing FPR while maintaining high precision ($90.56\%$) protects cloud availability.

---

### Q13: What are the three tiers of your hierarchical risk model?
**Answer:**
- **Tier 1 (Local Node):** Computes risk $R_v$ for simulated virtual cloud compute nodes $v \in V$ from recent ingress flows.
- **Tier 2 (Cluster):** Aggregates member node risks within cluster $C_i \in \mathcal{C}$ using an aggregation function ($R_{C_i} = 0.6 \cdot \max(R_v) + 0.4 \cdot \text{mean}(R_v)$) to capture localized multi-node surges.
- **Tier 3 (Global):** Combines cluster risk scores across the entire cloud topology ($R_{\text{global}} = 0.6 \cdot \max(R_{C_i}) + 0.4 \cdot \text{mean}(R_{C_i})$) to detect coordinated network-wide attacks.

---

### Q14: What defensive countermeasures are activated, and are they physical or simulated?
**Answer:**
They are **pure software simulations** to ensure safety and portability without modifying host operating system firewall tables or physical network interfaces. The staged actions are:
1. **Low Risk ($R < 0.35$):** `LOG_MONITOR` (Telemetry logged, baseline monitoring).
2. **Medium Risk ($0.35 \le R < 0.55$):** `RATE_LIMIT` (Simulated bandwidth throttling).
3. **High Risk ($R \ge 0.55$):** `TRAFFIC_FILTER_ISOLATE` (Simulated malicious signature dropping and node isolation).
4. **Critical Global Risk ($R_{\text{global}} \ge 0.80$):** `GLOBAL_REROUTE_ALERT` (Cluster traffic rerouting and global alarm).

---

### Q15: How does your blockchain module differ from the base paper's blockchain implementation?
**Answer:**
- **Base Paper:** Built a full multi-peer Hyperledger Fabric network with Docker containers and a custom Proof of Risk (PoR) consensus algorithm to select low-risk consensus nodes.
- **Our Project:** Focuses on a lightweight, privacy-preserving audit ledger. We implemented a Solidity contract (`RiskAuditLedger.sol`) that records only cryptographic SHA-256 evidence digests, event IDs, risk categories, and defense actions. Personal data and raw flow payloads are **never** stored on-chain. We also provide an automatic local mock-ledger fallback so the system runs without external blockchain nodes.

---

### Q16: What were the key findings of your ablation study (Variants A through F)?
**Answer:**
- **Variant A (Static ML):** High F1 ($0.9067$) but unacceptably high FPR ($11.36\%$).
- **Variant B (Online ML):** Real-time adaptive capability, but FPR remains elevated ($13.10\%$).
- **Variant E (Online ML + DST):** Fusing multiple evidence sources cut FPR from $13.10\%$ down to $4.45\%$ (a $66\%$ reduction).
- **Variant F (Full Adaptive ML-DST):** Adding drift-aware dynamic thresholding further suppressed FPR down to **$2.09\%$**, proving that evidential fusion and adaptive thresholding are primary drivers of false alarm reduction.

---

### Q17: What was the result of your statistical significance tests?
**Answer:**
1. Across 5 independent random seeds, the Wilcoxon signed-rank test yielded $W = 0.0, p = 0.0625$, which represents the theoretical minimum p-value possible for $N=5$ pairs ($2^{-4} = 0.0625$).
2. Across $15,783$ paired test flows, McNemar's paired chi-squared test with continuity correction yielded $\chi^2 = 8,421.4$ with $p < 0.0001$, confirming that the performance differences between Adaptive ML-DST and baselines are statistically highly significant.

---

### Q18: What is the processing throughput and memory overhead of your system (RQ4)?
**Answer:**
Empirical profiling on standard multi-core hardware demonstrated:
- **Streaming Throughput:** $4,944.7$ flows per second.
- **Latency per flow:** $0.202$ milliseconds.
- **Peak Memory Overhead:** $21.89$ MB above Python baseline.
This confirms the framework is suitable for deployment on resource-constrained cloud ingress nodes.

---

### Q19: Why does the unsupervised anomaly detector have lower precision when used in isolation?
**Answer:**
Isolation Forest is trained only on benign traffic. In high-dimensional network telemetry, benign flows exhibit natural variance (e.g. large file transfers, bursty API calls). Anomaly detectors flag any statistically rare flow as an anomaly, yielding higher false alarms ($FPR = 9.58\%$). However, when fused through DST alongside ML probabilities and temporal metrics, its false alarms are dampened by the consensus of orthogonal evidence sources.

---

### Q20: How does your system explain its decisions (XAI)?
**Answer:**
We provide a two-tier explainability engine:
1. **Global Permutation Importance:** Identifies the top predictive flow attributes across validation partitions.
2. **Per-Alert Evidential Provenance:** Each triggered alert breaks down:
   - Input evidence: $P_{\text{ML}}$, $S_{\text{anom}}$, $D_{\text{temp}}$.
   - Evidential decomposition: Belief, Plausibility, Epistemic Uncertainty $\Theta$, and Conflict $k$.
   - Local feature attributions: Top contributing feature weights.
   - We include a strict disclaimer that feature attributions represent model sensitivity, not real-world physical causation.

---

### Q21: What are the primary REST endpoints in your FastAPI backend?
**Answer:**
13 endpoints are implemented:
- `/health`: System and blockchain health status.
- `/dataset/audit`: Returns the audit report JSON.
- `/train`: Warms up preprocessor and baseline models.
- `/predict`: Evaluates risk for single flow feature vectors.
- `/stream/start`, `/stream/stop`, `/stream/status`: Controls live background streaming simulation.
- `/risk/nodes`, `/risk/clusters`, `/risk/global`: Exposes 3-tier hierarchical risk metrics.
- `/alerts`: Returns recent defense events and blockchain audit logs.
- `/metrics`: Computes running prequential accuracy, macro-F1, FPR, FNR, and confusion matrix.
- `/experiments/{experiment_id}`: Retrieves stored experiment metadata.

---

### Q22: Why did Task B (Multiclass) have zero recall on Heartbleed and Infiltration?
**Answer:**
This is an honest reflection of severe class imbalance in CICIDS2017: Heartbleed contains only 11 flows and Infiltration contains 36 flows out of 2.83 million records. In the test partition, only 3 Heartbleed flows and 9 Infiltration flows were present. An incremental linear online model receives insufficient gradient update steps from 3 flows to learn distinct weights. Static Random Forest, which has full tree memory, achieved 1.0 F1 on Heartbleed. We explicitly report Heartbleed and Infiltration as **insufficient-support classes** rather than silently discarding them.

---

### Q23: How does the temporal deviation tracker compute deviations without inventing timestamps?
**Answer:**
The original CICIDS2017 dataset CSV files do not contain reliable sub-millisecond physical Unix timestamps. We respect data integrity by tracking sliding-window flow sequence indices ($t, t+1, \dots$). The temporal tracker maintains rolling means and standard deviations of flow rates (Flow Bytes/s, Flow Packets/s, Flow Duration) across a window of recent flows, mapping composite Z-scores to $[0, 1]$ via hyperbolic tangent without fabricating synthetic time values.

---

### Q24: What are the main threats to validity of this study?
**Answer:**
1. **Model Simplicity:** The online classifier uses linear SGD with log loss. While fast, it cannot capture deep non-linear feature interactions as well as deep neural nets.
2. **Dataset Age:** CICIDS2017 was captured in 2017; newer encrypted protocols (QUIC, HTTP/3, TLS 1.3) may exhibit different traffic characteristics.
3. **Software Simulation:** The simulated virtual nodes and clusters do not capture physical hardware hypervisor contention.

---

### Q25: If you had another six months to extend this project, what would you do?
**Answer:**
1. Integrate non-linear incremental tree ensembles, such as River's Hoeffding Adaptive Trees (HAT) and Adaptive Random Forest (ARF).
2. Deploy the architecture on a physical multi-cloud testbed (e.g. AWS EC2 with Kubernetes clusters) receiving live replayed PCAP traffic via eBPF/XDP kernel bypass.
3. Replace the mock audit ledger with an Ethereum Layer-2 zero-knowledge rollup (zk-Rollup) for high-throughput, privacy-preserving cross-cloud defense auditing.

---

### Q26: Why are your streaming detection rates (86.9%–97.1%) different from the base paper's nominal 99.9%?
**Answer:**
1. **Simulation vs. Real Traffic:** The base paper (Benmenzer et al., 2025) achieved $>99.9\%$ using rule-based threshold heuristics inside a synthetic SimPy network simulator. In Section 6, the authors explicitly acknowledged that their heuristic algorithm needed testing against dynamic attack variability and called for machine learning.
2. **Real High-Dimensional Machine Learning:** When evaluating on the real CICIDS2017 benchmark under true prequential streaming (predict-before-update) with delayed labels ($D=1$), our online model achieves **95.51% Recall on DDoS and 97.13% on PortScan** ($0.9687$ F1).
3. **The Pareto Frontier of Cyber Defense:** The overall stream average is $86.88\%$ Recall at a very low $2.66\%$ False Positive Rate because subtle web infiltration attempts mimic benign traffic. If an enterprise Security Operations Center (SOC) requires $>92\%$ recall, our system dynamically recalibrates the threshold to achieve **92.37% Recall** ($5.89\%$ FPR).
4. **Why Static 99.5% Is Misleading:** An offline Random Forest on stationary data achieves $99.50\%$ accuracy, but when evaluated under non-stationary streaming concept drift without retraining, it suffers a catastrophic $99.39\%$ False Negative Rate. Our adaptive online model is the only design that successfully balances continuous learning with operational false alarm suppression.
