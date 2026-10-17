# Invalidation Notice: Preliminary Experimental Results

**Status:** ARCHIVED & INVALIDATED  
**Date of Invalidation:** September 18, 2026  
**Applicability:** All preliminary experimental result files, JSON summaries, CSV tables, and PNG figures generated prior to the Phase 2 dataset correction and temporal validation audit.

---

## Executive Summary

The experimental outputs stored in this directory (`reports/archive_preliminary_invalid/`) are preliminary artifacts and **cannot be used for academic publication or performance claims**. A systematic methodology audit revealed fundamental empirical and procedural defects that undermine the scientific validity of the reported figures.

---

## Specific Grounds for Invalidation

1. **Restricted Non-Representative Sample:**
   - The experiments evaluated a 63,130-row subsample rather than the complete 2.83-million flow CICIDS2017 collection.
   - The sampling mechanism evaluated class rarity within localized 30,000-row chunks, misclassifying major attack categories (such as DDoS and DoS Hulk) as rare when chunk slices had low counts.

2. **Absence of Genuine Chronological Replay:**
   - The main evaluation protocol utilized a conventional random stratified split (`train_test_split`) rather than preserving the chronological order of flows across days (Monday through Friday).
   - Random shuffling violates the non-stationary temporal structure of network attacks and masks concept drift.

3. **Cross-Partition Duplicate Contamination:**
   - Global exact deduplication via cryptographic row hashes (SHA-256) was not performed prior to splitting.
   - Identical network flow records crossed training and testing partitions, artificially inflating generalization metrics.

4. **Severely Degraded Attack Recall and Excessive False Negative Rate:**
   - The proposed streaming configuration produced an empirical Recall of approximately **24.41%** and a catastrophic False Negative Rate (FNR) of **75.59%**.
   - Over three-quarters of attack flows were missed. The seemingly low False Positive Rate (FPR) was an artifact of an overly conservative threshold driven up by contaminating attack scores in the benign baseline buffer.

5. **Synthetic Data in Experimental Visualizations:**
   - `scripts/generate_plots.py` (specifically `fig4_streaming_risk_threshold.png`) generated streaming risk trajectories and ADWIN concept-drift points using pseudo-random numbers (`np.random.randn`) rather than streaming execution logs.

6. **Flawed Latency, Throughput, and Memory Isolation:**
   - Reported throughput and peak memory metrics were computed across the execution runtime of the entire test suite rather than isolating the proposed ML-DST model pipeline independently under controlled conditions.

7. **Inaccurate Baseline Reproduction:**
   - The simplified DST baseline in the preliminary suite did not faithfully reproduce the hierarchical, smart-contract-coupled formulation of the base paper (*Benmenzer et al., IET Blockchain 2025*).

8. **Lack of Traceability for Statistical Claims:**
   - Several statistical claims (e.g., p-values, McNemar's tests) were not traceable to persisted paired flow-level prediction arrays.

---

## Remediation Requirements

All subsequent evaluations must adhere to the three verified protocols:
- **Protocol A:** Duplicate-safe, hash-disjoint conventional evaluation with validation-calibrated thresholds.
- **Protocol B:** Genuine chronological prequential evaluation (predict-before-update with delayed label feedback).
- **Protocol C:** Controlled concept-drift benchmark with known ground-truth change points.
