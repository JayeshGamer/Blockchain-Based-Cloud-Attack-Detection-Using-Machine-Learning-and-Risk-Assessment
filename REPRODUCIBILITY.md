# V6 Reproducibility Instructions & Verification Guide

## 1. System Requirements & Environment

- **Operating System**: Windows / Linux / macOS
- **Python**: Python 3.10+ (Tested on Python 3.12.7)
- **Key Dependencies**:
  - `numpy>=1.26.0`
  - `pandas>=2.1.0`
  - `scikit-learn>=1.4.0`
  - `river>=0.21.0`
  - `pyyaml>=6.0`
  - `pytest>=8.0.0`
  - `psutil>=5.9.0`

Install all dependencies via:
```bash
pip install -r requirements.txt
```

---

## 2. Test Suite Execution

### A. Run Targeted V6 Verification Suite (10 Requirements Tests)
```bash
pytest tests/test_v6_verification.py -v
```
**Expected Outcome**: 10 passed, 0 failed in ~15-20 seconds. Demonstrates:
1. Incremental prediction finalization without memory accumulation.
2. Independent, model-specific ROC-AUC and PR-AUC mapping.
3. Row-level Monday warm-up filtering and rebatching.
4. Per-row daily metric attribution.
5. Invariance of evaluated sequence to input chunk size.
6. Calibration FPR measurement under tied and finite-sample distributions.
7. Informative multi-file smoke test structure.
8. Delayed feedback queue bounded-memory guarantees.
9. Configuration propagation through Protocol B runner.
10. Defense engine bounded queue memory limits.

### B. Run Complete Project Test Suite (80 Tests)
```bash
pytest tests/ -v
```
**Expected Outcome**: 80 passed, 0 failed in ~55 seconds.

---

## 3. Informative Full-Path Smoke Test Execution

To execute the genuine chunked streaming engine across multi-file non-Monday attack windows (Tuesday, Wednesday, Friday-PortScan) with zero row accumulation:
```bash
python scripts/smoke_test_full_mode.py
```
**Artifacts Generated**:
- `reports/experiments/smoke_test/protocol_b_predictions.jsonl.gz`
- `reports/experiments/smoke_test/protocol_b_results.json`
- `reports/experiments/smoke_test/full_experiment_results.json`
- `reports/experiments/smoke_test/tables/` (5 CSV tables)
- `reports/experiments/smoke_test/IMPLEMENTATION_RESULTS.md`

---

## 4. Lightweight Benchmark Evaluation

To execute duplicate-safe Protocol A, prequential streaming Protocol B, adversarial evasion Protocol C, and fair ablation variants:
```bash
python scripts/run_experiments.py --config configs/lightweight.yaml --output-dir reports/experiments/lightweight
```
**Artifacts Generated**:
- `reports/experiments/lightweight/protocol_a_results.json`
- `reports/experiments/lightweight/protocol_b_results.json`
- `reports/experiments/lightweight/protocol_b_predictions.jsonl.gz`
- `reports/experiments/lightweight/protocol_c_results.json`
- `reports/experiments/lightweight/ablation_study_results.json`
- `reports/experiments/lightweight/full_experiment_results.json`
- `reports/experiments/lightweight/tables/`
- `reports/experiments/lightweight/IMPLEMENTATION_RESULTS.md`

---

## 5. Independent Prediction Integrity & Metric Recomputation

To verify that predictions were streamed incrementally without memory leaks, confirm zero Monday contamination, and recompute confusion matrices and AUC metrics bit-for-bit from disk-backed logs:
```bash
python scripts/verify_prediction_integrity.py
```
**Expected Outcome**: 100% confusion matrix match across all 8 models; bit-for-bit ROC-AUC and Average Precision verification.

---

## 6. Complete 2.83M Streaming Evaluation (Optional / High-Memory)

The shared `StreamingEvaluationEngine` and out-of-core pipeline in `src/data/pipeline.py` are fully verified for complete out-of-core execution over all 2,830,743 rows across 8 raw CSVs.
```bash
python scripts/run_experiments.py --config configs/full.yaml --output-dir reports/experiments/full_run
```
*Note*: This executes chunked extraction, deduplication, row-level Monday exclusion, and prequential streaming across ~2 GB of raw CSVs, requiring approximately 1.5 to 2.5 hours depending on disk I/O.
