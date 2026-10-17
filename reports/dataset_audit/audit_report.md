# CICIDS2017 Comprehensive Dataset Audit Report

**Date:** 2026-09-18
**Total Files Audited:** 8
**Total Records (Flows):** 2,830,743
**Total Features:** 78 numeric flow attributes

## 1. Executive Summary & Data Integrity

- **Total Missing Values:** 5,734
- **Total Infinite Values (±Inf):** 4,376 (predominantly in `Flow Bytes/s` and `Flow Packets/s`)
- **Total Duplicate Rows:** 256,457 (9.06%)
- **Identified Constant Features (8):** `Bwd Avg Bulk Rate, Bwd Avg Bytes/Bulk, Bwd Avg Packets/Bulk, Bwd PSH Flags, Bwd URG Flags, Fwd Avg Bulk Rate, Fwd Avg Bytes/Bulk, Fwd Avg Packets/Bulk`
- **Feature Leakage Check:** `PASSED - Clean`

## 2. Binary Class Distribution (Task A)

| Class | Count | Percentage | Imbalance Ratio |
| :--- | :--- | :--- | :--- |
| **BENIGN (0)** | 2,273,097 | 80.3% | 1.00 : 1 |
| **ATTACK (1)** | 557,646 | 19.7% | 1 : 4.08 |

## 3. Multiclass Distribution & Attack Families (Task B)

| Label | High-Level Attack Family | Count | Pct (%) | Ratio to Benign |
| :--- | :--- | :--- | :--- | :--- |
| `BENIGN` | BENIGN | 2,273,097 | 80.3004% | 1 : 1.0 |
| `DoS Hulk` | DoS | 231,073 | 8.1630% | 1 : 9.84 |
| `PortScan` | PortScan | 158,930 | 5.6144% | 1 : 14.3 |
| `DDoS` | DDoS | 128,027 | 4.5227% | 1 : 17.75 |
| `DoS GoldenEye` | DoS | 10,293 | 0.3636% | 1 : 220.84 |
| `FTP-Patator` | Brute Force | 7,938 | 0.2804% | 1 : 286.36 |
| `SSH-Patator` | Brute Force | 5,897 | 0.2083% | 1 : 385.47 |
| `DoS slowloris` | DoS | 5,796 | 0.2048% | 1 : 392.18 |
| `DoS Slowhttptest` | DoS | 5,499 | 0.1943% | 1 : 413.37 |
| `Bot` | Botnet | 1,966 | 0.0695% | 1 : 1156.2 |
| `Web Attack - Brute Force` | Web Attack | 1,507 | 0.0532% | 1 : 1508.36 |
| `Web Attack - XSS` | Web Attack | 652 | 0.0230% | 1 : 3486.35 |
| `Infiltration` | Infiltration | 36 | 0.0013% | 1 : 63141.58 |
| `Web Attack - Sql Injection` | Web Attack | 21 | 0.0007% | 1 : 108242.71 |
| `Heartbleed` | Exploit/Heartbleed | 11 | 0.0004% | 1 : 206645.18 |

## 4. Per-File Statistics

| File Name | Size (MB) | Rows | Duplicates | Inf Values | Missing Values |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | 73.55 | 225,745 | 2,630 | 64 | 68 |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | 73.34 | 286,467 | 72,339 | 727 | 742 |
| `Friday-WorkingHours-Morning.pcap_ISCX.csv` | 55.62 | 191,033 | 6,884 | 216 | 244 |
| `Monday-WorkingHours.pcap_ISCX.csv` | 168.73 | 529,918 | 26,919 | 810 | 874 |
| `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | 79.25 | 288,602 | 35,624 | 396 | 414 |
| `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | 49.61 | 170,366 | 6,061 | 250 | 270 |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | 128.82 | 445,909 | 24,058 | 327 | 528 |
| `Wednesday-workingHours.pcap_ISCX.csv` | 214.74 | 692,703 | 81,942 | 1,586 | 2,594 |

## 5. Sanitization and Preprocessing Mandates

1. **Whitespace & Unicode:** Column headers trimmed; Web Attack non-breaking/corrupted dashes normalized.
2. **Inf and NaN Treatment:** All $\pm\infty$ converted to NaN and imputed using median statistics derived strictly from training partitions.
3. **Constant Columns:** Constant zero-variance bulk rate columns (`Bwd Avg Bytes/Bulk`, `Fwd Avg Bulk Rate`, etc.) pruned from model input.
4. **Zero Leakage Invariant:** All transformations fit exclusively on the training split; test streams evaluated prequentially.
