# CICIDS2017 Corrected Dataset Audit & Global Deduplication Report

**Audit Date:** 2026-09-18T10:15:34.705451  
**Evaluation Protocol:** Full-Dataset Global Exact Deduplication via SHA-256 Row Hashing  
**Source Dataset:** 8 Chronological CSV Files (`dataset/`)

---

## 1. Global Deduplication Summary

| Metric | Value |
| :--- | :--- |
| **Total Raw Flows** | 2,830,743 |
| **Unique Deduplicated Flows** | 2,521,664 |
| **Duplicate Flows Removed** | 309,079 |
| **Global Duplicate Rate** | 10.92% |
| **Canonical Numeric Features** | 77 (excluding `Label` & duplicate `Fwd Header Length.1`) |
| **Constant (Zero-Variance) Features** | 8 |
| **Total Missing Feature Cells** | 5,734 |
| **Total Infinite Feature Cells** | 4,376 |

---

## 2. File-by-File Ingress & Deduplication

| Source File | Source Day | Raw Rows | Deduped Rows | Removed Dupes | Dupe Rate (%) | Missing / Inf |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `Monday-WorkingHours.pcap_ISCX.csv` | Monday | 529,918 | 502,983 | 26,935 | 5.08% | 874 / 810 |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | Tuesday | 445,909 | 421,791 | 24,118 | 5.41% | 528 / 327 |
| `Wednesday-workingHours.pcap_ISCX.csv` | Wednesday | 692,703 | 602,490 | 90,213 | 13.02% | 2,594 / 1,586 |
| `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | Thursday-Morning | 170,366 | 156,252 | 14,114 | 8.28% | 270 / 250 |
| `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | Thursday-Afternoon | 288,602 | 242,424 | 46,178 | 16.0% | 414 / 396 |
| `Friday-WorkingHours-Morning.pcap_ISCX.csv` | Friday-Morning | 191,033 | 173,036 | 17,997 | 9.42% | 244 / 216 |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | Friday-PortScan | 286,467 | 204,927 | 81,540 | 28.46% | 742 / 727 |
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | Friday-DDoS | 225,745 | 217,761 | 7,984 | 3.54% | 68 / 64 |

---

## 3. Class Distribution & Retention After Deduplication

| Attack Class | Attack Family | Raw Count | Deduped Count | Dupes Removed | Retention Rate |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **BENIGN** | BENIGN | 2,273,097 | 2,096,391 | 176,706 | 92.23% |
| **DoS Hulk** | DoS | 231,073 | 172,811 | 58,262 | 74.79% |
| **DDoS** | DDoS | 128,027 | 128,013 | 14 | 99.99% |
| **PortScan** | PortScan | 158,930 | 90,255 | 68,675 | 56.79% |
| **DoS GoldenEye** | DoS | 10,293 | 10,286 | 7 | 99.93% |
| **FTP-Patator** | Brute Force | 7,938 | 5,933 | 2,005 | 74.74% |
| **DoS slowloris** | DoS | 5,796 | 5,385 | 411 | 92.91% |
| **DoS Slowhttptest** | DoS | 5,499 | 5,228 | 271 | 95.07% |
| **SSH-Patator** | Brute Force | 5,897 | 3,219 | 2,678 | 54.59% |
| **Bot** | Botnet | 1,966 | 1,953 | 13 | 99.34% |
| **Web Attack - Brute Force** | Web Attack | 1,507 | 1,470 | 37 | 97.54% |
| **Web Attack - XSS** | Web Attack | 652 | 652 | 0 | 100.00% |
| **Infiltration** | Infiltration | 36 | 36 | 0 | 100.00% |
| **Web Attack - Sql Injection** | Web Attack | 21 | 21 | 0 | 100.00% |
| **Heartbleed** | Exploit/Heartbleed | 11 | 11 | 0 | 100.00% |

---

## 4. Constant Features Pruned

The following 8 features exhibit zero empirical variance across all 2.83M records and are safely pruned:
- `Bwd Avg Bulk Rate`
- `Bwd Avg Bytes/Bulk`
- `Bwd Avg Packets/Bulk`
- `Bwd PSH Flags`
- `Bwd URG Flags`
- `Fwd Avg Bulk Rate`
- `Fwd Avg Bytes/Bulk`
- `Fwd Avg Packets/Bulk`
