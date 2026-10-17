# Raw Dataset Directory (CICIDS2017)

The raw CICIDS2017 CSV captures belong in this folder but are **not tracked by Git**
(~843 MB compressed on disk, ~1.0 GB of CSV text; individual files exceed GitHub's
100 MB per-file limit). See the [root README](../README.md#-dataset) for details.

## Expected files

Place these eight CSV files directly in this directory:

| File | Flows |
| :--- | ---: |
| `Monday-WorkingHours.pcap_ISCX.csv` | 529,918 |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | 445,909 |
| `Wednesday-workingHours.pcap_ISCX.csv` | 692,703 |
| `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | 170,366 |
| `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | 288,602 |
| `Friday-WorkingHours-Morning.pcap_ISCX.csv` | 191,033 |
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | 225,745 |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | 286,467 |
| **Total** | **2,830,743** |

## How to obtain

Download the *Intrusion detection evaluation dataset (CIC-IDS2017)* from the
Canadian Institute for Cybersecurity:

<https://www.unb.ca/cic/datasets/ids-2017.html>

Extract the `MachineLearningCSV.zip` archive and copy the `*.pcap_ISCX.csv` files
into this directory, keeping the exact filenames above.

## You may not need the raw data

`data/processed/stratified_sample.csv.gz` already contains the 50,000-flow
chronological evaluation sample used in the paper, so the full pipeline
(`scripts/evaluate.py`, `scripts/predict.py`, `tests/`) runs without the raw CSVs.
The raw captures are only required to regenerate that sample from scratch with
`scripts/prepare_data.py`.
