# Safe Project Cleanup Manifest

This manifest documents all files and directories moved to the obsolete archive or removed during the V6 research-review finalization.

## 1. Archived Items (Preserved in rchive_obsolete/)
These items were preserved to maintain historical diagnostic evidence without cluttering the active project structure or final release bundle.

| Original Path | Archived Location | Reason / Purpose |
| :--- | :--- | :--- |
| V5_REVIEW.zip | archive_obsolete\V5_REVIEW.zip | Old version review zip; superseded by FINAL_PROJECT.zip |
| V6_REVIEW.zip | archive_obsolete\V6_REVIEW.zip | Old version review zip; superseded by FINAL_PROJECT.zip |
| paper | archive_obsolete\paper | Draft IEEE paper; paper writing postponed per project instructions |
| scripts/compile_paper.py | archive_obsolete\compile_paper.py | Paper compilation script for postponed draft |
| scripts/package_v5_review.py | archive_obsolete\package_v5_review.py | Old review packaging script |
| scripts/package_v6_review.py | archive_obsolete\package_v6_review.py | Old review packaging script |
| reports/test_run_light | archive_obsolete\test_run_light | Old temporary or invalidated experimental outputs |
| reports/test_run_gen | archive_obsolete\test_run_gen | Old temporary or invalidated experimental outputs |
| reports/test_queue_bound | archive_obsolete\test_queue_bound | Old temporary or invalidated experimental outputs |
| reports/verification | archive_obsolete\verification | Old temporary or invalidated experimental outputs |
| reports/archive_preliminary_invalid | archive_obsolete\archive_preliminary_invalid | Old temporary or invalidated experimental outputs |
| reports/archive_v3_preliminary | archive_obsolete\archive_v3_preliminary | Old temporary or invalidated experimental outputs |

## 2. Deleted Disposable Items
These items consisted strictly of compiled bytecode and ephemeral test caches that can be deterministically regenerated.

| Path Removed | Reason / Purpose |
| :--- | :--- |
| .\.pytest_cache | Disposable Python/pytest bytecode cache |
| .\scripts\__pycache__ | Disposable Python/pytest bytecode cache |
| .\src\__pycache__ | Disposable Python/pytest bytecode cache |
| .\src\api\__pycache__ | Disposable Python/pytest bytecode cache |
| .\src\blockchain\__pycache__ | Disposable Python/pytest bytecode cache |
| .\src\data\__pycache__ | Disposable Python/pytest bytecode cache |
| .\src\drift\__pycache__ | Disposable Python/pytest bytecode cache |
| .\src\dst\__pycache__ | Disposable Python/pytest bytecode cache |
| .\src\evaluation\__pycache__ | Disposable Python/pytest bytecode cache |
| .\src\features\__pycache__ | Disposable Python/pytest bytecode cache |
| .\src\models\__pycache__ | Disposable Python/pytest bytecode cache |
| .\src\risk\__pycache__ | Disposable Python/pytest bytecode cache |
| .\src\simulation\__pycache__ | Disposable Python/pytest bytecode cache |
| .\src\utils\__pycache__ | Disposable Python/pytest bytecode cache |
| .\tests\__pycache__ | Disposable Python/pytest bytecode cache |

## 3. Explicitly Preserved Items
- **Raw Datasets**: dataset/ (all 8 original raw CSV files, ~1 GB) preserved untouched.
- **Processed Data**: data/processed/ (lightweight_sample.csv.gz and stratified_sample.csv.gz).
- **Base Paper**: Base_Paper.pdf and BASE_PAPER_ANALYSIS.md preserved intact.
- **Reference Documentation**: REGRESSION_DIAGNOSIS.md, TRACEABILITY_MATRIX.md, VALIDITY_AUDIT.md.
- **Active Source Modules**: All core code in src/, scripts/, configs/, 	ests/, dashboard/.
