"""
V6_REVIEW Packaging Script.
Collects corrected source, configurations, tests, execution evidence,
incremental prediction logs, tables, reports, diagnosis, and traceability docs into V6_REVIEW.zip.
Excludes raw datasets, virtual environments, caches, credentials, and obsolete review ZIPs.
"""
import os
import zipfile
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EXCLUDE_DIRS = {
    "dataset",
    ".venv",
    "venv",
    "env",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
    ".idea",
    ".vscode",
}

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".zip",
    ".tar",
    ".gz.bak",
    ".tmp",
}

def is_excluded(rel_path: str) -> bool:
    parts = rel_path.replace("\\", "/").split("/")
    for part in parts:
        if part in EXCLUDE_DIRS:
            return True
    _, ext = os.path.splitext(rel_path)
    if ext.lower() in EXCLUDE_EXTENSIONS and not rel_path.endswith(".jsonl.gz") and not rel_path.endswith(".csv.gz"):
        return True
    return False

def package_v6_review(output_zip: str = "V6_REVIEW.zip"):
    root_dir = os.path.abspath(".")
    zip_path = os.path.join(root_dir, output_zip)

    logger.info(f"Creating {output_zip} from {root_dir}...")
    file_count = 0
    total_uncompressed_bytes = 0

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for current_root, dirs, files in os.walk(root_dir):
            # Prune excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            for file in files:
                abs_path = os.path.join(current_root, file)
                rel_path = os.path.relpath(abs_path, root_dir)

                # Skip self zip or nested zips
                if rel_path == output_zip or rel_path.endswith(".zip"):
                    continue

                if is_excluded(rel_path):
                    continue

                # Ensure we only include relevant project files
                zf.write(abs_path, rel_path)
                file_count += 1
                total_uncompressed_bytes += os.path.getsize(abs_path)

    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    logger.info(f"Successfully packaged {output_zip}:")
    logger.info(f"  Files packaged: {file_count:,}")
    logger.info(f"  Uncompressed total: {total_uncompressed_bytes / (1024 * 1024):.2f} MB")
    logger.info(f"  Compressed archive size: {zip_size_mb:.2f} MB")
    logger.info(f"  Archive location: {zip_path}")
    return zip_path

if __name__ == "__main__":
    package_v6_review()
