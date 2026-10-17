"""
Package V5_REVIEW.zip with strict exclusions:
Exclude raw datasets, virtual environments, pycache, git, credentials, and prior zip archives.
"""

import os
import zipfile
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("package_v5_review")

EXCLUDE_DIRS = {
    "dataset",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".git",
    ".idea",
    ".vscode",
}

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".zip",
}

EXCLUDE_FILES = {
    "Base_Paper.pdf",
    "base_paper_extracted.txt",
}


def package_v5_review(output_zip: str = "V5_REVIEW.zip"):
    root_dir = os.path.abspath(".")
    logger.info(f"Packaging {output_zip} from {root_dir}...")

    file_count = 0
    total_uncompressed_bytes = 0

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Prune excluded directories in place
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith(".")]

            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext in EXCLUDE_EXTENSIONS:
                    continue
                if fname in EXCLUDE_FILES:
                    continue

                full_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(full_path, root_dir)

                # Skip raw dataset csvs if any leaked into data/
                if "dataset" in rel_path.split(os.sep):
                    continue

                zf.write(full_path, rel_path)
                file_count += 1
                total_uncompressed_bytes += os.path.getsize(full_path)

    zip_size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    logger.info(f"Successfully created {output_zip} ({file_count} files, {zip_size_mb:.2f} MB)")
    return output_zip


if __name__ == "__main__":
    package_v5_review()
