"""
Packaging script for Adaptive ML-DST V6 Final Release.
Creates FINAL_PROJECT.zip containing:
- Clean source code (src/)
- Production scripts and entry points (scripts/)
- Model checkpoints and metadata (models/)
- Production configurations (configs/)
- Final reports, metrics, manifests, and verification logs (reports/final/)
- Test suite (tests/)
- Research documentation (docs/)
- Example inference dataset and README (data/)
- Web dashboard and contracts
- Root README, requirements.txt, pyproject.toml

Strictly excludes:
- Raw datasets (dataset/)
- Virtual environments (.venv, venv)
- Pytest and Python caches (__pycache__, .pytest_cache)
- Obsolete archives (archive_obsolete/)
- Nested ZIP archives (*.zip)
- Environment secret files (.env)
"""

import os
import zipfile
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EXCLUDE_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    ".venv",
    "venv",
    "env",
    "dataset",
    "archive_obsolete",
    ".system_generated",
    "scratch",
}

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".zip",
}

EXCLUDE_FILES = {
    ".env",
    ".DS_Store",
    "Thumbs.db",
}


def should_include_path(rel_path: str) -> bool:
    """Determines whether a file relative path should be included in the release zip."""
    parts = rel_path.replace("\\", "/").split("/")

    # Check excluded directory names in path
    for part in parts[:-1]:
        if part in EXCLUDE_DIRS:
            return False

    filename = parts[-1]
    if filename in EXCLUDE_FILES:
        return False

    # Exclude temporary or raw processed files if large
    if len(parts) > 1 and parts[0] == "data" and parts[1] == "processed":
        # Keep directory README or example but exclude large compressed caches from zip
        if filename.endswith(".csv.gz") or filename.endswith(".db"):
            return False

    # Exclude non-final reports
    if parts[0] == "reports" and len(parts) > 1 and parts[1] != "final":
        return False

    _, ext = os.path.splitext(filename)
    if ext.lower() in EXCLUDE_EXTENSIONS:
        return False

    return True


def package_release(output_zip: str = "FINAL_PROJECT.zip") -> str:
    """Packages the verified project into output_zip."""
    root_dir = os.getcwd()
    abs_output = os.path.abspath(output_zip)

    logger.info(f"Packaging final release archive to: {abs_output}")
    included_count = 0
    total_uncompressed_bytes = 0

    # Ensure output zip is removed first if already existing
    if os.path.exists(abs_output):
        try:
            os.remove(abs_output)
        except Exception as e:
            logger.warning(f"Could not remove existing {abs_output}: {e}")

    with zipfile.ZipFile(abs_output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Prune excluded directories in-place to avoid descending into them
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]

            for fname in filenames:
                full_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(full_path, root_dir)

                # Skip output zip itself
                if os.path.abspath(full_path) == abs_output:
                    continue

                if should_include_path(rel_path):
                    arcname = os.path.join("Adaptive_ML_DST_V6", rel_path)
                    zipf.write(full_path, arcname)
                    included_count += 1
                    total_uncompressed_bytes += os.path.getsize(full_path)

    zip_size_bytes = os.path.getsize(abs_output)
    logger.info("=" * 70)
    logger.info(f"SUCCESS: Packaged {included_count} files into {output_zip}")
    logger.info(f"Uncompressed size: {total_uncompressed_bytes / (1024*1024):.2f} MB")
    logger.info(f"Compressed ZIP size: {zip_size_bytes / (1024*1024):.2f} MB ({zip_size_bytes} bytes)")
    logger.info("=" * 70)
    return abs_output


if __name__ == "__main__":
    package_release()
