"""Automated downloader and safe extractor for the UCI Online Retail II dataset.

Dataset: Online Retail II (UCI Machine Learning Repository, ID: 502)
Archive URL: https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip
"""

from __future__ import annotations

import argparse
import logging
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# Configure module-level logger for status messages
LOGGER = logging.getLogger(__name__)

# Official UCI Machine Learning Repository download URL for Online Retail II
UCI_DATASET_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"

# Default directory where raw data files are saved (relative to project root)
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ARCHIVE_FILENAME = "online_retail_ii.zip"
SUPPORTED_EXTENSIONS = (".xlsx", ".xls", ".csv")


def find_existing_dataset(data_dir: Path = DEFAULT_DATA_DIR) -> Path | None:
    """Check if an extracted dataset file (.xlsx, .xls, or .csv) already exists."""
    if not data_dir.exists():
        return None

    # Check common UCI filenames first in a predictable order
    preferred_names = (
        "online_retail_II.xlsx",
        "online_retail_ii.xlsx",
        "online_retail_II.csv",
        "online_retail_ii.csv",
    )
    for filename in preferred_names:
        candidate = data_dir / filename
        if candidate.is_file():
            return candidate

    # Fallback: check for any file with a supported tabular extension
    for ext in SUPPORTED_EXTENSIONS:
        matches = sorted(data_dir.glob(f"*{ext}"))
        for match in matches:
            if match.is_file():
                return match

    return None


def find_existing_archive(data_dir: Path = DEFAULT_DATA_DIR) -> Path | None:
    """Check if a local ZIP archive of the dataset already exists in data_dir."""
    if not data_dir.exists():
        return None

    for filename in (ARCHIVE_FILENAME, "online+retail+ii.zip"):
        candidate = data_dir / filename
        if candidate.is_file():
            return candidate

    zip_files = sorted(data_dir.glob("*.zip"))
    for zip_path in zip_files:
        if zip_path.is_file():
            return zip_path

    return None


def safe_extract_zip(archive_path: Path, destination_dir: Path) -> list[Path]:
    """Safely extract a ZIP archive while preventing directory traversal attacks."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    resolved_destination = destination_dir.resolve()
    extracted_files: list[Path] = []

    with zipfile.ZipFile(archive_path, mode="r") as zip_handle:
        for member in zip_handle.infolist():
            # Verify that the extracted file path remains inside destination_dir
            target_path = (destination_dir / member.filename).resolve()
            if not target_path.is_relative_to(resolved_destination):
                raise ValueError(
                    f"Unsafe file path inside ZIP archive: {member.filename!r}"
                )

            zip_handle.extract(member, path=destination_dir)
            if not member.is_dir():
                extracted_files.append(target_path)

    return extracted_files


def download_archive(
    url: str = UCI_DATASET_URL,
    archive_path: Path | None = None,
    chunk_size: int = 65536,
    timeout: int = 60,
) -> Path:
    """Stream-download the dataset ZIP archive to disk with a progress bar."""
    if archive_path is None:
        archive_path = DEFAULT_DATA_DIR / ARCHIVE_FILENAME

    archive_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temporary '.part' file first so interrupted downloads are never mistaken for complete files
    temp_path = archive_path.with_suffix(archive_path.suffix + ".part")

    LOGGER.info("Downloading Online Retail II archive from %s ...", url)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        total_bytes = int(response.headers.get("content-length", 0))

        with open(temp_path, mode="wb") as file_handle:
            progress_bar = tqdm(
                total=total_bytes if total_bytes > 0 else None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=archive_path.name,
            )
            try:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        file_handle.write(chunk)
                        progress_bar.update(len(chunk))
            finally:
                progress_bar.close()

    # Atomically rename the completed download to the final archive filename
    temp_path.replace(archive_path)
    LOGGER.info("Saved archive to %s", archive_path)
    return archive_path


def download_dataset(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    url: str = UCI_DATASET_URL,
    force: bool = False,
) -> Path:
    """Download and safely extract the UCI Online Retail II archive into ``data_dir``.

    Skips downloading if an extracted dataset file or local archive already exists,
    unless ``force=True`` is specified.
    """
    target_dir = Path(data_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Return immediately if an extracted dataset is already present
    if not force:
        existing_dataset = find_existing_dataset(target_dir)
        if existing_dataset is not None:
            LOGGER.info(
                "Dataset already exists at %s. Skipping download.", existing_dataset
            )
            return existing_dataset

    # Step 2: Check for an existing ZIP archive before downloading over the network
    archive_path = find_existing_archive(target_dir)
    if force or archive_path is None:
        archive_path = download_archive(
            url=url,
            archive_path=target_dir / ARCHIVE_FILENAME,
        )
    else:
        LOGGER.info(
            "Found local archive at %s. Skipping network download.", archive_path
        )

    # Step 3: Extract the ZIP archive into data_dir
    LOGGER.info("Extracting %s into %s ...", archive_path.name, target_dir)
    extracted = safe_extract_zip(archive_path, target_dir)
    LOGGER.info("Extracted %d file(s).", len(extracted))

    # Step 4: Locate and return the extracted Excel or CSV dataset file
    dataset_path = find_existing_dataset(target_dir)
    if dataset_path is None:
        raise FileNotFoundError(
            f"Archive was extracted into {target_dir}, but no .xlsx/.xls/.csv file was found."
        )

    LOGGER.info("Online Retail II dataset is ready at %s", dataset_path)
    return dataset_path


def parse_args() -> argparse.Namespace:
    """Parse command-line options for running the download script directly."""
    parser = argparse.ArgumentParser(
        description="Download and extract the UCI Online Retail II dataset (ID: 502)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Target directory for storing the dataset (default: data/).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-extract even if local files already exist.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    cli_args = parse_args()
    output_path = download_dataset(data_dir=cli_args.data_dir, force=cli_args.force)
    print(f"Dataset ready: {output_path}")
