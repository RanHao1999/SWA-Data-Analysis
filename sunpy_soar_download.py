"""
Author: Hao Ran
Institution: UCL / MSSL
Email: hao.ran.24@ucl.ac.uk
GitHub: @RanHao1999
Created: 2025-12-11

Description:
    This script is designed to download SOAR data from a specified source and save it to a local directory.

    Includes retry logic for flaky SOAR server connections (ContentLengthError,
    ChunkedEncodingError, timeouts, etc.).
"""

# library imports
from sunpy.net import Fido
from sunpy.net import attrs as a
import sunpy_soar

import os
import sys
import time
import random
import glob
import shutil
from datetime import datetime, timedelta

os.chdir(sys.path[0])

# === CONFIG ===
TIME_START = '2024-05-09'
TIME_END = '2024-05-09'
DATA_FOLDER = "data/SO/"
MAX_RETRIES = 5          # per-product download attempts
RETRY_DELAY_MIN = 10      # seconds — initial backoff
RETRY_DELAY_MAX = 300     # seconds — cap
# === END CONFIG ===


def days_between(start_date: str, end_date: str):
    """
    Return a list of YYYY-MM-DD strings for all days between start_date and end_date (inclusive).
    """
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    if end < start:
        raise ValueError("end_date must not be earlier than start_date.")

    delta = end - start
    return [(start + timedelta(days=i)).isoformat() for i in range(delta.days + 1)]


def _partial_files_for_query(query_result, data_path):
    """Return set of file paths that would be created by this query.

    Checks both the finished file and the parfive temporary ``.part`` file.
    """
    paths = set()
    if len(query_result) == 0:
        return paths
    for row_idx in range(len(query_result[0])):
        fname = str(query_result[0]['Filename'][row_idx])
        paths.add(os.path.join(data_path, fname))
        paths.add(os.path.join(data_path, fname + ".part"))
    return paths


def _clean_partials(query_result, data_path):
    """Remove any partial / temporary files associated with *query_result*."""
    for path in _partial_files_for_query(query_result, data_path):
        if os.path.exists(path):
            try:
                os.remove(path)
                print(f"    Cleaned partial: {os.path.basename(path)}")
            except OSError as e:
                print(f"    [WARN] Could not remove {path}: {e}")


def _missing_files(query_result, data_path):
    """Return list of (row_index, filename, full_path) for files not yet on disk."""
    missing = []
    if len(query_result) == 0:
        return missing
    for row_idx in range(len(query_result[0])):
        fname = str(query_result[0]['Filename'][row_idx])
        full = os.path.join(data_path, fname)
        if not os.path.exists(full):
            missing.append((row_idx, fname, full))
    return missing


def fetch_with_retry(query_result, data_path, product_name,
                     max_retries=MAX_RETRIES):
    """
    Download files from *query_result* with retry on failure.

    Retryable errors (ContentLengthError, ChunkedEncodingError, timeouts, etc.)
    trigger exponential backoff.  Each attempt cleans up partial files before
    retrying so the next attempt starts fresh.

    Returns True if all files were downloaded successfully, False otherwise.
    """
    if len(query_result) == 0:
        return True

    missing = _missing_files(query_result, data_path)
    if not missing:
        print(f"    [{product_name}] all files already present — skip.")
        return True

    print(f"    [{product_name}] {len(missing)} file(s) to download.")
    for _, fname, _ in missing:
        print(f"      → {fname}")

    for attempt in range(1, max_retries + 1):
        try:
            # Clean any partials from a previous failed attempt
            _clean_partials(query_result, data_path)

            Fido.fetch(query_result, path=data_path)

            # Check what we actually got
            still_missing = _missing_files(query_result, data_path)
            if not still_missing:
                print(f"    [{product_name}] OK (attempt {attempt})")
                return True
            else:
                # parfive didn't raise but also didn't deliver everything
                print(f"    [{product_name}] {len(still_missing)} file(s) "
                      f"not downloaded on attempt {attempt}.")
                for _, fname, _ in still_missing:
                    print(f"      MISSING: {fname}")

        except Exception as e:
            # parfive wraps underlying errors — catch everything and inspect
            err_msg = str(e)
            if "ContentLengthError" in err_msg:
                print(f"    [{product_name}] SOAR truncated response "
                      f"(attempt {attempt}/{max_retries})")
            elif "ChunkedEncodingError" in err_msg:
                print(f"    [{product_name}] SOAR chunked encoding broke "
                      f"(attempt {attempt}/{max_retries})")
            elif "timeout" in err_msg.lower() or "timed out" in err_msg.lower():
                print(f"    [{product_name}] SOAR timeout "
                      f"(attempt {attempt}/{max_retries})")
            else:
                print(f"    [{product_name}] download error "
                      f"(attempt {attempt}/{max_retries}): {e}")

        if attempt == max_retries:
            print(f"    [{product_name}] FAILED after {max_retries} attempts "
                  f"— giving up.")
            return False

        # Exponential backoff with jitter: 10, 20, 40, 80, 160 (capped)
        delay = min(RETRY_DELAY_MIN * (2 ** (attempt - 1)), RETRY_DELAY_MAX)
        delay *= random.uniform(0.5, 1.5)  # jitter to de-synchronise from other retries
        print(f"    [{product_name}] retrying in {delay:.0f}s...")
        time.sleep(delay)

    return False


def main():
    """Main execution function."""

    days_in_between = days_between(TIME_START, TIME_END)
    overall_ok = True

    for day in days_in_between:
        day_in_path = day.replace('-', '')
        data_path = f'{DATA_FOLDER}{day_in_path}/'
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        print(f"\n{'─'*50}")
        print(f"Day: {day}  →  {data_path}")
        print(f"{'─'*50}")

        SWA = a.Instrument('SWA')
        MAG = a.Instrument('MAG')
        time_attr = a.Time(day, day)

        # Each product is a (query, product_label) tuple.
        products = [
            (SWA & time_attr & a.soar.Product("swa-pas-grnd-mom") & a.Level(2),
             "grnd_mom"),
            (SWA & time_attr & a.soar.Product("swa-pas-vdf") & a.Level(2),
             "vdf"),
            (SWA & time_attr & a.soar.Product("swa-pas-3d") & a.Level(1),
             "counts"),
            (MAG & time_attr & a.soar.Product("mag-srf-normal") & a.Level(2),
             "mag"),
        ]

        for query_spec, label in products:
            # Small random jitter so the pipeline doesn't hammer SOAR at
            # perfectly regular intervals (some load-balancers penalize that).
            time.sleep(random.uniform(0.5, 3.0))

            try:
                query = Fido.search(query_spec)
            except Exception as e:
                print(f"  [{label}] search query failed: {e}")
                overall_ok = False
                continue

            if len(query) == 0 or len(query[0]) == 0:
                print(f"  [{label}] no data found for {day}")
                continue

            ok = fetch_with_retry(query, data_path, label)
            if not ok:
                overall_ok = False

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
