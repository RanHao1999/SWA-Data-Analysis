#!/usr/bin/env python3
"""
Pipeline Controller — master orchestrator for Alpha Processing.

This script edits config blocks inside each sub-script and executes them
as independent subprocesses.  Data passes between steps via the filesystem
(data/SO/{day}/ → result/SO/{day}/ → result/SO/VDFs/).

Workflow per day:
  1. Download SOAR data      (sunpy_soar_download.py)
  2. GMM auto-parallelised    (gmm_auto_parallelised.py)
  3. Save sparse VDFs to HDF5 (Save_vdfs.py)
  4. Delete raw + intermediate (Delete_files.py)

Usage:
    conda activate research_env
    python /disk/plasma/hr2/Alpha_Processing/pipeline_controller.py

Author: Hao Ran
Created: 2025-06-30
"""

import os
import re
import sys
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

# Work relative to this script's own directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)


# =========================================================================
# Config update helpers
# =========================================================================

def _format_config_value(key, value, original_line):
    """Produce a 'KEY = <formatted_value>' line, preserving indentation."""
    indent = original_line[:len(original_line) - len(original_line.lstrip())]

    if isinstance(value, bool):
        return f"{indent}{key} = {value}\n"
    elif isinstance(value, (int, float)):
        return f"{indent}{key} = {value}\n"
    elif value is None:
        return f"{indent}{key} = None\n"
    elif isinstance(value, str):
        escaped = value.replace("'", "\\'")
        return f"{indent}{key} = '{escaped}'\n"
    else:
        return f"{indent}{key} = {value!r}\n"


def update_config(script_name, **kwargs):
    """Edit the # === CONFIG === block in *script_name* in-place.

    Each kwarg key is matched against config-block lines of the form
    ``KEY = ...``.  Only the first matching line per key is replaced.
    Raises ValueError if the script has no CONFIG block.
    """
    path = os.path.join(SCRIPT_DIR, script_name)
    with open(path) as f:
        lines = f.readlines()

    in_config = False
    block_found = False
    updated = set()
    new_lines = []

    for line in lines:
        stripped = line.strip()

        if stripped == '# === CONFIG ===':
            in_config = True
            block_found = True
            new_lines.append(line)
            continue
        if stripped == '# === END CONFIG ===':
            in_config = False
            new_lines.append(line)
            continue

        if in_config:
            matched = False
            for key, value in kwargs.items():
                if re.match(r'^\s*' + re.escape(key) + r'\s*=', line):
                    new_lines.append(_format_config_value(key, value, line))
                    updated.add(key)
                    matched = True
                    break
            if not matched:
                new_lines.append(line)
        else:
            new_lines.append(line)

    if not block_found:
        raise ValueError(f"{script_name}: no # === CONFIG === block found")

    missing = set(kwargs) - updated
    if missing:
        print(f"  [WARN] {script_name}: keys not found in config: {missing}")

    with open(path, 'w') as f:
        f.writelines(new_lines)


# =========================================================================
# Subprocess launchers
# =========================================================================

def _blas_free_env():
    """Return an environment dict that pins BLAS/OpenMP to 1 thread.

    Without this, each worker inside a ProcessPoolExecutor also tries to
    grab all cores for its own BLAS calls, causing
    N_processes × N_cores oversubscription.
    """
    env = os.environ.copy()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        env[var] = "1"
    return env


def run_script(script_name, desc, **cfg):
    """Update config and run *script_name* synchronously.

    Returns the subprocess exit code.
    """
    print(f"\n  [{desc}] Starting...")
    update_config(script_name, **cfg)
    result = subprocess.run(
        [sys.executable, script_name],
        cwd=SCRIPT_DIR,
        env=_blas_free_env(),
    )
    if result.returncode != 0:
        print(f"  [{desc}] FAILED (exit code {result.returncode})")
    else:
        print(f"  [{desc}] OK")
    return result.returncode


def launch_script(script_name, desc, **cfg):
    """Update config and launch *script_name* as a background subprocess.

    Returns the ``subprocess.Popen`` object so the caller can ``.wait()``.
    """
    print(f"\n  [{desc}] Launching in background...")
    update_config(script_name, **cfg)
    proc = subprocess.Popen(
        [sys.executable, script_name],
        cwd=SCRIPT_DIR,
        env=_blas_free_env(),
    )
    return proc


# =========================================================================
# Date utilities
# =========================================================================

def days_between(t_start, t_end):
    """Return list of YYYYMMDD strings between two datetimes (inclusive)."""
    d0 = t_start.date() if isinstance(t_start, datetime) else t_start
    d1 = t_end.date() if isinstance(t_end, datetime) else t_end
    if d0 > d1:
        return []
    result = []
    current = d0
    while current <= d1:
        result.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return result


def _to_iso_date(yymmdd):
    """Convert 'YYYYMMDD' → 'YYYY-MM-DD' (format sunpy_soar_download expects)."""
    return f"{yymmdd[:4]}-{yymmdd[4:6]}-{yymmdd[6:8]}"


def _data_ready(yymmdd):
    """Return True if all required CDF files exist for *yymmdd*."""
    data_dir = os.path.join(SCRIPT_DIR, "data", "SO", yymmdd)
    if not os.path.isdir(data_dir):
        return False
    files = os.listdir(data_dir)
    required = ['pas-vdf', 'pas-grnd-mom', 'pas-3d', 'mag-srf-normal']
    for pattern in required:
        if not any(pattern in f and not f.startswith('._') for f in files):
            return False
    return True

# =========================================================================
# Main pipeline
# =========================================================================

def main():
    # =====================================================================
    # USER SETTINGS — edit these
    # =====================================================================
    PIPELINE_TSTART = datetime(2024, 9, 13, 3, 6, 47)
    PIPELINE_TEND   = datetime(2024, 12, 31, 23, 59, 59)

    DT_WANTED   = 4.0   # desired output cadence (s)
    N_PROCESSES = 30    # parallel workers for GMM fitting
    _DELETE     = True  # False → keep raw data & intermediate products (skip cleanup)
    # =====================================================================

    days = days_between(PIPELINE_TSTART, PIPELINE_TEND)
    if not days:
        print("No days to process — check PIPELINE_TSTART/PIPELINE_TEND.")
        return 1

    total_tstart = time.time()
    print(f"Pipeline: {len(days)} day(s)  |  "
          f"{PIPELINE_TSTART} → {PIPELINE_TEND}")
    print(f"Parameters:  dt={DT_WANTED}s, workers={N_PROCESSES}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Pre-download the very first day (must finish before processing)
    # ------------------------------------------------------------------
    print(f"\n{'─'*60}")
    print(f"Pre-downloading day 1: {days[0]}")
    run_script(
        "sunpy_soar_download.py",
        f"download {days[0]}",
        TIME_START=_to_iso_date(days[0]),
        TIME_END=_to_iso_date(days[0]),
    )
    # Note: download may fail (datagap / server error).  _data_ready()
    # will catch it on the first iteration and skip the day.

    download_pool = ThreadPoolExecutor(max_workers=1)
    next_dl_future = None
    bg_dl_failed = False      # did the background download for *this* day fail?
    bg_dl_day = None          # which day was pre-fetched?

    for i, day in enumerate(days):
        print(f"\n{'='*60}")
        print(f"DAY {i+1}/{len(days)}: {day}")
        print(f"{'='*60}")

        # --------------------------------------------------------------
        # If the background pre-fetch for THIS day failed, re-download now
        # --------------------------------------------------------------
        if bg_dl_failed and bg_dl_day == day:
            print(f"\n  ⚠ Background download for {day} failed earlier — "
                  f"re-downloading synchronously.")
            rc = run_script(
                "sunpy_soar_download.py",
                f"download {day} (retry)",
                TIME_START=_to_iso_date(day),
                TIME_END=_to_iso_date(day),
            )
            if rc != 0:
                print(f"  [FATAL] Re-download of {day} also failed — "
                      f"skipping day.")
                bg_dl_failed = False
                bg_dl_day = None
                # Still kick off next day's pre-fetch so pipeline can continue.
                if i + 1 < len(days):
                    next_day = days[i + 1]
                    print(f"\n  ▶ Pre-fetching day {i+2}: {next_day} (background)")
                    next_dl_future = download_pool.submit(
                        launch_script,
                        "sunpy_soar_download.py",
                        f"download {next_day} (bg)",
                        TIME_START=_to_iso_date(next_day),
                        TIME_END=_to_iso_date(next_day),
                    )
                    bg_dl_day = next_day
                continue
            bg_dl_failed = False
            bg_dl_day = None

        # --------------------------------------------------------------
        # Kick off download of NEXT day in background
        # --------------------------------------------------------------
        if i + 1 < len(days):
            next_day = days[i + 1]
            print(f"\n  ▶ Pre-fetching day {i+2}: {next_day} (background)")
            next_dl_future = download_pool.submit(
                launch_script,
                "sunpy_soar_download.py",
                f"download {next_day} (bg)",
                TIME_START=_to_iso_date(next_day),
                TIME_END=_to_iso_date(next_day),
            )
            bg_dl_day = next_day

        # --------------------------------------------------------------
        # Skip day if required data files are missing (datagap / download failure)
        # --------------------------------------------------------------
        if not _data_ready(day):
            print(f"\n  [SKIP] {day}: data missing (datagap or download failed) "
                  f"— moving to next day.")
            # Still wait for in-flight background download before moving on.
            if next_dl_future is not None:
                try:
                    proc = next_dl_future.result()
                    ret = proc.wait()
                    bg_dl_failed = (ret != 0)
                except Exception as e:
                    print(f"  [ERROR] Background download crashed: {e}")
                    bg_dl_failed = True
                next_dl_future = None
            continue

        # --------------------------------------------------------------
        # GMM process current day (foreground)
        # --------------------------------------------------------------
        start_iso = PIPELINE_TSTART.strftime("%Y-%m-%d %H:%M:%S") if i == 0 else _to_iso_date(day) + " 00:00:00"
        end_iso   = _to_iso_date(day) + " 23:59:59"

        rc = run_script(
            "gmm_auto_parallelised.py",
            f"GMM {day}",
            YYMMDD=day,
            T_START_ISO=start_iso,
            T_END_ISO=end_iso,
            DT_WANTED=DT_WANTED,
            N_PROCESSES=N_PROCESSES,
            _PLOT=False,
        )

        if rc != 0:
            print(f"  [{day}] GMM failed — skipping VDF save and cleanup.")
            # Still wait for in-flight download before moving to next day.
            if next_dl_future is not None:
                try:
                    proc = next_dl_future.result()
                    ret = proc.wait()
                    bg_dl_failed = (ret != 0)
                    if bg_dl_failed:
                        print(f"  [WARN] Background download failed "
                              f"(exit {ret})")
                except Exception as e:
                    print(f"  [ERROR] Background download crashed: {e}")
                    bg_dl_failed = True
                next_dl_future = None
            continue

        # --------------------------------------------------------------
        # Wait for next day's background download to finish
        # --------------------------------------------------------------
        if next_dl_future is not None:
            print(f"\n  ⏳ Waiting for next-day download to finish...")
            try:
                proc = next_dl_future.result()
                ret = proc.wait()
                bg_dl_failed = (ret != 0)
                if bg_dl_failed:
                    print(f"  [WARN] Background download for day "
                          f"{days[i+1]} failed (exit {ret}) — "
                          f"will re-download when its turn comes.")
            except Exception as e:
                print(f"  [ERROR] Background download crashed: {e}")
                bg_dl_failed = True
            next_dl_future = None

        # --------------------------------------------------------------
        # Save VDFs + cleanup (skip if _DELETE=False)
        # --------------------------------------------------------------
        if _DELETE:
            run_script(
                "Save_vdfs.py",
                f"save VDFs {day}",
                DAY=day,
            )

            run_script(
                "Delete_files.py",
                f"cleanup {day}",
                DAY_BEGIN=day,
                DAY_END=day,
            )
        else:
            print(f"  [{day}] _DELETE=False — keeping raw data & intermediates.")

    download_pool.shutdown(wait=False)

    total_tend = time.time()
    elapsed = total_tend - total_tstart
    print(f"\n{'='*60}")
    print(f"Pipeline finished.  Total time: {elapsed/60:.1f} min "
          f"({elapsed/3600:.2f} hr)")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
